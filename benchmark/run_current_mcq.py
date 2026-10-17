"""MCQ da configuração AWS atual, sem depender de um benchmark de retrieval."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
from time import perf_counter
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from benchmark.matrix import AWS_MCQ_MODELS, SPLIT_SEED
from benchmark.prepare import build_split, validate_benchmark
from benchmark import report_mcq_html as dashboard
from benchmark.run_mcq import (
    MCQ_PROMPT_VERSION,
    MCQ_SYSTEM_PROMPT,
    MCQStructuredResponse,
    build_human_prompt,
    classify_selected_option,
    extract_usage,
    validate_mcq_rows,
)
from scripts.cloud import require_operations_enabled
from src.config.config_manager import ConfigManager
from src.generation.context_builder import build_context, build_context_with_enrichment
from src.generation.provider import create_generation_provider

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "mcq/current/runs"


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_json(s3, bucket, key, *, optional=False):
    if optional:
        listed = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if not any(item["Key"] == key for item in listed.get("Contents", [])):
            return None
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def write_json(s3, bucket, key, value):
    s3.put_object(
        Bucket=bucket, Key=key, Body=json_bytes(value), ContentType="application/json"
    )


def load_questions(payload, split, limit):
    if split not in {"dev", "holdout"} or not 0 <= limit <= 99999:
        raise ValueError("Split ou limite de questões inválido.")
    source = pd.read_csv(io.BytesIO(payload)).fillna("")
    validate_benchmark(source)
    prepared = build_split(source)
    validate_mcq_rows(prepared)
    selected = prepared[prepared["split"].eq(split)].copy()
    if limit:
        selected = selected.head(limit)
    if selected.empty:
        raise ValueError("O split escolhido não possui questões.")
    return prepared, selected


def load_aws_config(out, s3, ssm, previous=None):
    """Usa a mesma release da API e apenas a credencial Qdrant de leitura."""
    buckets = out["buckets"]
    selected = previous["data_release_key"] if previous else "production/current.json"
    release = read_json(s3, buckets["EMBEDDING_INDEXES_BUCKET"], selected)
    release_id = str(release.get("run_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", release_id):
        raise ValueError(
            "Release de dados inválida; publique o ETL antes do benchmark."
        )
    key = f"production/releases/{release_id}.json"
    frozen = read_json(s3, buckets["EMBEDDING_INDEXES_BUCKET"], key)
    if frozen != release:
        raise ValueError("O ponteiro publicado diverge da release imutável.")
    settings = json.loads(
        ssm.get_parameter(Name=out["app_settings_parameter"], WithDecryption=True)[
            "Parameter"
        ]["Value"]
    )
    if not settings.get("QDRANT_URL", "").startswith("https://") or not settings.get(
        "QDRANT_API_KEY"
    ):
        raise ValueError("Configuração Qdrant da aplicação incompleta.")
    os.environ.update({name: str(value) for name, value in buckets.items()})
    os.environ.update(
        STORAGE_PROVIDER="s3",
        AWS_REGION=out["region"],
        DATA_RELEASE_KEY=key,
        QDRANT_URL=settings["QDRANT_URL"],
        QDRANT_API_KEY=settings["QDRANT_API_KEY"],
    )
    config = ConfigManager.load(
        env_file=ROOT / "config/absent.env", yaml_file=ROOT / "config/config.aws.yaml"
    )
    return config, key, digest(frozen)


def public_config(config):
    return {
        name: getattr(config, name).model_dump(mode="json")
        for name in (
            "embedding",
            "retrievals",
            "retrieval_index",
            "enrichment_retrieval",
            "generation",
        )
    } | {"retrieval_strategy": config.retrieval_strategy}


def select_model(config, choice):
    if choice is None:
        return config
    if choice not in AWS_MCQ_MODELS:
        raise ValueError("Selecione qwen3-32b ou llama4-scout.")
    preset = AWS_MCQ_MODELS[choice]
    selected = config.model_copy(deep=True)
    selected.generation = type(config.generation).model_validate(
        config.generation.model_dump()
        | {key: preset[key] for key in ("provider", "model")}
    )
    return selected


def run_metadata(config, state):
    generation = config["effective_config"]["generation"]
    model = generation["model"]
    label = next(
        (item["label"] for item in AWS_MCQ_MODELS.values() if item["model"] == model),
        model,
    )
    revision = str(config["revision"])
    comparable = {
        key: value
        for key, value in config.items()
        if key not in {"revision", "effective_config"}
    }
    comparable["effective_config"] = json.loads(json.dumps(config["effective_config"]))
    comparable["effective_config"]["generation"] = {
        key: value
        for key, value in generation.items()
        if key
        not in {
            "model",
            "provider",
            "base_url",
            "model_prices",
            "input_usd_per_million",
            "output_usd_per_million",
        }
    }
    release = config["data_release_key"].rsplit("/", 1)[-1].removesuffix(".json")
    return {
        "run_id": state["run_id"],
        "revision": revision,
        "model": model,
        "model_label": label,
        "provider": generation["provider"],
        "split": config["split"],
        "run_label": f'Benchmark {label} · commit {revision[:8]} · {config["split"].upper()}',
        "comparison_id": digest(comparable),
        "comparison_label": f'{release} · {state["summary"]["expected"]} questões · {config["effective_config"]["retrieval_strategy"]}',
        "status": state["status"],
        "updated_at": state["updated_at"],
        "summary": state["summary"],
    }


def report_frame(config, state, records, questions):
    """Adapta o checkpoint AWS ao contrato do dashboard original, sem perder colunas."""
    question_map = {str(row.query_id): row for row in questions.itertuples(index=False)}
    generation = config["effective_config"]["generation"]
    config_id = "aws-current"
    rows = []
    for row in records.values():
        question = question_map.get(str(row["query_id"]))
        incoming = row.get("total_input_tokens", row["input_tokens"])
        outgoing = row.get("total_output_tokens", row["output_tokens"])
        generation_ms = row.get("generation_ms")
        attempts = row.get("attempts", [])
        flat = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        rows.append(
            flat
            | {
                "run_id": state["run_id"],
                "revision": config["revision"],
                "config_id": config_id,
                "selection_rank": 1,
                "consensus_score": None,
                "context_id": str(row["query_id"]),
                "model": generation["model"],
                "query_type": getattr(question, "query_type", ""),
                "answer_text": row["answer"],
                "is_correct": row["selected_class"] == "correct",
                "is_ambiguous": row["selected_class"] == "ambiguous",
                "is_wrong": row["selected_class"] == "wrong",
                "is_invalid": row["selected_class"] == "invalid",
                "generation_latency_ms": generation_ms,
                "input_tokens": incoming,
                "output_tokens": outgoing,
                "total_tokens": incoming + outgoing,
                "tokens_per_second": (
                    row["output_tokens"] / (generation_ms / 1000)
                    if generation_ms
                    else None
                ),
                "generation_attempts": sum(item["calls"] for item in attempts),
                "context_chars": len(row.get("context", "")),
                "used_evidence_count": len(row.get("evidence_ids", [])),
                "generation_error_type": row.get("error_type", ""),
                "generation_error_message": row.get("diagnostics", {}).get(
                    "reason", ""
                ),
            }
        )
    return pd.DataFrame(rows).reindex(
        columns=list(
            dict.fromkeys(
                [
                    *dashboard.RESULT_COLUMNS,
                    "run_id",
                    "revision",
                    "status",
                    "latency_ms",
                    "estimated_cost_usd",
                    *[key for row in rows for key in row],
                ]
            )
        )
    )


def render_report(config, state, records, questions):
    frame = report_frame(config, state, records, questions)
    effective = config["effective_config"]
    generation = effective["generation"]
    return dashboard.render_html(
        run_id=state["run_id"],
        run_manifest=state,
        run_config={
            "split": config["split"],
            "benchmark_sha256": config["benchmark_sha256"],
            "prompt": {"version": config["prompt_version"]},
            "source_retrieval_run_id": config["data_release_key"],
            "generation_overrides": generation,
            "models": [{"name": generation["model"]}],
        },
        results=frame,
        questions=questions,
        configs=[
            {
                "config_id": "aws-current",
                "selection_rank": 1,
                "label": f'{effective["retrieval_strategy"]} · {effective["embedding"]["selected_model"]} · release publicada',
            }
        ],
        provenance=run_metadata(config, state) | {"config": config},
    )


class CurrentMCQPipeline:
    def __init__(self, config):
        from src.retrieval.service import RetrievalService
        from src.retrieval.enrichment import EnrichmentRetriever

        self.config = config
        self.retrieval = RetrievalService(config)
        self.enrichment = None
        self.enrichment_error = None
        if config.enrichment_retrieval.enabled:
            try:
                self.enrichment = EnrichmentRetriever(config)
            except Exception as exc:
                if not config.enrichment_retrieval.fail_open:
                    raise
                self.enrichment_error = type(exc).__name__
        self.provider = create_generation_provider(config.generation)

    def evaluate(self, question):
        started = perf_counter()
        snapshot = getattr(self.provider, "usage_snapshot", None)
        before = snapshot() if snapshot else None
        raw = None
        result = {
            "query_id": str(question.query_id),
            "query": str(question.query_rewritten),
            "canonical_work": str(question.canonical_work),
            "difficulty": str(question.difficulty),
            "correct_option": str(question.correct_option),
            "status": "error",
            "selected_class": "invalid",
            "selected_option": None,
            "answer": "",
            "fatal": False,
            "context": "",
            "enrichment_error_type": self.enrichment_error,
        }
        try:
            retrieved = self.retrieval.search(
                question.query_rewritten, method=self.config.retrieval_strategy
            )
            enrichment = None
            if self.enrichment:
                try:
                    enrichment = self.enrichment.search(
                        question.query_rewritten,
                        method=self.config.retrieval_strategy,
                        textual_index_id=retrieved.index_id,
                    )
                except Exception as exc:
                    if not self.config.enrichment_retrieval.fail_open:
                        raise
                    result["enrichment_error_type"] = type(exc).__name__
            if enrichment is not None:
                context, evidences, usage = build_context_with_enrichment(
                    retrieved.hits,
                    enrichment_hits=enrichment.hits,
                    max_chars=self.config.generation.max_context_chars,
                    enrichment_max_chars=self.config.enrichment_retrieval.max_context_chars,
                    enrichment_max_fraction=self.config.enrichment_retrieval.max_context_fraction,
                )
                result["enrichment_index_id"] = enrichment.index_id
            else:
                context, evidences = build_context(
                    retrieved.hits, max_chars=self.config.generation.max_context_chars
                )
                usage = {
                    "total_chars": len(context),
                    "enrichment_chars": 0,
                    "enrichment_hits": 0,
                }
            result.update(
                context=context,
                context_usage=usage,
                evidence_ids=[item["chunk_id"] for item in evidences],
                index_id=retrieved.index_id,
                silver_dataset_id=retrieved.silver_dataset_id,
            )
            generation_started = perf_counter()
            structured = self.provider.invoke_structured(
                [
                    ("system", MCQ_SYSTEM_PROMPT),
                    ("human", build_human_prompt(question, context)),
                ],
                MCQStructuredResponse,
                include_raw=True,
            )
            result["generation_ms"] = (perf_counter() - generation_started) * 1000
            raw = structured.get("raw")
            if structured.get("parsed") is None:
                raise structured.get("parsing_error") or ValueError(
                    "Resposta estruturada ausente."
                )
            parsed = MCQStructuredResponse.model_validate(structured["parsed"])
            if not parsed.resposta.strip():
                raise ValueError("Resposta estruturada vazia.")
            result.update(
                status="ok",
                selected_option=parsed.opcao_escolhida,
                selected_class=classify_selected_option(
                    question, parsed.opcao_escolhida
                ),
                answer=parsed.resposta,
            )
        except Exception as exc:
            diagnostics = getattr(exc, "diagnostics", {})
            result.update(
                error_type=type(exc).__name__,
                fatal=bool(getattr(exc, "batch_fatal", False)),
                diagnostics={
                    key: diagnostics[key]
                    for key in (
                        "reason",
                        "error_code",
                        "http_status",
                        "request_id",
                        "quota_scope",
                        "finish_reason",
                    )
                    if key in diagnostics
                },
            )
        elapsed = (perf_counter() - started) * 1000
        if raw is not None:
            result["raw_model_content"] = str(raw.content)
        if snapshot:
            after = snapshot()
            tokens = {
                key: after[key] - before[key]
                for key in ("input_tokens", "output_tokens", "calls")
            }
        else:
            values = extract_usage(raw, elapsed)
            tokens = {
                "input_tokens": int(values["input_tokens"] or 0),
                "output_tokens": int(values["output_tokens"] or 0),
                "calls": int(raw is not None),
            }
        prices = self.config.generation.model_prices.get(self.config.generation.model)
        if prices is None and (
            self.config.generation.input_usd_per_million
            or self.config.generation.output_usd_per_million
        ):
            prices = {
                "input": self.config.generation.input_usd_per_million,
                "output": self.config.generation.output_usd_per_million,
            }
        cost = (
            (
                tokens["input_tokens"] * prices["input"]
                + tokens["output_tokens"] * prices["output"]
            )
            / 1_000_000
            if prices
            else None
        )
        return (
            result
            | tokens
            | {"latency_ms": elapsed, "estimated_cost_usd": cost, "at": now()}
        )


def summarize(records, expected):
    rows = list(records.values())
    completed = len(rows)
    counts = {
        kind: sum(row["selected_class"] == kind for row in rows)
        for kind in ("correct", "ambiguous", "wrong", "invalid")
    }
    attempts = [attempt for row in rows for attempt in row["attempts"]]
    latencies = pd.Series([row["latency_ms"] for row in rows], dtype=float)
    return {
        "expected": expected,
        "attempted": completed,
        "pending": expected - completed,
        "errors": sum(row["status"] != "ok" for row in rows),
        **counts,
        "accuracy": counts["correct"] / completed if completed else None,
        "latency_mean_ms": float(latencies.mean()) if completed else None,
        "latency_p50_ms": float(latencies.quantile(0.5)) if completed else None,
        "latency_p95_ms": float(latencies.quantile(0.95)) if completed else None,
        "input_tokens": sum(row["input_tokens"] for row in attempts),
        "output_tokens": sum(row["output_tokens"] for row in attempts),
        "calls": sum(row["calls"] for row in attempts),
        "estimated_cost_usd": (
            sum(row["estimated_cost_usd"] for row in attempts)
            if attempts
            and all(row["estimated_cost_usd"] is not None for row in attempts)
            else None
        ),
    }


def publish_overview(s3, bucket, output, *, split, current_run_id):
    runs = []
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{PREFIX}/", Delimiter="/"
    )
    for page in pages:
        for item in page.get("CommonPrefixes", []):
            run_prefix = item["Prefix"].rstrip("/")
            run_id = run_prefix.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
                continue
            state = read_json(s3, bucket, f"{run_prefix}/manifest.json", optional=True)
            config = read_json(s3, bucket, f"{run_prefix}/config.json", optional=True)
            if not state or not config or config.get("phase") != "mcq_current":
                continue
            if state.get("run_id") != run_id or state.get("signature") != digest(
                config
            ):
                continue
            entry = run_metadata(config, state)
            report_key = f"{run_prefix}/report.html"
            try:
                content = s3.get_object(Bucket=bucket, Key=report_key)["Body"].read()
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
                    raise
                entry["has_report"] = False
            else:
                entry["has_report"] = True
                destination = output / "runs" / run_id / "report.html"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            runs.append(entry)
    runs.sort(key=lambda row: (row["updated_at"], row["run_id"]), reverse=True)
    html = dashboard.render_run_overview(
        runs, split=split, current_run_id=current_run_id
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "index.json").write_bytes(json_bytes(runs))
    s3.put_object(
        Bucket=bucket,
        Key="mcq/current/index.html",
        Body=html.encode(),
        ContentType="text/html; charset=utf-8",
    )
    write_json(s3, bucket, "mcq/current/index.json", runs)


def publish(
    s3,
    bucket,
    prefix,
    config,
    state,
    records,
    output,
    questions,
    *,
    write_manifest=True,
):
    output.mkdir(parents=True, exist_ok=True)
    state.update(run_metadata(config, state))
    frame = report_frame(config, state, records, questions)
    parquet = io.BytesIO()
    frame.to_parquet(parquet, index=False)
    files = {
        "summary.json": (json_bytes(state), "application/json"),
        "results.csv": (frame.to_csv(index=False).encode(), "text/csv; charset=utf-8"),
        "results.parquet": (parquet.getvalue(), "application/octet-stream"),
        "report.html": (
            render_report(config, state, records, questions).encode(),
            "text/html; charset=utf-8",
        ),
    }
    for name, (payload, content_type) in files.items():
        (output / name).write_bytes(payload)
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{name}",
            Body=payload,
            ContentType=content_type,
        )
    (output / "config.json").write_bytes(json_bytes(config))
    (output / "questions.csv").write_bytes(questions.to_csv(index=False).encode())
    if write_manifest:
        write_json(s3, bucket, f"{prefix}/manifest.json", state)
    publish_overview(
        s3, bucket, output, split=config["split"], current_run_id=state["run_id"]
    )
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as file:
            stats = state["summary"]
            file.write(f'### {state["run_label"]} — {state["status"]}\n\n')
            file.write(
                f'Questões: {stats["attempted"]}/{stats["expected"]}; acertos: {stats["correct"]}; erros: {stats["errors"]}; pendentes: {stats["pending"]}.\n\n'
            )
            file.write(f"Relatório privado: `s3://{bucket}/{prefix}/report.html`\n")
            file.write(f"\nVisão geral: `s3://{bucket}/mcq/current/index.html`\n\n")
            file.write(
                "Baixe o artifact desta execução e abra `index.html`; os relatórios individuais estão em `runs/`.\n"
            )


def report_saved_run(args, *, s3=None):
    """Regenera HTML e visão geral sem SSM, recuperação ou chamadas de inferência."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.run_id):
        raise ValueError("Informe o run_id salvo para regenerar o relatório.")
    out = {
        key: value["value"]
        for key, value in json.loads(args.deployment_file.read_text()).items()
    }
    s3 = s3 or boto3.client("s3", region_name=out["region"])
    require_operations_enabled(s3, out["ops_bucket"])
    bucket, prefix = out["buckets"]["BENCHMARKS_BUCKET"], f"{PREFIX}/{args.run_id}"
    config = read_json(s3, bucket, f"{prefix}/config.json")
    state = read_json(s3, bucket, f"{prefix}/manifest.json")
    if (
        config.get("phase") != "mcq_current"
        or state.get("run_id") != args.run_id
        or state.get("signature") != digest(config)
    ):
        raise ValueError(
            "Configuração ou manifest incompatíveis; relatório não substituído."
        )
    prepared = pd.read_csv(
        io.BytesIO(
            s3.get_object(Bucket=bucket, Key=f"{prefix}/split.csv")["Body"].read()
        )
    ).fillna("")
    questions = prepared.loc[prepared["split"].eq(config["split"])].copy()
    if config["limit_queries"]:
        questions = questions.head(config["limit_queries"])
    records = {}
    for question in questions.itertuples(index=False):
        item = read_json(
            s3, bucket, f"{prefix}/questions/{question.query_id}.json", optional=True
        )
        if item:
            if item.get("signature") != state["signature"] or str(
                item.get("query_id")
            ) != str(question.query_id):
                raise ValueError(
                    "Checkpoint de questão incompatível; relatório não substituído."
                )
            records[str(question.query_id)] = item
    # A operação não altera o checkpoint ou a identidade da execução original.
    publish(
        s3,
        bucket,
        prefix,
        config,
        state,
        records,
        args.output,
        questions,
        write_manifest=False,
    )
    print(f'Relatórios regenerados: {state["run_label"]}', flush=True)
    return 0


def run(args, *, s3=None, ssm=None, pipeline_factory=CurrentMCQPipeline):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.run_id) or not re.fullmatch(
        r"[0-9a-f]{40}", str(args.revision or "")
    ):
        raise ValueError("Use run_id válido e SHA completo.")
    out = {
        key: value["value"]
        for key, value in json.loads(args.deployment_file.read_text()).items()
    }
    s3 = s3 or boto3.client("s3", region_name=out["region"])
    ssm = ssm or boto3.client("ssm", region_name=out["region"])
    require_operations_enabled(s3, out["ops_bucket"])
    bucket, prefix = out["buckets"]["BENCHMARKS_BUCKET"], f"{PREFIX}/{args.run_id}"
    previous = read_json(s3, bucket, f"{prefix}/config.json", optional=True)
    config, release_key, release_hash = load_aws_config(out, s3, ssm, previous)
    config = select_model(config, getattr(args, "model", None))
    payload = (ROOT / "AED/benchmark_questions_final.csv").read_bytes()
    prepared, questions = load_questions(payload, args.split, args.limit_queries)
    identity = {
        "schema_version": 1,
        "phase": "mcq_current",
        "revision": args.revision,
        "split": args.split,
        "split_seed": SPLIT_SEED,
        "limit_queries": args.limit_queries,
        "benchmark_sha256": hashlib.sha256(payload).hexdigest(),
        "prompt_version": MCQ_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(MCQ_SYSTEM_PROMPT.encode()).hexdigest(),
        "data_release_key": release_key,
        "data_release_sha256": release_hash,
        "execution_parallelism": 1,
        "effective_config": public_config(config),
    }
    signature = digest(identity)
    if previous and digest(previous) != signature:
        raise ValueError(
            "SHA, configuração, dataset ou split divergem deste run_id. Use um novo ID."
        )
    write_json(s3, bucket, f"{prefix}/config.json", identity)
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/split.csv",
        Body=prepared.to_csv(index=False).encode(),
        ContentType="text/csv; charset=utf-8",
    )
    records = {}
    for question in questions.itertuples(index=False):
        key = f"{prefix}/questions/{question.query_id}.json"
        item = read_json(s3, bucket, key, optional=True)
        if item:
            if item.get("signature") != signature or item.get("query_id") != str(
                question.query_id
            ):
                raise ValueError(
                    "Checkpoint de questão incompatível; nenhuma resposta substituída."
                )
            records[str(question.query_id)] = item
    state = {
        "run_id": args.run_id,
        "signature": signature,
        "split": args.split,
        "status": "running",
        "updated_at": now(),
        "summary": summarize(records, len(questions)),
    }
    write_json(s3, bucket, f"{prefix}/manifest.json", state)
    error_type = None
    try:
        pending = [
            question
            for question in questions.itertuples(index=False)
            if records.get(str(question.query_id), {}).get("status") != "ok"
        ]
        if pending:
            pipeline = pipeline_factory(config)
            consecutive_errors = 0
            for question in pending:
                require_operations_enabled(s3, out["ops_bucket"])
                query_id = str(question.query_id)
                item = pipeline.evaluate(question)
                attempts = records.get(query_id, {}).get("attempts", [])
                item["attempts"] = attempts + [
                    {
                        key: item[key]
                        for key in (
                            "status",
                            "input_tokens",
                            "output_tokens",
                            "calls",
                            "estimated_cost_usd",
                            "latency_ms",
                            "at",
                        )
                    }
                ]
                item["attempt_count"] = len(item["attempts"])
                item["total_input_tokens"] = sum(
                    attempt["input_tokens"] for attempt in item["attempts"]
                )
                item["total_output_tokens"] = sum(
                    attempt["output_tokens"] for attempt in item["attempts"]
                )
                costs = [attempt["estimated_cost_usd"] for attempt in item["attempts"]]
                item["total_estimated_cost_usd"] = (
                    sum(costs) if all(cost is not None for cost in costs) else None
                )
                item["signature"] = signature
                write_json(s3, bucket, f"{prefix}/questions/{query_id}.json", item)
                records[query_id] = item
                state.update(
                    updated_at=now(), summary=summarize(records, len(questions))
                )
                write_json(s3, bucket, f"{prefix}/manifest.json", state)
                print(
                    f'[{len(records)}/{len(questions)}] {query_id}: {item["selected_class"]}',
                    flush=True,
                )
                consecutive_errors = (
                    consecutive_errors + 1 if item["status"] != "ok" else 0
                )
                if item["fatal"] or consecutive_errors >= 3:
                    state["stop_reason"] = (
                        "fatal_provider" if item["fatal"] else "consecutive_errors"
                    )
                    break
    except Exception as exc:
        error_type = type(exc).__name__
    stats = summarize(records, len(questions))
    status = (
        "complete"
        if not error_type and not stats["errors"] and not stats["pending"]
        else "incomplete"
    )
    state.update(status=status, updated_at=now(), summary=stats)
    if error_type:
        state["error_type"] = error_type
    publish(s3, bucket, prefix, identity, state, records, args.output, questions)
    print(
        f'[{status}] MCQ {args.split}: {stats["correct"]}/{stats["attempted"]} corretas; {stats["errors"]} erros; {stats["pending"]} pendentes.'
    )
    print(f"Relatório: s3://{bucket}/{prefix}/report.html")
    if status != "complete":
        print(
            "Retome o mesmo run_id e SHA para tentar somente erros e pendências. Consulte manifest.json e questions/*.json."
        )
    return 0 if status == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-file", required=True, type=Path)
    parser.add_argument("--revision", default=os.getenv("GITHUB_SHA"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--model", choices=list(AWS_MCQ_MODELS), default="qwen3-32b")
    parser.add_argument("--operation", choices=["run", "report"], default="run")
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mcq-current")
    args = parser.parse_args()
    try:
        code = report_saved_run(args) if args.operation == "report" else run(args)
    except Exception as exc:
        print(
            (
                f"Benchmark não iniciado: {exc}"
                if type(exc) is ValueError
                else f"Benchmark interrompido: {type(exc).__name__}; confira permissões, release e logs."
            ),
            flush=True,
        )
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
