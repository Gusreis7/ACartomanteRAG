from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
from langchain_core.messages import AIMessage
from moto import mock_aws
import pandas as pd
import pytest

from benchmark import run_current_mcq as mcq
from src.config.config_manager import ConfigManager
from src.generation.provider import GenerationResponseError
from src.retrieval.types import RetrievalHit


@pytest.fixture(autouse=True)
def environment():
    original = dict(os.environ)
    for key in tuple(os.environ):
        if key.startswith(
            (
                "QDRANT_",
                "RUSTFS_",
                "LLM_",
                "ENRICHMENT_",
                "ACARTOMANTE_",
                "DATA_RELEASE",
            )
        ) or key in {
            "CONFIG_FILE",
            "ENV_FILE",
            "GITHUB_STEP_SUMMARY",
            "COLIBRI_REVISION",
        }:
            os.environ.pop(key)
    os.environ.update(AWS_DEFAULT_REGION="us-east-1", AWS_EC2_METADATA_DISABLED="true")
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def aws(tmp_path):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        ssm = boto3.client("ssm", region_name="us-east-1")
        buckets = {
            name: "mcq-test-" + suffix
            for name, suffix in {
                "RAW_BUCKET": "raw",
                "BRONZE_BUCKET": "bronze",
                "SILVER_BUCKET": "silver",
                "EMBEDDING_INDEXES_BUCKET": "indexes",
                "ENRICHMENT_BUCKET": "enrichment",
                "BENCHMARKS_BUCKET": "benchmarks",
            }.items()
        }
        for name in [*buckets.values(), "mcq-test-ops"]:
            s3.create_bucket(Bucket=name)
        release = {
            "status": "success",
            "run_id": "corpus-v1",
            "index_id": "idx-colibri-v1",
            "enrichment_index_pointer": "production/enrichment-v1.json",
            "bronze_pointer": "production/bronze-v1.json",
            "silver_pointer": "production/silver-v1.json",
        }
        for key in ["production/current.json", "production/releases/corpus-v1.json"]:
            mcq.write_json(s3, buckets["EMBEDDING_INDEXES_BUCKET"], key, release)
        ssm.put_parameter(
            Name="/mcq-test/app",
            Type="SecureString",
            Value=json.dumps(
                {
                    "QDRANT_URL": "https://qdrant.example.org",
                    "QDRANT_API_KEY": "read-key",
                    "REDIS_PASSWORD": "private",
                    "REDIS_INSPECT_PASSWORD": "inspect",
                }
            ),
        )
        out = {
            "region": "us-east-1",
            "ops_bucket": "mcq-test-ops",
            "buckets": buckets,
            "app_settings_parameter": "/mcq-test/app",
        }
        deployment = tmp_path / "deployment.json"
        deployment.write_text(
            json.dumps({key: {"value": value} for key, value in out.items()})
        )
        args = argparse.Namespace(
            deployment_file=deployment,
            revision="a" * 40,
            run_id="mcq-dev-test",
            split="dev",
            limit_queries=2,
            output=tmp_path / "reports",
        )
        yield SimpleNamespace(
            s3=s3,
            ssm=ssm,
            out=out,
            args=args,
            release=release,
            bucket=buckets["BENCHMARKS_BUCKET"],
            prefix=f"{mcq.PREFIX}/{args.run_id}",
        )


def result(question, *, error=False, fatal=False, wrong=False):
    return {
        "query_id": str(question.query_id),
        "query": str(question.query_rewritten),
        "canonical_work": str(question.canonical_work),
        "difficulty": str(question.difficulty),
        "correct_option": str(question.correct_option),
        "status": "error" if error else "ok",
        "selected_class": "invalid" if error else "wrong" if wrong else "correct",
        "selected_option": None if error else "A",
        "answer": "<script>alert(1)</script>" if not error else "",
        "fatal": fatal,
        "latency_ms": 100.0,
        "input_tokens": 100,
        "output_tokens": 10,
        "calls": 1,
        "estimated_cost_usd": 0.000021,
        "at": mcq.now(),
    }


def run(aws, evaluate):
    pipeline = SimpleNamespace(evaluate=evaluate)
    return mcq.run(
        aws.args, s3=aws.s3, ssm=aws.ssm, pipeline_factory=lambda config: pipeline
    )


def state(aws):
    return mcq.read_json(aws.s3, aws.bucket, f"{aws.prefix}/manifest.json")


def test_real_dataset_split_is_disjoint_and_preserves_existing_seed():
    payload = (mcq.ROOT / "AED/benchmark_questions_final.csv").read_bytes()
    first, dev = mcq.load_questions(payload, "dev", 0)
    second, holdout = mcq.load_questions(payload, "holdout", 0)
    pd.testing.assert_frame_equal(first, second)
    assert not set(dev.query_id) & set(holdout.query_id)
    assert len(dev) + len(holdout) == len(first)
    _, sample = mcq.load_questions(payload, "dev", 2)
    assert list(sample.query_id) == list(dev.query_id.head(2))


@pytest.mark.parametrize("split,limit", [("all", 0), ("dev", -1), ("holdout", 100000)])
def test_invalid_selection_rejected(split, limit):
    with pytest.raises(ValueError):
        mcq.load_questions(b"", split, limit)


def test_full_success_exports_report_and_never_exports_credentials(aws):
    assert run(aws, result) == 0
    current = state(aws)
    assert current["status"] == "complete" and current["summary"]["correct"] == 2
    assert current["summary"]["input_tokens"] == 200
    assert current["summary"]["estimated_cost_usd"] == pytest.approx(0.000042)
    config = mcq.read_json(aws.s3, aws.bucket, f"{aws.prefix}/config.json")
    assert config["effective_config"]["generation"]["model"] == "qwen.qwen3-32b"
    assert config["effective_config"]["generation"]["max_tokens"] == 1024
    assert config["effective_config"]["generation"]["temperature"] == 0.1
    assert config["effective_config"]["retrieval_index"]["index_id"] == "idx-colibri-v1"
    assert config["data_release_key"] == "production/releases/corpus-v1.json"
    report = (aws.args.output / "report.html").read_text()
    assert "\\u003cscript\\u003ealert(1)" in report and "<script>alert" not in report
    for private in ["read-key", "qdrant.example.org"]:
        assert private not in report and private not in json.dumps(config)
    assert "REDIS_PASSWORD" not in report and "REDIS_INSPECT_PASSWORD" not in report
    exported = pd.read_parquet(
        io.BytesIO(
            aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/results.parquet")[
                "Body"
            ].read()
        )
    )
    assert len(exported) == 2 and set(exported.status) == {"ok"}
    assert exported.total_input_tokens.sum() == 200


def test_resume_only_errors_accumulates_cost_and_keeps_original_release(aws):
    calls = []

    def first(question):
        calls.append(str(question.query_id))
        return result(question, error=len(calls) == 2)

    assert run(aws, first) == 2
    failed_id = calls[1]
    assert state(aws)["summary"]["errors"] == 1
    newer = aws.release | {"run_id": "corpus-v2", "index_id": "idx-colibri-v2"}
    mcq.write_json(
        aws.s3,
        aws.out["buckets"]["EMBEDDING_INDEXES_BUCKET"],
        "production/current.json",
        newer,
    )
    calls.clear()

    def retry(question):
        calls.append(str(question.query_id))
        return result(question)

    assert run(aws, retry) == 0
    assert calls == [failed_id]
    assert state(aws)["summary"]["calls"] == 3
    assert state(aws)["summary"]["estimated_cost_usd"] == pytest.approx(0.000063)
    final = mcq.read_json(
        aws.s3, aws.bucket, f"{aws.prefix}/questions/{failed_id}.json"
    )
    assert final["attempt_count"] == 2 and final["total_input_tokens"] == 200
    assert (
        mcq.read_json(aws.s3, aws.bucket, f"{aws.prefix}/config.json")[
            "data_release_key"
        ]
        == "production/releases/corpus-v1.json"
    )


def test_valid_wrong_answers_are_not_retried(aws):
    assert run(aws, lambda question: result(question, wrong=True)) == 0
    assert state(aws)["summary"]["accuracy"] == 0
    never = Mock(side_effect=AssertionError("Não deve gerar novamente"))
    assert mcq.run(aws.args, s3=aws.s3, ssm=aws.ssm, pipeline_factory=never) == 0
    never.assert_not_called()


@pytest.mark.parametrize(
    "field,value", [("revision", "b" * 40), ("split", "holdout"), ("limit_queries", 1)]
)
def test_resume_mismatch_never_replaces_results(aws, field, value):
    assert run(aws, result) == 0
    original = aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/manifest.json")[
        "Body"
    ].read()
    setattr(aws.args, field, value)
    with pytest.raises(ValueError, match="divergem"):
        run(aws, result)
    assert (
        aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/manifest.json")[
            "Body"
        ].read()
        == original
    )


def test_fatal_quota_stops_and_preserves_remaining_questions(aws):
    calls = []

    def fail(question):
        calls.append(str(question.query_id))
        return result(question, error=True, fatal=True)

    assert run(aws, fail) == 2
    assert len(calls) == 1 and state(aws)["summary"]["pending"] == 1
    assert state(aws)["stop_reason"] == "fatal_provider"
    assert (aws.args.output / "report.html").exists()


def test_emergency_between_questions_stops_with_checkpoint(aws):
    def stop(question):
        mcq.write_json(
            aws.s3, aws.out["ops_bucket"], "config/emergency.json", {"active": True}
        )
        return result(question)

    assert run(aws, stop) == 2
    assert state(aws)["summary"]["correct"] == 1
    assert state(aws)["summary"]["pending"] == 1
    assert state(aws)["error_type"] == "RuntimeError"


def test_model_initialization_failure_still_exports_partial_report(aws):
    def fail(config):
        raise RuntimeError("init")

    assert mcq.run(aws.args, s3=aws.s3, ssm=aws.ssm, pipeline_factory=fail) == 2
    assert state(aws)["summary"]["pending"] == 2
    assert (aws.args.output / "report.html").exists()


def pipeline(aws, *, enrichment=None, provider=None):
    config, _, _ = mcq.load_aws_config(aws.out, aws.s3, aws.ssm)
    instance = mcq.CurrentMCQPipeline.__new__(mcq.CurrentMCQPipeline)
    instance.config, instance.enrichment, instance.enrichment_error = (
        config,
        enrichment,
        None,
    )
    instance.retrieval = SimpleNamespace(
        search=Mock(
            return_value=SimpleNamespace(
                hits=[
                    RetrievalHit(
                        1,
                        "chunk-1",
                        0.8,
                        "Capitu conversa com Bentinho.",
                        {"canonical_work": "Dom Casmurro"},
                    )
                ],
                index_id="idx-colibri-v1",
                silver_dataset_id="silver-v1",
            )
        )
    )
    instance.provider = provider
    payload = (mcq.ROOT / "AED/benchmark_questions_final.csv").read_bytes()
    _, questions = mcq.load_questions(payload, "dev", 1)
    return instance, next(questions.itertuples(index=False))


def test_current_pipeline_structured_output_and_retry_token_accounting(aws):
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def invoke(messages, schema, **kwargs):
        assert schema is mcq.MCQStructuredResponse and kwargs == {"include_raw": True}
        assert "Capitu conversa com Bentinho" in messages[1][1]
        assert "GABARITO_PRIVADO_TESTE" not in messages[1][1]
        usage.update(input_tokens=300, output_tokens=60, calls=3)
        return {
            "parsed": schema(
                resposta="As evidências sustentam a escolha.", opcao_escolhida="A"
            ),
            "raw": AIMessage(content='{"resposta":"ok","opcao_escolhida":"A"}'),
            "parsing_error": None,
        }

    provider = SimpleNamespace(
        usage_snapshot=lambda: dict(usage), invoke_structured=invoke
    )
    instance, question = pipeline(aws, provider=provider)
    question = SimpleNamespace(
        **question._asdict(), unused_gold="GABARITO_PRIVADO_TESTE"
    )
    item = instance.evaluate(question)
    assert item["status"] == "ok" and item["calls"] == 3
    assert item["input_tokens"] == 300 and item["output_tokens"] == 60
    assert item["estimated_cost_usd"] == pytest.approx(
        (300 * 0.15 + 60 * 0.6) / 1_000_000
    )
    instance.retrieval.search.assert_called_once_with(
        question.query_rewritten, method="dense_cross_encoder"
    )


def test_current_pipeline_keeps_invalid_response_and_diagnostics(aws):
    error = GenerationResponseError(
        "truncada", diagnostics={"reason": "truncated", "request_id": "request-1"}
    )
    provider = SimpleNamespace(
        invoke_structured=lambda *args, **kwargs: {
            "parsed": None,
            "raw": AIMessage(content='{"resposta":'),
            "parsing_error": error,
        }
    )
    instance, question = pipeline(aws, provider=provider)
    item = instance.evaluate(question)
    assert item["status"] == "error" and item["selected_class"] == "invalid"
    assert item["diagnostics"]["request_id"] == "request-1"
    assert item["raw_model_content"] == '{"resposta":'


def test_enrichment_fail_closed_prevents_generation(aws):
    provider = SimpleNamespace(invoke_structured=Mock())
    instance, question = pipeline(
        aws,
        enrichment=SimpleNamespace(
            search=Mock(side_effect=RuntimeError("unavailable"))
        ),
        provider=provider,
    )
    assert instance.config.enrichment_retrieval.fail_open is False
    assert instance.evaluate(question)["status"] == "error"
    provider.invoke_structured.assert_not_called()


@pytest.mark.parametrize(
    "choice,provider,model",
    [
        ("qwen3-32b", "bedrock-mantle", "qwen.qwen3-32b"),
        ("llama4-scout", "bedrock", "us.meta.llama4-scout-17b-instruct-v1:0"),
    ],
)
def test_selecting_model_preserves_all_other_runtime_settings(
    aws, choice, provider, model
):
    config, _, _ = mcq.load_aws_config(aws.out, aws.s3, aws.ssm)
    original = config.model_dump()
    selected = mcq.select_model(config, choice)
    expected = config.model_dump()
    expected["generation"].update(provider=provider, model=model)
    assert selected.model_dump() == expected
    assert config.model_dump() == original
    assert selected.generation.model_prices[model]["input"] > 0


def test_unknown_model_never_writes_results(aws):
    aws.args.model = "another-model"
    with pytest.raises(ValueError, match="Selecione"):
        run(aws, result)
    assert (
        mcq.read_json(aws.s3, aws.bucket, f"{aws.prefix}/config.json", optional=True)
        is None
    )


def test_model_change_requires_a_new_run(aws):
    assert run(aws, result) == 0
    before = aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/manifest.json")[
        "Body"
    ].read()
    aws.args.model = "llama4-scout"
    with pytest.raises(ValueError, match="divergem"):
        run(aws, result)
    assert (
        aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/manifest.json")[
            "Body"
        ].read()
        == before
    )


def test_overview_contains_models_commits_and_offline_reports(aws):
    assert run(aws, result) == 0
    first_id = aws.args.run_id
    aws.args.run_id = "scout-test"
    aws.args.model = "llama4-scout"
    aws.args.revision = "b" * 40
    assert run(aws, result) == 0
    catalog = mcq.read_json(aws.s3, aws.bucket, "mcq/current/index.json")
    assert {row["model_label"] for row in catalog} == {"Qwen3 32B", "Llama 4 Scout"}
    assert {row["revision"] for row in catalog} == {"a" * 40, "b" * 40}
    assert len({row["comparison_id"] for row in catalog}) == 1
    for row in catalog:
        assert (
            row["revision"][:8] in row["run_label"]
            and row["model_label"] in row["run_label"]
        )
        assert (aws.args.output / "runs" / row["run_id"] / "report.html").is_file()
    assert {row["run_id"] for row in catalog} == {first_id, "scout-test"}
    html = (aws.args.output / "index.html").read_text()
    for text in [
        "Acertos por run",
        "Latência p95 por run",
        "Tokens por run",
        "Custo estimado por run",
    ]:
        assert text in html
    assert '<select id="split">' in html and '<select id="group">' in html
    exported = pd.read_parquet(aws.args.output / "results.parquet")
    assert set(exported.model) == {"us.meta.llama4-scout-17b-instruct-v1:0"}
    assert set(exported.revision) == {"b" * 40}


def test_old_runs_without_new_metadata_are_listed(aws):
    assert run(aws, result) == 0
    manifest = state(aws)
    for key in [
        "run_label",
        "model",
        "model_label",
        "provider",
        "comparison_id",
        "comparison_label",
        "revision",
    ]:
        manifest.pop(key)
    mcq.write_json(aws.s3, aws.bucket, f"{aws.prefix}/manifest.json", manifest)
    mcq.publish_overview(
        aws.s3, aws.bucket, aws.args.output, split="dev", current_run_id=aws.args.run_id
    )
    [entry] = mcq.read_json(aws.s3, aws.bucket, "mcq/current/index.json")
    assert entry["model_label"] == "Qwen3 32B" and entry["revision"] == "a" * 40


@pytest.mark.parametrize(
    "field", ["split", "data_release_sha256", "benchmark_sha256", "prompt_sha256"]
)
def test_overview_keeps_different_protocols_separate(aws, field):
    import copy

    assert run(aws, result) == 0
    config = mcq.read_json(aws.s3, aws.bucket, f"{aws.prefix}/config.json")
    changed = copy.deepcopy(config)
    changed[field] = "holdout" if field == "split" else "different"
    assert (
        mcq.run_metadata(config, state(aws))["comparison_id"]
        != mcq.run_metadata(changed, state(aws))["comparison_id"]
    )


def test_generation_parameter_change_has_a_separate_comparison_group(aws):
    import copy

    assert run(aws, result) == 0
    config = mcq.read_json(aws.s3, aws.bucket, f"{aws.prefix}/config.json")
    changed = copy.deepcopy(config)
    changed["effective_config"]["generation"]["temperature"] += 0.1
    assert (
        mcq.run_metadata(config, state(aws))["comparison_id"]
        != mcq.run_metadata(changed, state(aws))["comparison_id"]
    )


def test_regenerate_old_report_does_not_run_inference_or_change_checkpoint(
    aws, monkeypatch
):
    assert run(aws, result) == 0
    keys = ["config.json", "manifest.json"]
    before = {
        key: aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/{key}")[
            "Body"
        ].read()
        for key in keys
    }
    monkeypatch.setattr(
        mcq, "load_aws_config", Mock(side_effect=AssertionError("Não carregar runtime"))
    )
    monkeypatch.setattr(
        mcq,
        "create_generation_provider",
        Mock(side_effect=AssertionError("Sem inferência")),
    )
    monkeypatch.setattr(
        mcq, "load_questions", Mock(side_effect=AssertionError("Use o split salvo"))
    )
    aws.args.revision = "c" * 40
    aws.args.model = "llama4-scout"
    assert mcq.report_saved_run(aws.args, s3=aws.s3) == 0
    for key in keys:
        assert (
            aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/{key}")[
                "Body"
            ].read()
            == before[key]
        )
    html = (aws.args.output / "report.html").read_text()
    assert "commit aaaaaaaa" in html and "Benchmark Qwen3 32B" in html
    mcq.create_generation_provider.assert_not_called()


def test_regenerate_rejects_incompatible_checkpoint(aws):
    assert run(aws, result) == 0
    manifest = state(aws) | {"signature": "incorrect"}
    mcq.write_json(aws.s3, aws.bucket, f"{aws.prefix}/manifest.json", manifest)
    with pytest.raises(ValueError, match="incompatíveis"):
        mcq.report_saved_run(aws.args, s3=aws.s3)


def test_partial_run_has_original_dashboard_and_explicit_status(aws):
    assert run(aws, lambda question: result(question, error=True, fatal=True)) == 2
    html = (aws.args.output / "report.html").read_text()
    assert "Resultado parcial" in html and '"status":"incomplete"' in html
    for text in [
        "function renderGeneral()",
        "function renderCompare()",
        "function renderModelTab(",
    ]:
        assert text in html


def test_workflow_uses_model_and_commit_and_keeps_cache_at_step_level():
    import yaml

    workflow = yaml.load(
        (mcq.ROOT / ".github/workflows/benchmark-mcq.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert set(
        workflow["on"]["workflow_dispatch"]["inputs"]["model"]["options"]
    ) == set(mcq.AWS_MCQ_MODELS)
    assert (
        "github.sha" in workflow["run-name"] and "inputs.model" in workflow["run-name"]
    )
    job = workflow["jobs"]["mcq"]
    assert "runner.temp" not in json.dumps(job["env"])
    assert any(
        step.get("uses", "").startswith("actions/upload-artifact@")
        and "always()" in step["if"]
        for step in job["steps"]
    )


def test_scout_uses_its_configured_token_prices(aws):
    provider = SimpleNamespace(
        invoke_structured=lambda *args, **kwargs: {
            "parsed": mcq.MCQStructuredResponse(
                resposta="Evidência", opcao_escolhida="A"
            ),
            "raw": AIMessage(
                content="Evidência",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                },
            ),
            "parsing_error": None,
        }
    )
    instance, question = pipeline(aws, provider=provider)
    instance.config = mcq.select_model(instance.config, "llama4-scout")
    assert instance.evaluate(question)["estimated_cost_usd"] == pytest.approx(
        (100 * 0.17 + 10 * 0.66) / 1_000_000
    )


def test_original_dashboard_and_dev_holdout_wrapper_remain_usable(aws):
    assert run(aws, result) == 0
    config = mcq.read_json(aws.s3, aws.bucket, f"{aws.prefix}/config.json")
    records = {}
    _, questions = mcq.load_questions(
        (mcq.ROOT / "AED/benchmark_questions_final.csv").read_bytes(), "dev", 2
    )
    for question in questions.itertuples(index=False):
        item = mcq.read_json(
            aws.s3, aws.bucket, f"{aws.prefix}/questions/{question.query_id}.json"
        )
        records[str(question.query_id)] = item
    frame = mcq.report_frame(config, state(aws), records, questions)
    legacy = mcq.dashboard.render_html(
        run_id="legacy",
        run_manifest=state(aws),
        run_config={"models": [], "split": "dev"},
        results=frame,
        questions=questions,
        configs=[{"config_id": "aws-current", "label": "Atual"}],
    )
    assert "__REPORT_STYLE__" not in legacy and '"provenance":{}' in legacy
    dev = {
        "run_id": "legacy",
        "split": "dev",
        "results": frame,
        "questions": questions,
        "configs": [{"config_id": "aws-current", "label": "Atual"}],
        "html": legacy,
    }
    combined = mcq.dashboard.render_dual_html(dev=dev, holdout=None)
    assert "__DEV_B64__" not in combined and "__COMPARISON__" not in combined


def test_regenerate_question_mismatch_preserves_existing_html(aws):
    assert run(aws, result) == 0
    before = aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/report.html")[
        "Body"
    ].read()
    _, questions = mcq.load_questions(
        (mcq.ROOT / "AED/benchmark_questions_final.csv").read_bytes(), "dev", 2
    )
    query_id = str(questions.iloc[0].query_id)
    key = f"{aws.prefix}/questions/{query_id}.json"
    item = mcq.read_json(aws.s3, aws.bucket, key) | {"signature": "incorrect"}
    mcq.write_json(aws.s3, aws.bucket, key, item)
    with pytest.raises(ValueError, match="Checkpoint de questão"):
        mcq.report_saved_run(aws.args, s3=aws.s3)
    assert (
        aws.s3.get_object(Bucket=aws.bucket, Key=f"{aws.prefix}/report.html")[
            "Body"
        ].read()
        == before
    )


def test_three_consecutive_failures_stop_without_processing_whole_split(aws):
    aws.args.limit_queries = 5
    calls = []

    def fail(question):
        calls.append(str(question.query_id))
        return result(question, error=True)

    assert run(aws, fail) == 2
    assert len(calls) == 3 and state(aws)["summary"]["pending"] == 2
    assert state(aws)["stop_reason"] == "consecutive_errors"


def test_enrichment_uses_published_text_index_and_same_context_builder(aws):
    provider = SimpleNamespace(
        invoke_structured=lambda *args, **kwargs: {
            "parsed": mcq.MCQStructuredResponse(
                resposta="Resumo e trecho sustentam a escolha.", opcao_escolhida="A"
            ),
            "raw": AIMessage(content="ok"),
            "parsing_error": None,
        }
    )
    enrichment = SimpleNamespace(
        search=Mock(
            return_value=SimpleNamespace(
                index_id="enrichment-colibri-v1",
                hits=[
                    {
                        "titulo": "Dom Casmurro",
                        "resumo": "Resumo auxiliar de teste.",
                        "text": "Resumo auxiliar de teste.",
                    }
                ],
            )
        )
    )
    instance, question = pipeline(aws, enrichment=enrichment, provider=provider)
    item = instance.evaluate(question)
    assert (
        item["status"] == "ok"
        and item["enrichment_index_id"] == "enrichment-colibri-v1"
    )
    assert item["context_usage"]["enrichment_chars"] > 0
    enrichment.search.assert_called_once_with(
        question.query_rewritten,
        method="dense_cross_encoder",
        textual_index_id="idx-colibri-v1",
    )


def test_active_emergency_prevents_any_benchmark_artifact(aws):
    mcq.write_json(
        aws.s3, aws.out["ops_bucket"], "config/emergency.json", {"active": True}
    )
    with pytest.raises(RuntimeError, match="emergência"):
        run(aws, result)
    assert aws.s3.list_objects_v2(Bucket=aws.bucket).get("Contents", []) == []
