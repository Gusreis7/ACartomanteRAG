#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from benchmark.matrix import BENCHMARK_PREFIX, OLLAMA_MODELS
from benchmark.prepare import make_qdrant_manager
from src.config.config_manager import ConfigManager
from src.generation.context_builder import build_context
from src.generation.provider import OllamaGenerationProvider
from src.storage.rustfs_manager import RustFSManager

SPLIT_KEY = f"{BENCHMARK_PREFIX}/prepared/splits/benchmark_with_split.csv"
MCQ_PROMPT_VERSION = "mcq_v1"
OPTION_KEYS = ("A", "B", "C", "D", "E")
VALID_OPTION_CLASSES = {"correct", "ambiguous", "wrong"}
DEFAULT_MAX_CONSECUTIVE_ERRORS = 5
ENRICHMENT_INDEX_CURRENT_KEY = "enrichment/current.json"

MCQ_SYSTEM_PROMPT = """
Você participa de um benchmark de compreensão da obra de Machado de Assis.

Use EXCLUSIVAMENTE as evidências fornecidas. Não use conhecimento externo para
preencher lacunas.

Escolha exatamente uma das alternativas A, B, C, D ou E.

A saída deve conter somente os campos estruturados solicitados:
- resposta: resposta final curta, em português, justificando a escolha com base
  nas evidências. Quando útil, cite [E1], [E2], etc.;
- opcao_escolhida: uma única letra entre A, B, C, D ou E.

Não produza cadeia de pensamento, notas de bastidor ou campos adicionais.
""".strip()


class MCQStructuredResponse(BaseModel):
    resposta: str = Field(
        min_length=1,
        description=(
            "Resposta final curta em português, baseada somente nas evidências."
        ),
    )
    opcao_escolhida: Literal["A", "B", "C", "D", "E"] = Field(
        description="Alternativa escolhida: A, B, C, D ou E."
    )


@dataclass(frozen=True)
class FrozenHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa a etapa MCQ sobre as 10 configurações de retrieval "
            "selecionadas no DEV, reutilizando os contextos congelados."
        )
    )
    parser.add_argument(
        "--retrieval-run-id",
        required=True,
        help="Run DEV de retrieval que contém final_selection_10.json.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="ID do run MCQ. Se omitido, cria mcq_dev_<timestamp>.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reusa shards model/config já concluídos no mesmo run-id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve shards já existentes.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Executa 2 questões, 1 config e 1 modelo.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Limita o número de query_id DEV.",
    )
    parser.add_argument(
        "--limit-configs",
        type=int,
        default=None,
        help="Limita o número de configs finais, na ordem do score balanceado.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(item["name"] for item in OLLAMA_MODELS),
        default=None,
        help="Subset opcional dos quatro modelos Ollama congelados.",
    )
    parser.add_argument(
        "--skip-model-id-check",
        action="store_true",
        help="Não valida o prefixo do digest local contra o ID congelado.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_ERRORS,
        help="Interrompe um modelo após N erros de chamada consecutivos.",
    )
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=2,
        help="Repete uma geração/parse com erro antes de contabilizá-la como falha.",
    )
    parser.add_argument(
        "--ollama-num-batch",
        type=int,
        default=None,
        help=(
            "Define options.num_batch do Ollama. Use 64/128 se houver pressão "
            "de VRAM. Não altera a quantidade de queries executadas em paralelo."
        ),
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help=(
            "Retoma automaticamente o run MCQ running/failed mais recente "
            "associado ao --retrieval-run-id."
        ),
    )
    parser.add_argument(
        "--enrichment",
        action="store_true",
        help=(
            "Adiciona ao contexto os resumos semanticamente recuperados do "
            "índice de enrichment. O retrieval textual congelado não é alterado."
        ),
    )
    parser.add_argument(
        "--enrichment-top-k",
        type=int,
        default=3,
        help="Quantidade de resumos recuperados por query/config. Padrão: 3.",
    )
    parser.add_argument(
        "--enrichment-max-chars",
        type=int,
        default=6000,
        help="Orçamento adicional para resumos recuperados. Padrão: 6000 chars.",
    )
    parser.add_argument(
        "--enrichment-default-embedder",
        default=None,
        help=(
            "Embedder do enrichment para configs sem embedder (ex.: BM25). "
            "Padrão: embedding.selected_model."
        ),
    )
    parser.add_argument(
        "--enrichment-debug-samples",
        type=int,
        default=3,
        help="Imprime os hits das primeiras N combinações query/config. Padrão: 3.",
    )
    return parser.parse_args()


def sanitize(value: str, *, max_len: int = 180) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip())
    if not value:
        raise ValueError("Valor vazio após sanitização.")
    return value[:max_len]


def default_run_id(*, enrichment: bool = False) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "mcq_enriched_dev" if enrichment else "mcq_dev"
    return f"{prefix}_{stamp}"


def read_json(storage: RustFSManager, bucket: str, key: str) -> dict[str, Any]:
    raw = storage.download_bytes(bucket, key)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON inválido: s3://{bucket}/{key}")
    return value


def find_latest_resumable_run(
    storage: RustFSManager,
    bucket: str,
    retrieval_run_id: str,
    *,
    enrichment_enabled: bool,
) -> str:
    run_prefix = "mcq_enriched_dev_" if enrichment_enabled else "mcq_dev_"
    prefix = f"{BENCHMARK_PREFIX}/runs/{run_prefix}"
    candidates: list[tuple[str, str]] = []
    for key in storage.list_objects(bucket, prefix=prefix):
        if not key.endswith("/manifest.json"):
            continue
        try:
            manifest = read_json(storage, bucket, key)
        except Exception:
            continue
        if str(manifest.get("source_retrieval_run_id") or "") != retrieval_run_id:
            continue
        if bool(manifest.get("enrichment_enabled", False)) != bool(enrichment_enabled):
            continue
        if str(manifest.get("status") or "") not in {"running", "failed"}:
            continue
        run_id = key.split("/runs/", 1)[1].split("/", 1)[0]
        stamp = str(manifest.get("created_at_utc") or run_id)
        candidates.append((stamp, run_id))

    if not candidates:
        raise RuntimeError(
            "Nenhum run MCQ running/failed encontrado para --resume-latest."
        )
    candidates.sort()
    return candidates[-1][1]


def upload_json(
    storage: RustFSManager,
    bucket: str,
    key: str,
    value: Any,
) -> None:
    storage.upload_bytes(
        bucket,
        key,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )


def dataframe_parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def upload_dataframe(
    storage: RustFSManager,
    bucket: str,
    key: str,
    frame: pd.DataFrame,
) -> None:
    storage.upload_bytes(
        bucket,
        key,
        dataframe_parquet_bytes(frame),
        content_type="application/vnd.apache.parquet",
    )


def read_parquet(storage: RustFSManager, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(storage.download_bytes(bucket, key)))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=json_default,
        ).encode("utf-8")
    ).hexdigest()


def ollama_request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    body = None
    method = "GET"
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Falha acessando Ollama em {url}: {exc}") from exc

    value = json.loads(raw.decode("utf-8")) if raw else {}
    return value if isinstance(value, dict) else {}


def resolve_ollama_models(
    *,
    base_url: str,
    requested_names: list[str],
    strict_ids: bool,
) -> list[dict[str, Any]]:
    tags = ollama_request(base_url, "/api/tags")
    installed = tags.get("models") or []
    by_name: dict[str, dict[str, Any]] = {}
    for item in installed:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            name = str(item.get(key) or "").strip()
            if name:
                by_name[name] = item

    expected = {item["name"]: item for item in OLLAMA_MODELS}
    resolved: list[dict[str, Any]] = []

    for name in requested_names:
        if name not in expected:
            raise RuntimeError(f"Modelo não congelado na matrix: {name}")
        local = by_name.get(name)
        if local is None:
            raise RuntimeError(
                f"Modelo Ollama não instalado: {name}. Rode `ollama list`."
            )

        digest = str(local.get("digest") or "")
        digest_clean = digest.removeprefix("sha256:")
        expected_id = str(expected[name]["ollama_id"])
        if strict_ids and not digest_clean.startswith(expected_id):
            raise RuntimeError(
                "ID/digest Ollama divergente. "
                f"model={name}; esperado_prefix={expected_id}; "
                f"obtido={digest or '<vazio>'}."
            )

        resolved.append(
            {
                "name": name,
                "expected_ollama_id": expected_id,
                "actual_digest": digest,
                "size": local.get("size"),
                "modified_at": local.get("modified_at"),
                "details": local.get("details"),
            }
        )

    return resolved


def unload_ollama_model(base_url: str, model: str) -> None:
    try:
        ollama_request(
            base_url,
            "/api/generate",
            payload={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=30.0,
        )
        print(f"[gpu] unload solicitado: {model}")
    except Exception as exc:
        print(f"[warn] não foi possível descarregar {model}: {exc}")


def validate_mcq_rows(frame: pd.DataFrame) -> None:
    required = {
        "query_id",
        "split",
        "query_rewritten",
        "canonical_work",
        "query_type",
        "difficulty",
        "correct_option",
    }
    for option in OPTION_KEYS:
        required.add(f"option_{option}")
        required.add(f"option_{option}_class")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Benchmark MCQ sem colunas: {missing}")

    if frame["query_id"].astype(str).duplicated().any():
        raise RuntimeError("query_id duplicado no split preparado.")

    problems: list[str] = []
    for row in frame.itertuples(index=False):
        query_id = str(row.query_id)
        correct_option = str(row.correct_option).strip().upper()
        if correct_option not in OPTION_KEYS:
            problems.append(f"{query_id}: correct_option inválida")
            continue

        classes = {
            option: str(getattr(row, f"option_{option}_class")).strip().lower()
            for option in OPTION_KEYS
        }
        if any(value not in VALID_OPTION_CLASSES for value in classes.values()):
            problems.append(f"{query_id}: classe de alternativa inválida")
        if sum(value == "correct" for value in classes.values()) != 1:
            problems.append(f"{query_id}: esperado exatamente 1 correct")
        if classes.get(correct_option) != "correct":
            problems.append(
                f"{query_id}: correct_option não aponta para classe correct"
            )
        if not str(row.query_rewritten).strip():
            problems.append(f"{query_id}: query_rewritten vazia")
        for option in OPTION_KEYS:
            if not str(getattr(row, f"option_{option}")).strip():
                problems.append(f"{query_id}: option_{option} vazia")

    if problems:
        preview = "\n".join(f"- {item}" for item in problems[:20])
        raise RuntimeError(f"Benchmark MCQ inválido:\n{preview}")


def selected_config_ids(
    selection: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = selection.get("selected") or []
    if len(rows) != 10:
        raise RuntimeError(
            f"final_selection_10.json deveria ter 10 configs; encontrou {len(rows)}."
        )

    ids: list[str] = []
    lookup: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(rows, start=1):
        config_id = str(row.get("config_id") or "").strip()
        if not config_id:
            raise RuntimeError("Config selecionada sem config_id.")
        if config_id in lookup:
            raise RuntimeError(f"config_id duplicado na seleção: {config_id}")
        item = dict(row)
        item["selection_rank"] = rank
        ids.append(config_id)
        lookup[config_id] = item
    return ids, lookup


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except Exception:
        return default


def load_prepared_query_vectors(
    *,
    storage: RustFSManager,
    bucket: str,
    alias: str,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    prefix = f"{BENCHMARK_PREFIX}/prepared/query_embeddings/{alias}"
    meta_key = f"{prefix}/queries.parquet"
    vectors_key = f"{prefix}/vectors.npz"
    manifest_key = f"{prefix}/manifest.json"
    for key in (meta_key, vectors_key, manifest_key):
        if not storage.object_exists(bucket, key):
            raise FileNotFoundError(f"Query embeddings ausentes: s3://{bucket}/{key}")

    manifest = read_json(storage, bucket, manifest_key)
    metadata = pd.read_parquet(io.BytesIO(storage.download_bytes(bucket, meta_key)))
    npz = np.load(io.BytesIO(storage.download_bytes(bucket, vectors_key)))
    vectors = np.asarray(npz["vectors"], dtype=np.float32)
    if len(metadata) != len(vectors):
        raise RuntimeError(
            f"Query embeddings {alias}: metadata={len(metadata)} vectors={len(vectors)}."
        )

    lookup: dict[tuple[str, str], np.ndarray] = {}
    for row_index, row in metadata.reset_index(drop=True).iterrows():
        key = (str(row["query_id"]), str(row["query_variant"]))
        if key in lookup:
            raise RuntimeError(f"Query embedding duplicado: {alias} {key}")
        lookup[key] = vectors[row_index]
    return lookup, manifest


def qdrant_enrichment_search(
    client: Any,
    *,
    collection: str,
    vector: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    query_vector = np.asarray(vector, dtype=np.float32).tolist()
    query_points = getattr(client, "query_points", None)
    if callable(query_points):
        response = query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
    else:
        search = getattr(client, "search", None)
        if not callable(search):
            raise RuntimeError("QdrantClient não possui query_points nem search.")
        points = search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    rows: list[dict[str, Any]] = []
    for point in points:
        payload = dict(getattr(point, "payload", None) or {})
        rows.append(
            {
                **payload,
                "score": safe_float(getattr(point, "score", 0.0)),
                "qdrant_point_id": str(getattr(point, "id", "")),
            }
        )
    return rows


def format_enrichment_context(
    hits: list[dict[str, Any]],
    *,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    used: list[dict[str, Any]] = []
    chars = 0

    for rank, hit in enumerate(hits, start=1):
        title = str(hit.get("titulo") or "não informada")
        document_type = str(hit.get("document_type") or "summary")
        section = str(hit.get("section_label") or "").strip()
        part_index = hit.get("part_index")
        part_count = hit.get("part_count")
        text = str(hit.get("text") or "").strip()
        events = [
            str(item).strip() for item in (hit.get("events") or []) if str(item).strip()
        ]

        header = [f"[R{rank}]", f"Obra: {title}"]
        header.append(
            "Tipo: resumo da obra"
            if document_type == "work_summary"
            else "Tipo: resumo de seção"
        )
        if section:
            header.append(f"Seção: {section}")
        if part_index is not None and part_count is not None and int(part_count) > 1:
            header.append(f"Parte: {part_index}/{part_count}")

        body = "\n".join(header) + "\nResumo: " + text
        if events:
            body += "\nEventos:\n" + "\n".join(f"- {event}" for event in events)
        body += "\n\n"

        remaining = max_chars - chars
        if remaining <= 0:
            break
        truncated = False
        if len(body) > remaining:
            body = body[:remaining].rstrip() + " …\n"
            truncated = True

        blocks.append(body)
        chars += len(body)
        used.append(
            {
                "rank": rank,
                "enrichment_document_id": hit.get("enrichment_document_id"),
                "qdrant_point_id": hit.get("qdrant_point_id"),
                "score": safe_float(hit.get("score")),
                "document_type": document_type,
                "titulo": title,
                "section_label": section or None,
                "section_index": hit.get("section_index"),
                "part_index": part_index,
                "part_count": part_count,
                "truncated": truncated,
            }
        )
        if truncated:
            break

    return "".join(blocks).strip(), used


def build_enrichment_context_lookup(
    *,
    config: Any,
    storage: RustFSManager,
    benchmark_bucket: str,
    questions_df: pd.DataFrame,
    config_ids: list[str],
    selection_lookup: dict[str, dict[str, Any]],
    top_k: int,
    max_chars: int,
    default_embedder: str,
    debug_samples: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    indexes_bucket = config.storage.embedding_indexes_bucket
    if not storage.object_exists(indexes_bucket, ENRICHMENT_INDEX_CURRENT_KEY):
        raise FileNotFoundError(
            "Índices de enrichment ainda não foram publicados: "
            f"s3://{indexes_bucket}/{ENRICHMENT_INDEX_CURRENT_KEY}"
        )
    index_set = read_json(storage, indexes_bucket, ENRICHMENT_INDEX_CURRENT_KEY)
    indexes = index_set.get("indexes") or {}

    alias_by_config: dict[str, str] = {}
    for config_id in config_ids:
        selected = selection_lookup[config_id]
        raw_alias = selected.get("embedder")
        alias = str(raw_alias).strip() if raw_alias is not None else ""
        if not alias or alias.casefold() in {"none", "nan", "null"}:
            alias = default_embedder
        if alias not in indexes:
            raise RuntimeError(
                f"Índice enrichment não publicado para alias={alias} config={config_id}."
            )
        alias_by_config[config_id] = alias

    vector_lookups: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    for alias in sorted(set(alias_by_config.values())):
        lookup, query_manifest = load_prepared_query_vectors(
            storage=storage,
            bucket=benchmark_bucket,
            alias=alias,
        )
        index_meta = indexes[alias]
        index_manifest = read_json(
            storage, indexes_bucket, str(index_meta["manifest_key"])
        )
        emb = index_manifest.get("embedding") or {}
        identity_pairs = {
            "model": (query_manifest.get("model"), emb.get("model")),
            "resolved_revision": (
                query_manifest.get("resolved_revision"),
                emb.get("resolved_revision"),
            ),
            "dimension": (query_manifest.get("dimension"), emb.get("dimension")),
            "query_prompt_name": (
                query_manifest.get("query_prompt_name"),
                emb.get("query_prompt_name"),
            ),
        }
        mismatches = {
            key: values
            for key, values in identity_pairs.items()
            if values[0] != values[1]
        }
        if mismatches:
            raise RuntimeError(
                f"Query vectors e enrichment index incompatíveis para {alias}: {mismatches}"
            )
        vector_lookups[alias] = lookup

    qdrant = make_qdrant_manager(config)
    client = getattr(qdrant, "client", qdrant)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    printed = 0

    for config_id in config_ids:
        alias = alias_by_config[config_id]
        index_meta = indexes[alias]
        collection = str(index_meta["collection"])
        vectors = vector_lookups[alias]
        for row in questions_df.itertuples(index=False):
            query_id = str(row.query_id)
            vector_key = (query_id, "rewritten")
            if vector_key not in vectors:
                raise RuntimeError(f"Query vector ausente: {alias} {vector_key}")
            hits = qdrant_enrichment_search(
                client,
                collection=collection,
                vector=vectors[vector_key],
                limit=top_k,
            )
            context, used = format_enrichment_context(hits, max_chars=max_chars)
            output[(config_id, query_id)] = {
                "context": context,
                "context_chars": len(context),
                "hits": used,
                "hit_count": len(used),
                "embedder": alias,
                "index_id": str(index_meta["index_id"]),
                "collection": collection,
            }

            if printed < debug_samples:
                printed += 1
                print(
                    f"[enrichment sample {printed}/{debug_samples}] "
                    f"config={config_id} query={query_id} alias={alias}"
                )
                for hit in used:
                    print(
                        "  "
                        f"#{hit['rank']} score={hit['score']:.4f} "
                        f"{hit['document_type']} | {hit['titulo']} | "
                        f"{hit.get('section_label') or '-'}"
                    )

    return output, {
        "index_set_key": ENRICHMENT_INDEX_CURRENT_KEY,
        "source_enrichment_dataset_id": index_set.get("source_enrichment_dataset_id"),
        "source_enrichment_manifest_sha256": index_set.get(
            "source_enrichment_manifest_sha256"
        ),
        "aliases": alias_by_config,
        "indexes": {
            alias: indexes[alias] for alias in sorted(set(alias_by_config.values()))
        },
    }


def build_frozen_contexts(
    *,
    results_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    config_ids: list[str],
    selection_lookup: dict[str, dict[str, Any]],
    max_chars: int,
    source_retrieval_run_id: str,
    enrichment_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    selected_query_ids = set(questions_df["query_id"].astype(str).tolist())
    frame = results_df[
        results_df["query_variant"].astype(str).eq("rewritten")
        & results_df["config_id"].astype(str).isin(config_ids)
        & results_df["query_id"].astype(str).isin(selected_query_ids)
    ].copy()

    expected_pairs = len(questions_df) * len(config_ids)
    groups = frame.groupby(["config_id", "query_id"], sort=False)
    if groups.ngroups != expected_pairs:
        raise RuntimeError(
            "Quantidade de contextos recuperados divergente. "
            f"esperado={expected_pairs}; encontrado={groups.ngroups}."
        )

    question_lookup = {
        str(row.query_id): row for row in questions_df.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []

    for (config_id_raw, query_id_raw), group in groups:
        config_id = str(config_id_raw)
        query_id = str(query_id_raw)
        if query_id not in question_lookup:
            raise RuntimeError(f"query_id do retrieval não está no DEV: {query_id}")

        group = group.sort_values("rank", kind="stable")
        hits: list[FrozenHit] = []
        retrieved_chunk_ids: list[str] = []
        for item in group.itertuples(index=False):
            chunk_id = str(getattr(item, "chunk_id"))
            retrieved_chunk_ids.append(chunk_id)
            rerank_score = getattr(item, "rerank_score", None)
            retrieval_score = getattr(item, "retrieval_score", 0.0)
            score = (
                safe_float(rerank_score)
                if rerank_score is not None and not pd.isna(rerank_score)
                else safe_float(retrieval_score)
            )
            metadata = {
                "titulo": getattr(item, "titulo", None),
                "autor": getattr(item, "autor", None),
                "categoria": getattr(item, "categoria", None),
                "section_title": getattr(item, "section_title", None),
                "section_header": getattr(item, "section_header", None),
            }
            hits.append(
                FrozenHit(
                    chunk_id=chunk_id,
                    text=str(getattr(item, "text", "") or ""),
                    score=score,
                    metadata=metadata,
                )
            )

        baseline_context, evidences = build_context(hits, max_chars=max_chars)
        used_chunk_ids = [str(item["chunk_id"]) for item in evidences]

        enrichment = (enrichment_lookup or {}).get((config_id, query_id))
        enrichment_context = str((enrichment or {}).get("context") or "").strip()
        if enrichment_context:
            context = (
                "EVIDÊNCIAS TEXTUAIS ORIGINAIS:\n"
                + baseline_context
                + "\n\nCONTEXTO ENRIQUECIDO RECUPERADO:\n"
                + enrichment_context
            )
        else:
            context = baseline_context

        enrichment_hits = list((enrichment or {}).get("hits") or [])
        context_id = stable_hash(
            {
                "source_retrieval_run_id": source_retrieval_run_id,
                "config_id": config_id,
                "query_id": query_id,
                "query_variant": "rewritten",
                "max_chars": max_chars,
                "used_chunk_ids": used_chunk_ids,
                "baseline_context": baseline_context,
                "enrichment_index_id": (enrichment or {}).get("index_id"),
                "enrichment_hit_ids": [
                    item.get("enrichment_document_id") for item in enrichment_hits
                ],
                "enrichment_context": enrichment_context,
            }
        )[:24]

        selection = selection_lookup[config_id]
        rows.append(
            {
                "context_id": context_id,
                "source_retrieval_run_id": source_retrieval_run_id,
                "config_id": config_id,
                "selection_rank": int(selection["selection_rank"]),
                "consensus_score": safe_float(selection.get("consensus_score")),
                "query_id": query_id,
                "query_variant": "rewritten",
                "context": context,
                "context_chars": len(context),
                "baseline_context_chars": len(baseline_context),
                "enrichment_enabled": bool(enrichment_context),
                "enrichment_context_chars": len(enrichment_context),
                "enrichment_hit_count": len(enrichment_hits),
                "enrichment_embedder": (enrichment or {}).get("embedder"),
                "enrichment_index_id": (enrichment or {}).get("index_id"),
                "enrichment_collection": (enrichment or {}).get("collection"),
                "enrichment_hits_json": json.dumps(
                    enrichment_hits, ensure_ascii=False, default=json_default
                ),
                "retrieved_chunk_count": len(retrieved_chunk_ids),
                "used_evidence_count": len(evidences),
                "retrieved_chunk_ids_json": json.dumps(
                    retrieved_chunk_ids, ensure_ascii=False
                ),
                "used_chunk_ids_json": json.dumps(used_chunk_ids, ensure_ascii=False),
                "evidences_json": json.dumps(
                    evidences, ensure_ascii=False, default=json_default
                ),
            }
        )

    contexts = (
        pd.DataFrame(rows)
        .sort_values(["selection_rank", "query_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(contexts) != expected_pairs:
        raise RuntimeError(
            f"Contextos congelados: esperado={expected_pairs}; obtido={len(contexts)}."
        )
    return contexts


def build_human_prompt(row: Any, context: str) -> str:
    alternatives = "\n".join(
        f"{option}) {str(getattr(row, f'option_{option}')).strip()}"
        for option in OPTION_KEYS
    )
    return (
        "EVIDÊNCIAS RECUPERADAS:\n"
        f"{context}\n\n"
        "PERGUNTA:\n"
        f"{str(row.query_rewritten).strip()}\n\n"
        "ALTERNATIVAS:\n"
        f"{alternatives}\n\n"
        "Escolha uma única alternativa. Preencha `resposta` com uma resposta "
        "final curta baseada nas evidências e `opcao_escolhida` com A, B, C, D ou E."
    )


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_usage(raw_message: Any, latency_ms: float) -> dict[str, Any]:
    usage = getattr(raw_message, "usage_metadata", None) or {}
    response_meta = getattr(raw_message, "response_metadata", None) or {}

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if input_tokens is None:
        input_tokens = response_meta.get("prompt_eval_count")
    if output_tokens is None:
        output_tokens = response_meta.get("eval_count")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    eval_duration_ns = response_meta.get("eval_duration")
    tokens_per_second = None
    try:
        if output_tokens is not None and eval_duration_ns:
            tokens_per_second = float(output_tokens) / (
                float(eval_duration_ns) / 1_000_000_000.0
            )
        elif output_tokens is not None and latency_ms > 0:
            tokens_per_second = float(output_tokens) / (latency_ms / 1000.0)
    except Exception:
        tokens_per_second = None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
        "ollama_total_duration_ns": response_meta.get("total_duration"),
        "ollama_load_duration_ns": response_meta.get("load_duration"),
        "ollama_prompt_eval_duration_ns": response_meta.get("prompt_eval_duration"),
        "ollama_eval_duration_ns": eval_duration_ns,
    }


def classify_selected_option(row: Any, selected_option: str | None) -> str:
    if selected_option not in OPTION_KEYS:
        return "invalid"
    value = str(getattr(row, f"option_{selected_option}_class")).strip().lower()
    return value if value in VALID_OPTION_CLASSES else "invalid"


def execute_one(
    *,
    provider: OllamaGenerationProvider,
    benchmark_row: Any,
    context_row: Any,
    model_meta: dict[str, Any],
    config_id: str,
    source_retrieval_run_id: str,
    generation_retries: int,
) -> dict[str, Any]:
    human_prompt = build_human_prompt(benchmark_row, str(context_row.context))
    started = time.perf_counter()

    parsed: MCQStructuredResponse | None = None
    raw_message: Any = None
    raw_content = ""
    parsing_error: Any = None
    error_type: str | None = None
    error_message: str | None = None
    attempts = 0

    for attempt in range(generation_retries + 1):
        attempts = attempt + 1
        parsed = None
        raw_message = None
        raw_content = ""
        parsing_error = None
        error_type = None
        error_message = None

        try:
            structured = provider.invoke_structured(
                [
                    ("system", MCQ_SYSTEM_PROMPT),
                    ("human", human_prompt),
                ],
                MCQStructuredResponse,
                include_raw=True,
            )
            if isinstance(structured, dict):
                raw_message = structured.get("raw")
                parsing_error = structured.get("parsing_error")
                parsed_value = structured.get("parsed")
            else:
                parsed_value = structured

            if isinstance(parsed_value, MCQStructuredResponse):
                parsed = parsed_value
            elif isinstance(parsed_value, dict):
                parsed = MCQStructuredResponse.model_validate(parsed_value)
            elif parsed_value is not None:
                parsed = MCQStructuredResponse.model_validate(parsed_value)

            raw_content = content_to_text(getattr(raw_message, "content", None))
            if parsed is None:
                if parsing_error is not None:
                    error_type = type(parsing_error).__name__
                    error_message = str(parsing_error)
                else:
                    error_type = "StructuredOutputError"
                    error_message = "Saída estruturada sem objeto parsed."
        except Exception as exc:
            error_type = type(exc).__name__
            cause = exc.__cause__
            if cause is not None:
                error_message = f"{exc} | cause={type(cause).__name__}: {cause}"
            else:
                error_message = str(exc)

        if parsed is not None:
            break
        if attempt < generation_retries:
            time.sleep(0.25)

    latency_ms = (time.perf_counter() - started) * 1000.0
    selected_option = parsed.opcao_escolhida if parsed is not None else None
    answer_text = parsed.resposta if parsed is not None else ""
    selected_class = classify_selected_option(benchmark_row, selected_option)
    usage = extract_usage(raw_message, latency_ms)

    return {
        "mcq_run_id": None,
        "source_retrieval_run_id": source_retrieval_run_id,
        "config_id": config_id,
        "selection_rank": int(context_row.selection_rank),
        "consensus_score": safe_float(context_row.consensus_score),
        "context_id": str(context_row.context_id),
        "query_id": str(benchmark_row.query_id),
        "split": str(benchmark_row.split),
        "canonical_work": str(benchmark_row.canonical_work),
        "difficulty": str(benchmark_row.difficulty),
        "query_type": str(benchmark_row.query_type),
        "query_variant": "rewritten",
        "query": str(benchmark_row.query_rewritten),
        "model": str(model_meta["name"]),
        "expected_ollama_id": str(model_meta["expected_ollama_id"]),
        "actual_digest": str(model_meta.get("actual_digest") or ""),
        "prompt_version": MCQ_PROMPT_VERSION,
        "generation_attempts": attempts,
        "answer_text": answer_text,
        "selected_option": selected_option,
        "selected_class": selected_class,
        "correct_option": str(benchmark_row.correct_option).strip().upper(),
        "is_correct": selected_class == "correct",
        "is_ambiguous": selected_class == "ambiguous",
        "is_wrong": selected_class == "wrong",
        "is_invalid": selected_class == "invalid",
        "raw_model_content": raw_content,
        "structured_parsing_error": (
            str(parsing_error) if parsing_error is not None else None
        ),
        "generation_error_type": error_type,
        "generation_error_message": error_message,
        "generation_latency_ms": latency_ms,
        "context_chars": int(context_row.context_chars),
        "baseline_context_chars": int(
            getattr(context_row, "baseline_context_chars", context_row.context_chars)
        ),
        "used_evidence_count": int(context_row.used_evidence_count),
        "enrichment_enabled": bool(getattr(context_row, "enrichment_enabled", False)),
        "enrichment_context_chars": int(
            getattr(context_row, "enrichment_context_chars", 0)
        ),
        "enrichment_hit_count": int(getattr(context_row, "enrichment_hit_count", 0)),
        "enrichment_embedder": getattr(context_row, "enrichment_embedder", None),
        "enrichment_index_id": getattr(context_row, "enrichment_index_id", None),
        "enrichment_collection": getattr(context_row, "enrichment_collection", None),
        **usage,
        "created_at_utc": utc_now(),
    }


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else 0.0


def summarize_group(group: pd.DataFrame, scope: str, **keys: Any) -> dict[str, Any]:
    total = len(group)
    valid = (~group["is_invalid"].astype(bool)).sum()
    return {
        "scope": scope,
        **keys,
        "n": int(total),
        "valid_n": int(valid),
        "accuracy": float(group["is_correct"].astype(float).mean()),
        "ambiguous_rate": float(group["is_ambiguous"].astype(float).mean()),
        "wrong_rate": float(group["is_wrong"].astype(float).mean()),
        "invalid_rate": float(group["is_invalid"].astype(float).mean()),
        "latency_mean_ms": float(
            pd.to_numeric(group["generation_latency_ms"], errors="coerce").mean()
        ),
        "latency_p50_ms": percentile(group["generation_latency_ms"], 0.50),
        "latency_p95_ms": percentile(group["generation_latency_ms"], 0.95),
        "input_tokens_mean": float(
            pd.to_numeric(group["input_tokens"], errors="coerce").dropna().mean()
            if pd.to_numeric(group["input_tokens"], errors="coerce").notna().any()
            else 0.0
        ),
        "output_tokens_mean": float(
            pd.to_numeric(group["output_tokens"], errors="coerce").dropna().mean()
            if pd.to_numeric(group["output_tokens"], errors="coerce").notna().any()
            else 0.0
        ),
        "tokens_per_second_mean": float(
            pd.to_numeric(group["tokens_per_second"], errors="coerce").dropna().mean()
            if pd.to_numeric(group["tokens_per_second"], errors="coerce").notna().any()
            else 0.0
        ),
    }


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (model, config_id), group in results.groupby(
        ["model", "config_id"], sort=False
    ):
        rows.append(
            summarize_group(
                group,
                "model_config",
                model=str(model),
                config_id=str(config_id),
            )
        )

    for model, group in results.groupby("model", sort=False):
        rows.append(
            summarize_group(
                group,
                "model",
                model=str(model),
                config_id=None,
            )
        )

    for config_id, group in results.groupby("config_id", sort=False):
        rows.append(
            summarize_group(
                group,
                "config",
                model=None,
                config_id=str(config_id),
            )
        )

    rows.append(
        summarize_group(
            results,
            "overall",
            model=None,
            config_id=None,
        )
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.limit_queries is not None and args.limit_queries <= 0:
        raise SystemExit("--limit-queries deve ser > 0.")
    if args.limit_configs is not None and args.limit_configs <= 0:
        raise SystemExit("--limit-configs deve ser > 0.")
    if args.max_consecutive_errors <= 0:
        raise SystemExit("--max-consecutive-errors deve ser > 0.")
    if args.generation_retries < 0:
        raise SystemExit("--generation-retries deve ser >= 0.")
    if args.ollama_num_batch is not None and args.ollama_num_batch <= 0:
        raise SystemExit("--ollama-num-batch deve ser > 0.")
    if args.enrichment_top_k <= 0:
        raise SystemExit("--enrichment-top-k deve ser > 0.")
    if args.enrichment_max_chars <= 0:
        raise SystemExit("--enrichment-max-chars deve ser > 0.")
    if args.enrichment_debug_samples < 0:
        raise SystemExit("--enrichment-debug-samples deve ser >= 0.")
    if args.resume_latest and args.run_id:
        raise SystemExit("Use --resume-latest ou --run-id, não ambos.")

    config = ConfigManager.load()
    storage = RustFSManager.from_config(config.environment.rustfs)
    bucket = config.storage.benchmarks_bucket

    retrieval_run_id = sanitize(args.retrieval_run_id)
    retrieval_prefix = f"{BENCHMARK_PREFIX}/runs/{retrieval_run_id}"
    retrieval_manifest_key = f"{retrieval_prefix}/manifest.json"
    retrieval_config_key = f"{retrieval_prefix}/config.json"
    selection_key = f"{retrieval_prefix}/retrieval/final_selection_10.json"
    results_key = f"{retrieval_prefix}/retrieval/results.parquet"

    for key in (
        retrieval_manifest_key,
        retrieval_config_key,
        selection_key,
        results_key,
        SPLIT_KEY,
    ):
        if not storage.object_exists(bucket, key):
            raise FileNotFoundError(f"Artefato ausente: s3://{bucket}/{key}")

    retrieval_manifest = read_json(storage, bucket, retrieval_manifest_key)
    retrieval_config = read_json(storage, bucket, retrieval_config_key)
    selection = read_json(storage, bucket, selection_key)

    if retrieval_manifest.get("status") != "success":
        raise RuntimeError("Run de retrieval não está com status=success.")
    if str(retrieval_config.get("split")) != "dev":
        raise RuntimeError("MCQ de seleção deve usar retrieval split=dev.")
    if str(selection.get("split")) != "dev":
        raise RuntimeError("final_selection_10.json não pertence ao DEV.")

    benchmark_sha = str(retrieval_config.get("benchmark_sha256") or "")
    if str(selection.get("benchmark_sha256") or "") != benchmark_sha:
        raise RuntimeError("Hash do benchmark diverge entre retrieval e seleção.")

    config_ids, selection_lookup = selected_config_ids(selection)
    if args.limit_configs is not None:
        config_ids = config_ids[: args.limit_configs]
    if args.smoke:
        config_ids = config_ids[:1]

    questions = pd.read_csv(io.BytesIO(storage.download_bytes(bucket, SPLIT_KEY)))
    questions = questions[questions["split"].astype(str).eq("dev")].copy()
    validate_mcq_rows(questions)
    questions = questions.sort_values("query_id", kind="stable").reset_index(drop=True)
    if args.limit_queries is not None:
        questions = questions.head(args.limit_queries).copy()
    if args.smoke:
        questions = questions.head(2).copy()
    if questions.empty:
        raise RuntimeError("Nenhuma questão DEV selecionada.")

    retrieval_results = read_parquet(storage, bucket, results_key)

    enrichment_lookup = None
    enrichment_metadata: dict[str, Any] | None = None
    if args.enrichment:
        default_enrichment_embedder = (
            args.enrichment_default_embedder or config.embedding.selected_model
        )
        enrichment_lookup, enrichment_metadata = build_enrichment_context_lookup(
            config=config,
            storage=storage,
            benchmark_bucket=bucket,
            questions_df=questions,
            config_ids=config_ids,
            selection_lookup=selection_lookup,
            top_k=args.enrichment_top_k,
            max_chars=args.enrichment_max_chars,
            default_embedder=default_enrichment_embedder,
            debug_samples=args.enrichment_debug_samples,
        )

    contexts = build_frozen_contexts(
        results_df=retrieval_results,
        questions_df=questions,
        config_ids=config_ids,
        selection_lookup=selection_lookup,
        max_chars=config.generation.max_context_chars,
        source_retrieval_run_id=retrieval_run_id,
        enrichment_lookup=enrichment_lookup,
    )

    model_names = args.models or [item["name"] for item in OLLAMA_MODELS]
    if args.smoke:
        model_names = model_names[:1]
    model_metadata = resolve_ollama_models(
        base_url=config.generation.base_url,
        requested_names=model_names,
        strict_ids=not args.skip_model_id_check,
    )
    model_lookup = {item["name"]: item for item in model_metadata}

    if args.resume_latest:
        run_id = find_latest_resumable_run(
            storage,
            bucket,
            retrieval_run_id,
            enrichment_enabled=args.enrichment,
        )
        args.resume = True
        print(f"[resume-latest] run_id={run_id}")
    else:
        run_id = sanitize(args.run_id or default_run_id(enrichment=args.enrichment))

    print(f"[run] run_id={run_id}")
    run_prefix = f"{BENCHMARK_PREFIX}/runs/{run_id}"
    mcq_prefix = f"{run_prefix}/mcq"
    manifest_key = f"{run_prefix}/manifest.json"

    if storage.object_exists(bucket, manifest_key) and not (args.resume or args.force):
        raise RuntimeError(
            f"run_id MCQ já existe: {run_id}. Use --resume ou outro --run-id."
        )

    generation_config = config.generation.model_copy(
        update={
            "temperature": 0.0,
            "max_tokens": min(int(config.generation.max_tokens), 256),
            "max_parallel": 1,
        },
        deep=True,
    )

    context_key = f"{mcq_prefix}/contexts.parquet"
    if storage.object_exists(bucket, context_key) and args.resume and not args.force:
        existing_contexts = read_parquet(storage, bucket, context_key)
        expected_pairs = len(questions) * len(config_ids)
        if len(existing_contexts) != expected_pairs:
            raise RuntimeError(
                "contexts.parquet existente não corresponde ao escopo atual."
            )
        contexts = existing_contexts
        print(f"[resume] contexts={len(contexts)}")
    else:
        upload_dataframe(storage, bucket, context_key, contexts)
        print(f"[freeze] contexts={len(contexts)}")

    schema_json = MCQStructuredResponse.model_json_schema()
    run_config = {
        "run_id": run_id,
        "phase": "mcq",
        "created_at_utc": utc_now(),
        "source_retrieval_run_id": retrieval_run_id,
        "source_retrieval_results": results_key,
        "source_final_selection": selection_key,
        "benchmark_sha256": benchmark_sha,
        "split": "dev",
        "query_variant": "rewritten",
        "base_queries": len(questions),
        "retrieval_configs": len(config_ids),
        "models": model_metadata,
        "expected_generations": len(questions) * len(config_ids) * len(model_names),
        "generation_overrides": {
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_tokens,
            "num_ctx": generation_config.num_ctx,
            "max_context_chars": generation_config.max_context_chars,
            "max_parallel": generation_config.max_parallel,
            "reasoning": False,
            "ollama_num_batch": args.ollama_num_batch,
            "generation_retries": args.generation_retries,
        },
        "prompt": {
            "version": MCQ_PROMPT_VERSION,
            "system": MCQ_SYSTEM_PROMPT,
            "structured_schema": schema_json,
            "fields": ["resposta", "opcao_escolhida"],
        },
        "selected_config_ids": config_ids,
        "enrichment": {
            "enabled": bool(args.enrichment),
            "top_k": args.enrichment_top_k if args.enrichment else 0,
            "max_chars": args.enrichment_max_chars if args.enrichment else 0,
            "default_embedder": (
                (args.enrichment_default_embedder or config.embedding.selected_model)
                if args.enrichment
                else None
            ),
            "index_set": enrichment_metadata,
            "retrieval_uses_canonical_work_filter": False,
            "baseline_text_context_evicted": False,
        },
        "selection_method": selection.get("method"),
        "runtime": {
            "python": sys.version,
        },
    }
    upload_json(storage, bucket, f"{run_prefix}/config.json", run_config)

    completed_shards: list[str] = []
    run_manifest = {
        "run_id": run_id,
        "phase": "mcq",
        "status": "running",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "source_retrieval_run_id": retrieval_run_id,
        "benchmark_sha256": benchmark_sha,
        "split": "dev",
        "enrichment_enabled": bool(args.enrichment),
        "completed_shards": completed_shards,
        "expected_shards": len(model_names) * len(config_ids),
    }
    upload_json(storage, bucket, manifest_key, run_manifest)

    question_lookup = {
        str(row.query_id): row for row in questions.itertuples(index=False)
    }
    context_lookup = {
        (str(row.config_id), str(row.query_id)): row
        for row in contexts.itertuples(index=False)
    }

    all_frames: list[pd.DataFrame] = []

    try:
        for model_number, model_name in enumerate(model_names, start=1):
            meta = model_lookup[model_name]
            model_slug = sanitize(model_name)
            model_cfg = generation_config.model_copy(
                update={"model": model_name},
                deep=True,
            )

            print(f"\n=== MODEL {model_number}/{len(model_names)}: {model_name} ===")
            print(
                f"[model-id] esperado={meta['expected_ollama_id']} "
                f"digest={str(meta.get('actual_digest') or '')[:24]}..."
            )

            provider: OllamaGenerationProvider | None = None
            try:
                provider = OllamaGenerationProvider(
                    model_cfg,
                    reasoning=False,
                    num_batch=args.ollama_num_batch,
                )
                provider.warmup()
                print("[warmup] ok")

                consecutive_errors = 0
                for config_number, config_id in enumerate(config_ids, start=1):
                    shard_id = f"{model_name}|{config_id}"
                    shard_key = f"{mcq_prefix}/shards/{model_slug}/{config_id}.parquet"
                    exists = storage.object_exists(bucket, shard_key)
                    if exists and args.resume and not args.force:
                        print(
                            f"[{config_number}/{len(config_ids)}] [resume] {config_id}"
                        )
                        shard = read_parquet(storage, bucket, shard_key)
                        all_frames.append(shard)
                        completed_shards.append(shard_id)
                        continue
                    if exists and not args.force:
                        raise RuntimeError(
                            f"Shard já existe: {shard_key}. Use --resume ou --force."
                        )

                    print(
                        f"[{config_number}/{len(config_ids)}] {config_id} "
                        f"| questions={len(questions)}"
                    )
                    rows: list[dict[str, Any]] = []

                    for question_number, benchmark_row in enumerate(
                        questions.itertuples(index=False), start=1
                    ):
                        query_id = str(benchmark_row.query_id)
                        context_row = context_lookup[(config_id, query_id)]
                        result = execute_one(
                            provider=provider,
                            benchmark_row=benchmark_row,
                            context_row=context_row,
                            model_meta=meta,
                            config_id=config_id,
                            source_retrieval_run_id=retrieval_run_id,
                            generation_retries=args.generation_retries,
                        )
                        result["mcq_run_id"] = run_id
                        rows.append(result)

                        if result["generation_error_type"]:
                            message = str(result.get("generation_error_message") or "")
                            if len(message) > 500:
                                message = message[:500] + "..."
                            print(
                                f"  [error] query={query_id} "
                                f"attempts={result.get('generation_attempts')} "
                                f"type={result['generation_error_type']} "
                                f"message={message}"
                            )

                        if result["generation_error_type"]:
                            consecutive_errors += 1
                        else:
                            consecutive_errors = 0

                        if consecutive_errors >= args.max_consecutive_errors:
                            raise RuntimeError(
                                f"{model_name}: {consecutive_errors} erros de "
                                "geração consecutivos. Interrompendo para evitar "
                                "repetir falha sistêmica."
                            )

                        if question_number % 10 == 0 or question_number == len(
                            questions
                        ):
                            correct = sum(bool(item["is_correct"]) for item in rows)
                            invalid = sum(bool(item["is_invalid"]) for item in rows)
                            print(
                                f"  progress {question_number}/{len(questions)} "
                                f"| correct={correct} | invalid={invalid}"
                            )

                    shard = pd.DataFrame(rows)
                    upload_dataframe(storage, bucket, shard_key, shard)
                    all_frames.append(shard)
                    completed_shards.append(shard_id)
                    run_manifest.update(
                        {
                            "updated_at_utc": utc_now(),
                            "completed_shards": completed_shards,
                        }
                    )
                    upload_json(storage, bucket, manifest_key, run_manifest)

            finally:
                if provider is not None:
                    del provider
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                unload_ollama_model(config.generation.base_url, model_name)

        results = pd.concat(all_frames, ignore_index=True)
        expected = len(questions) * len(config_ids) * len(model_names)
        if len(results) != expected:
            raise RuntimeError(
                f"Resultados MCQ incompletos: esperado={expected}; obtido={len(results)}."
            )

        summary = build_summary(results)
        upload_dataframe(storage, bucket, f"{mcq_prefix}/results.parquet", results)
        upload_dataframe(storage, bucket, f"{mcq_prefix}/summary.parquet", summary)

        payload = {
            "run_id": run_id,
            "source_retrieval_run_id": retrieval_run_id,
            "split": "dev",
            "query_variant": "rewritten",
            "enrichment_enabled": bool(args.enrichment),
            "retrieval_configs": len(config_ids),
            "models": len(model_names),
            "base_queries": len(questions),
            "generations": len(results),
            "correct": int(results["is_correct"].sum()),
            "ambiguous": int(results["is_ambiguous"].sum()),
            "wrong": int(results["is_wrong"].sum()),
            "invalid": int(results["is_invalid"].sum()),
            "accuracy": float(results["is_correct"].mean()),
            "ambiguous_rate": float(results["is_ambiguous"].mean()),
            "wrong_rate": float(results["is_wrong"].mean()),
            "invalid_rate": float(results["is_invalid"].mean()),
            "summary_rows": summary.to_dict(orient="records"),
        }
        upload_json(storage, bucket, f"{mcq_prefix}/summary.json", payload)

        run_manifest.update(
            {
                "status": "success",
                "updated_at_utc": utc_now(),
                "completed_at_utc": utc_now(),
                "completed_shards": completed_shards,
                "outputs": {
                    "contexts": context_key,
                    "results": f"{mcq_prefix}/results.parquet",
                    "summary_parquet": f"{mcq_prefix}/summary.parquet",
                    "summary_json": f"{mcq_prefix}/summary.json",
                },
            }
        )
        upload_json(storage, bucket, manifest_key, run_manifest)

        print("\n=== MCQ BENCHMARK OK ===")
        print(f"run_id: {run_id}")
        print(f"source retrieval: {retrieval_run_id}")
        print(f"enrichment: {'on' if args.enrichment else 'off'}")
        if args.enrichment:
            print(f"enrichment top_k: {args.enrichment_top_k}")
        print(f"configs: {len(config_ids)}")
        print(f"models: {len(model_names)}")
        print(f"queries: {len(questions)}")
        print(f"generations: {len(results)}")
        print(f"accuracy: {float(results['is_correct'].mean()):.3f}")
        print(f"ambiguous: {float(results['is_ambiguous'].mean()):.3f}")
        print(f"wrong: {float(results['is_wrong'].mean()):.3f}")
        print(f"invalid: {float(results['is_invalid'].mean()):.3f}")
        print(f"results: s3://{bucket}/{mcq_prefix}/results.parquet")
        print(f"summary: s3://{bucket}/{mcq_prefix}/summary.parquet")
        print(f"manifest: s3://{bucket}/{manifest_key}")

    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "updated_at_utc": utc_now(),
                "completed_shards": completed_shards,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=30),
                },
            }
        )
        upload_json(storage, bucket, manifest_key, run_manifest)
        raise


if __name__ == "__main__":
    main()
