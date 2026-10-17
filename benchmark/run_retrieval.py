#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import re
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from benchmark.matrix import BENCHMARK_PREFIX
from benchmark.prepare import (
    dataframe_parquet_bytes,
    json_bytes,
    load_json,
    make_qdrant_manager,
)
from benchmark.retrieval_metrics import EVAL_KS, query_metrics
from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager

PREPARATION_MANIFEST_KEY = f"{BENCHMARK_PREFIX}/prepared/preparation_manifest.json"
MATRIX_KEY = f"{BENCHMARK_PREFIX}/prepared/matrix.json"
SPLIT_KEY = f"{BENCHMARK_PREFIX}/prepared/splits/benchmark_with_split.csv"


@dataclass(frozen=True)
class RuntimeConfig:
    config_id: str
    experiment_id: str
    silver_dataset_id: str
    chunk_strategy: str
    chunk_size: int
    overlap: int
    retrieval_strategy: str
    embedder: str | None
    index_id: str | None
    collection: str | None


class SilverCorpusCache:
    """Mantém no máximo uma Silver em RAM por vez."""

    def __init__(self, *, storage: RustFSManager, bucket: str) -> None:
        self.storage = storage
        self.bucket = bucket
        self.current_id: str | None = None
        self.current_df: pd.DataFrame | None = None
        self.current_bm25: Any = None
        self.current_tokenized: list[list[str]] | None = None

    def load(self, silver_dataset_id: str) -> pd.DataFrame:
        if self.current_id == silver_dataset_id and self.current_df is not None:
            return self.current_df

        self.clear()
        key = f"datasets/{silver_dataset_id}/chunks.parquet"
        payload = self.storage.download_bytes(self.bucket, key)
        frame = pd.read_parquet(io.BytesIO(payload))
        if frame.empty:
            raise RuntimeError(f"Silver vazia: {silver_dataset_id}")
        if "chunk_id" not in frame.columns or "text" not in frame.columns:
            raise RuntimeError(f"Silver {silver_dataset_id} sem chunk_id/text.")

        self.current_id = silver_dataset_id
        self.current_df = frame.reset_index(drop=True)
        return self.current_df

    def bm25(
        self, silver_dataset_id: str, tokenizer: Callable[[str], list[str]]
    ) -> Any:
        frame = self.load(silver_dataset_id)
        if self.current_bm25 is not None:
            return self.current_bm25

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "Dependência rank_bm25 não encontrada. "
                "Ela é necessária para o sweep BM25."
            ) from exc

        print(
            f"[bm25] construindo corpus | silver={silver_dataset_id} "
            f"| chunks={len(frame)}"
        )
        tokenized = [tokenizer(str(text)) for text in frame["text"].tolist()]
        self.current_tokenized = tokenized
        self.current_bm25 = BM25Okapi(tokenized)
        return self.current_bm25

    def clear(self) -> None:
        self.current_id = None
        self.current_df = None
        self.current_bm25 = None
        self.current_tokenized = None
        gc.collect()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o sweep de retrieval do ACartomanteRAG usando os "
            "artefatos congelados por benchmark.prepare."
        )
    )
    parser.add_argument(
        "--split",
        choices=("dev", "holdout", "all"),
        default="dev",
        help="Por padrão somente DEV. Holdout exige --unlock-holdout.",
    )
    parser.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="Confirma explicitamente execução sobre holdout/all.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "ID da execução. Se omitido, cria um timestamp. Reuse o mesmo "
            "run-id para retomar uma execução interrompida."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pula shards já concluídos no run-id informado.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permite sobrescrever shards existentes do run-id.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke curto: fixed/512, 2 queries, BM25 + Dense Qwen + " "Dense+CE Qwen."
        ),
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Limita casos base após o filtro de split; útil para smoke/debug.",
    )
    parser.add_argument(
        "--query-variants",
        nargs="+",
        choices=("manual", "rewritten"),
        default=("manual", "rewritten"),
    )
    parser.add_argument(
        "--reranker-device",
        default="auto",
        help="auto, cuda, cpu etc. Padrão: auto.",
    )
    parser.add_argument(
        "--selected-from-run-id",
        default=None,
        help=(
            "Filtra o sweep para as 10 configs congeladas em "
            "<run>/retrieval/final_selection_10.json. Para validação final, "
            "use o run DEV que originou a seleção."
        ),
    )
    return parser.parse_args()


def sanitize_run_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    if not value:
        raise ValueError("run_id vazio após sanitização.")
    return value[:160]


def default_run_id(split: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"retrieval_{split}_{stamp}"


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_simple(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return value.casefold()


def fallback_bm25_tokenizer(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_simple(text), flags=re.UNICODE)


def resolve_bm25_tokenizer() -> tuple[Callable[[str], list[str]], str]:
    """Prefere o tokenizer compartilhado do runtime quando houver.

    O fallback existe para manter o benchmark executável sem depender de um
    nome interno específico de src.retrieval.text.
    """
    try:
        from src.retrieval import text as retrieval_text
    except Exception:
        return fallback_bm25_tokenizer, "benchmark.fallback_bm25_tokenizer"

    for name in (
        "tokenize_for_bm25",
        "tokenize_bm25",
        "bm25_tokenize",
        "tokenize_text",
        "tokenize",
    ):
        candidate = getattr(retrieval_text, name, None)
        if not callable(candidate):
            continue
        try:
            probe = candidate("Machado de Assis")
        except Exception:
            continue
        if isinstance(probe, (list, tuple)) and all(
            isinstance(item, str) for item in probe
        ):
            return (
                lambda value, fn=candidate: list(fn(value)),
                f"src.retrieval.text.{name}",
            )

    return fallback_bm25_tokenizer, "benchmark.fallback_bm25_tokenizer"


def load_query_vectors(
    *,
    storage: RustFSManager,
    bucket: str,
    alias: str,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    prefix = f"{BENCHMARK_PREFIX}/prepared/query_embeddings/{alias}"
    meta_key = f"{prefix}/queries.parquet"
    vectors_key = f"{prefix}/vectors.npz"
    manifest_key = f"{prefix}/manifest.json"

    manifest = load_json(storage, bucket, manifest_key)
    metadata = pd.read_parquet(io.BytesIO(storage.download_bytes(bucket, meta_key)))
    npz = np.load(io.BytesIO(storage.download_bytes(bucket, vectors_key)))
    vectors = np.asarray(npz["vectors"], dtype=np.float32)

    if len(metadata) != len(vectors):
        raise RuntimeError(
            f"Query embeddings {alias}: metadata={len(metadata)} "
            f"vectors={len(vectors)}."
        )

    lookup: dict[tuple[str, str], np.ndarray] = {}
    for row_index, row in metadata.reset_index(drop=True).iterrows():
        key = (str(row["query_id"]), str(row["query_variant"]))
        if key in lookup:
            raise RuntimeError(f"Query embedding duplicado: {alias} {key}")
        lookup[key] = vectors[row_index]

    return lookup, manifest


def index_lookup(matrix: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for item in matrix.get("resolved", {}).get("indexes", []):
        alias = str(item.get("model_alias") or "")
        signature = item.get("signature") or {}
        manifest = item.get("manifest") or {}
        source = manifest.get("source") or {}
        silver_id = str(
            signature.get("silver_dataset_id") or source.get("silver_dataset_id") or ""
        )
        if not alias or not silver_id:
            continue
        output[(silver_id, alias)] = item
    return output


def build_runtime_configs(matrix: dict[str, Any]) -> list[RuntimeConfig]:
    indexes = index_lookup(matrix)
    output: list[RuntimeConfig] = []

    for item in matrix.get("resolved", {}).get("retrieval_configs", []):
        strategy = str(item["strategy"])
        embedder = item.get("embedder")
        silver_id = str(item["silver_dataset_id"])
        index_id: str | None = None
        collection: str | None = None

        if embedder is not None:
            resolved = indexes.get((silver_id, str(embedder)))
            if resolved is None:
                raise RuntimeError(
                    "Índice não resolvido na matrix para "
                    f"silver={silver_id}, embedder={embedder}."
                )
            index_id = str(resolved["index_id"])
            collection = str(resolved.get("collection") or index_id)

        chunk_strategy = str(item.get("chunk_strategy") or "unknown")
        chunk_size = int(item.get("chunk_size") or 0)
        overlap = int(item.get("overlap") or 0)
        suffix = strategy if embedder is None else f"{strategy}_{embedder}"
        config_id = f"{chunk_strategy}_c{chunk_size}_o{overlap}__{suffix}"
        signature = {
            "silver_dataset_id": silver_id,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "retrieval_strategy": strategy,
            "embedder": embedder,
            "index_id": index_id,
            "collection": collection,
        }
        experiment_id = sha256_json(signature)[:16]

        output.append(
            RuntimeConfig(
                config_id=config_id,
                experiment_id=experiment_id,
                silver_dataset_id=silver_id,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                retrieval_strategy=strategy,
                embedder=(str(embedder) if embedder is not None else None),
                index_id=index_id,
                collection=collection,
            )
        )

    if len(output) != 30:
        raise RuntimeError(
            f"Esperadas 30 configs de retrieval; encontradas {len(output)}."
        )
    return output


def smoke_filter(configs: list[RuntimeConfig]) -> list[RuntimeConfig]:
    wanted = {
        ("bm25", None),
        ("dense", "qwen3_embedding_06b"),
        ("dense_cross_encoder", "qwen3_embedding_06b"),
    }
    return [
        item
        for item in configs
        if item.chunk_strategy == "fixed"
        and item.chunk_size == 512
        and (item.retrieval_strategy, item.embedder) in wanted
    ]


def filter_selected_configs(
    configs: list[RuntimeConfig],
    *,
    selection: dict[str, Any],
    benchmark_sha256: str,
) -> tuple[list[RuntimeConfig], list[str]]:
    if str(selection.get("split") or "") != "dev":
        raise RuntimeError(
            "A seleção final usada no holdout precisa ter sido congelada no DEV."
        )
    if str(selection.get("benchmark_sha256") or "") != str(benchmark_sha256):
        raise RuntimeError(
            "Hash do benchmark diverge entre preparação e final_selection_10.json."
        )

    rows = selection.get("selected") or []
    selected_ids = [str(row.get("config_id") or "").strip() for row in rows]
    if len(selected_ids) != 10 or any(not item for item in selected_ids):
        raise RuntimeError(
            "final_selection_10.json deve conter exatamente 10 config_id válidos."
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("final_selection_10.json contém config_id duplicado.")

    lookup = {item.config_id: item for item in configs}
    missing = [item for item in selected_ids if item not in lookup]
    if missing:
        raise RuntimeError(
            "Configs selecionadas não existem na matrix preparada: "
            + ", ".join(missing)
        )

    return [lookup[item] for item in selected_ids], selected_ids


def qdrant_search(
    client: Any,
    *,
    collection: str,
    vector: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    query_vector = vector.astype(np.float32, copy=False).tolist()

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

    output: list[dict[str, Any]] = []
    for point in points:
        payload = dict(getattr(point, "payload", None) or {})
        output.append(
            {
                **payload,
                "retrieval_score": float(getattr(point, "score", 0.0)),
                "rerank_score": None,
                "qdrant_point_id": str(getattr(point, "id", "")),
            }
        )
    return output


def bm25_search(
    *,
    frame: pd.DataFrame,
    bm25: Any,
    query: str,
    tokenizer: Callable[[str], list[str]],
    top_k: int,
) -> list[dict[str, Any]]:
    scores = np.asarray(bm25.get_scores(tokenizer(query)), dtype=np.float64)
    if len(scores) != len(frame):
        raise RuntimeError("BM25 retornou score count diferente do corpus.")

    order = np.argsort(-scores, kind="stable")[:top_k]
    output: list[dict[str, Any]] = []
    for row_index in order.tolist():
        payload = frame.iloc[row_index].to_dict()
        output.append(
            {
                **payload,
                "retrieval_score": float(scores[row_index]),
                "rerank_score": None,
                "qdrant_point_id": None,
            }
        )
    return output


class LazyCrossEncoder:
    def __init__(self, *, model: str, revision: str, device: str) -> None:
        self.model_name = model
        self.revision = revision
        self.requested_device = device
        self.resolved_device = self._resolve_device(device)
        self._model: Any = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers é necessário para Dense+CrossEncoder."
            ) from exc

        print(f"[reranker] loading {self.model_name} | device={self.resolved_device}")
        kwargs: dict[str, Any] = {
            "device": self.resolved_device,
        }
        if self.revision:
            kwargs["revision"] = self.revision
        self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        model = self._ensure()
        pairs = [(query, str(item.get("text") or "")) for item in candidates]
        scores = np.asarray(
            model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        ).reshape(-1)
        if len(scores) != len(candidates):
            raise RuntimeError("CrossEncoder retornou quantidade inválida de scores.")

        order = np.argsort(-scores, kind="stable")[:top_k]
        output: list[dict[str, Any]] = []
        for candidate_index in order.tolist():
            item = dict(candidates[candidate_index])
            item["rerank_score"] = float(scores[candidate_index])
            output.append(item)
        return output


def row_json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def compact_retrieved_item(item: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "chunk_id",
        "silver_dataset_id",
        "source_document_id",
        "titulo",
        "autor",
        "categoria",
        "requested_strategy",
        "effective_strategy",
        "section_index",
        "section_type",
        "section_number",
        "section_header",
        "section_title",
        "chunk_index",
        "token_count",
        "start_char",
        "end_char",
        "text",
        "retrieval_score",
        "rerank_score",
        "qdrant_point_id",
    )
    return {field: row_json_safe(item.get(field)) for field in fields}


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else 0.0


def build_summary(query_df: pd.DataFrame) -> pd.DataFrame:
    if query_df.empty:
        return pd.DataFrame()

    group_columns = [
        "config_id",
        "experiment_id",
        "chunk_strategy",
        "chunk_size",
        "overlap",
        "retrieval_strategy",
        "embedder",
        "query_variant",
    ]

    rows: list[dict[str, Any]] = []
    for key, group in query_df.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, key))
        row["queries"] = len(group)
        for k in EVAL_KS:
            row[f"evidence_coverage_at_{k}"] = float(
                group[f"evidence_coverage_at_{k}"].mean()
            )
            row[f"hit_at_{k}"] = float(group[f"hit_at_{k}"].mean())
            row[f"correct_work_at_{k}"] = float(group[f"correct_work_at_{k}"].mean())

        row["mrr_at_10"] = float(group["mrr_at_10"].mean())
        row["ndcg_at_10"] = float(group["ndcg_at_10"].mean())
        row["coverage_auc_at_10"] = float(group["coverage_auc_at_10"].mean())

        for latency in (
            "retrieval_latency_ms",
            "qdrant_latency_ms",
            "rerank_latency_ms",
        ):
            row[f"{latency}_mean"] = float(
                pd.to_numeric(group[latency], errors="coerce").dropna().mean()
                if pd.to_numeric(group[latency], errors="coerce").notna().any()
                else 0.0
            )
            row[f"{latency}_p50"] = percentile(group[latency], 0.50)
            row[f"{latency}_p95"] = percentile(group[latency], 0.95)

        rows.append(row)

    return pd.DataFrame(rows)


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


def load_shard(
    storage: RustFSManager,
    bucket: str,
    key: str,
) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(storage.download_bytes(bucket, key)))


def execute_one_query(
    *,
    runtime: RuntimeConfig,
    row: Any,
    query_variant: str,
    query: str,
    config: Any,
    qdrant_client: Any,
    corpus_cache: SilverCorpusCache,
    bm25_tokenizer: Callable[[str], list[str]],
    vector_lookups: dict[str, dict[tuple[str, str], np.ndarray]],
    reranker: LazyCrossEncoder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qdrant_ms: float | None = None
    rerank_ms: float | None = None
    start_total = time.perf_counter()

    if runtime.retrieval_strategy == "bm25":
        frame = corpus_cache.load(runtime.silver_dataset_id)
        bm25 = corpus_cache.bm25(runtime.silver_dataset_id, bm25_tokenizer)
        start = time.perf_counter()
        retrieved = bm25_search(
            frame=frame,
            bm25=bm25,
            query=query,
            tokenizer=bm25_tokenizer,
            top_k=config.retrievals.bm25.top_k,
        )
        qdrant_ms = None
        retrieval_core_ms = (time.perf_counter() - start) * 1000.0

    elif runtime.retrieval_strategy in {"dense", "dense_cross_encoder"}:
        if runtime.embedder is None or runtime.collection is None:
            raise RuntimeError(f"Config Dense sem embedder/index: {runtime.config_id}")
        lookup = vector_lookups[runtime.embedder]
        vector_key = (str(row.query_id), query_variant)
        if vector_key not in lookup:
            raise RuntimeError(f"Query vector ausente: {runtime.embedder} {vector_key}")

        candidate_k = (
            config.retrievals.dense.top_k
            if runtime.retrieval_strategy == "dense"
            else config.retrievals.dense_cross_encoder.candidate_k
        )

        start = time.perf_counter()
        candidates = qdrant_search(
            qdrant_client,
            collection=runtime.collection,
            vector=lookup[vector_key],
            limit=candidate_k,
        )
        qdrant_ms = (time.perf_counter() - start) * 1000.0
        retrieval_core_ms = qdrant_ms

        if runtime.retrieval_strategy == "dense":
            retrieved = candidates[: config.retrievals.dense.top_k]
        else:
            start = time.perf_counter()
            retrieved = reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=config.retrievals.dense_cross_encoder.top_k,
                batch_size=config.embedding.execution.batch_size,
            )
            rerank_ms = (time.perf_counter() - start) * 1000.0
    else:
        raise ValueError(f"Retrieval não suportado: {runtime.retrieval_strategy}")

    total_ms = (time.perf_counter() - start_total) * 1000.0

    compact = [compact_retrieved_item(item) for item in retrieved]
    rank_rows, metrics = query_metrics(
        gold_evidence=str(row.evidence_text),
        canonical_work=str(row.canonical_work),
        retrieved=compact,
        evidence_min_coverage=config.retrieval_benchmark.evidence_min_coverage,
    )

    query_metadata = {
        "retrieval_latency_ms": total_ms,
        "retrieval_core_latency_ms": retrieval_core_ms,
        "qdrant_latency_ms": qdrant_ms,
        "rerank_latency_ms": rerank_ms,
        "query_embedding_cached": runtime.embedder is not None,
        "query_embedding_latency_ms": None,
        **metrics,
    }
    return rank_rows, query_metadata


def main() -> None:
    args = parse_args()
    if args.split in {"holdout", "all"} and not args.unlock_holdout:
        raise SystemExit(
            "Holdout protegido. Use --unlock-holdout somente quando a seleção "
            "dos finalistas estiver congelada."
        )
    if args.limit_queries is not None and args.limit_queries <= 0:
        raise SystemExit("--limit-queries deve ser > 0.")
    if args.smoke and args.selected_from_run_id:
        raise SystemExit(
            "Use --smoke sem --selected-from-run-id; a seleção final é destinada "
            "à execução congelada."
        )

    config = ConfigManager.load()
    storage = RustFSManager.from_config(config.environment.rustfs)
    bucket = config.storage.benchmarks_bucket

    for required_key in (PREPARATION_MANIFEST_KEY, MATRIX_KEY, SPLIT_KEY):
        if not storage.object_exists(bucket, required_key):
            raise FileNotFoundError(
                f"Preparação incompleta: s3://{bucket}/{required_key} ausente."
            )

    prep = load_json(storage, bucket, PREPARATION_MANIFEST_KEY)
    matrix = load_json(storage, bucket, MATRIX_KEY)
    split_df = pd.read_csv(io.BytesIO(storage.download_bytes(bucket, SPLIT_KEY)))

    if prep.get("status") != "success":
        raise RuntimeError("preparation_manifest não está com status=success.")
    if prep.get("benchmark_sha256") != matrix.get("benchmark_sha256"):
        raise RuntimeError("Hash do benchmark diverge entre preparation e matrix.")

    if args.split != "all":
        split_df = split_df[split_df["split"].eq(args.split)].copy()
    if args.limit_queries is not None:
        split_df = split_df.head(args.limit_queries).copy()
    if args.smoke:
        split_df = split_df.head(2).copy()

    if split_df.empty:
        raise RuntimeError("Nenhuma query após filtros de split/limit.")

    configs = build_runtime_configs(matrix)
    selection_source: dict[str, Any] | None = None
    selected_config_ids: list[str] | None = None
    selection_key: str | None = None
    selection_run_id: str | None = None

    if args.selected_from_run_id:
        selection_run_id = sanitize_run_id(args.selected_from_run_id)
        selection_key = (
            f"{BENCHMARK_PREFIX}/runs/{selection_run_id}/"
            "retrieval/final_selection_10.json"
        )
        if not storage.object_exists(bucket, selection_key):
            raise FileNotFoundError(
                f"Seleção final ausente: s3://{bucket}/{selection_key}"
            )
        selection_source = load_json(storage, bucket, selection_key)
        configs, selected_config_ids = filter_selected_configs(
            configs,
            selection=selection_source,
            benchmark_sha256=prep["benchmark_sha256"],
        )
        print(f"[selection] run={selection_run_id} | configs={len(configs)}")

    if args.smoke:
        configs = smoke_filter(configs)
        if len(configs) != 3:
            raise RuntimeError(f"Smoke esperava 3 configs; encontrou {len(configs)}.")

    query_variants = tuple(args.query_variants)
    run_id = sanitize_run_id(args.run_id or default_run_id(args.split))
    run_prefix = f"{BENCHMARK_PREFIX}/runs/{run_id}"
    retrieval_prefix = f"{run_prefix}/retrieval"

    existing_manifest_key = f"{run_prefix}/manifest.json"
    if storage.object_exists(bucket, existing_manifest_key) and not (
        args.resume or args.force
    ):
        raise RuntimeError(
            f"run_id já existe: {run_id}. Use --resume ou outro --run-id."
        )

    bm25_tokenizer, bm25_tokenizer_name = resolve_bm25_tokenizer()
    qdrant = make_qdrant_manager(config)
    qdrant_client = getattr(qdrant, "client", qdrant)

    required_embedders = sorted(
        {item.embedder for item in configs if item.embedder is not None}
    )
    vector_lookups: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    vector_manifests: dict[str, dict[str, Any]] = {}
    for alias in required_embedders:
        assert alias is not None
        lookup, vector_manifest = load_query_vectors(
            storage=storage,
            bucket=bucket,
            alias=alias,
        )
        if vector_manifest.get("benchmark_sha256") != prep.get("benchmark_sha256"):
            raise RuntimeError(f"Query embeddings {alias} pertencem a outro benchmark.")
        vector_lookups[alias] = lookup
        vector_manifests[alias] = vector_manifest

    reranker = LazyCrossEncoder(
        model=config.retrievals.dense_cross_encoder.model,
        revision=config.retrievals.dense_cross_encoder.revision,
        device=args.reranker_device,
    )
    corpus_cache = SilverCorpusCache(
        storage=storage,
        bucket=config.storage.silver_bucket,
    )

    run_config = {
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "phase": "retrieval",
        "split": args.split,
        "smoke": args.smoke,
        "query_variants": list(query_variants),
        "base_queries": len(split_df),
        "query_executions_per_config": len(split_df) * len(query_variants),
        "configs": len(configs),
        "selection_source_run_id": selection_run_id,
        "selection_source_key": selection_key,
        "selected_config_ids": selected_config_ids,
        "selection_method": (
            selection_source.get("method") if selection_source is not None else None
        ),
        "expected_retrieval_executions": (
            len(split_df) * len(query_variants) * len(configs)
        ),
        "benchmark_sha256": prep["benchmark_sha256"],
        "matrix_key": MATRIX_KEY,
        "preparation_manifest_key": PREPARATION_MANIFEST_KEY,
        "evidence_min_coverage": config.retrieval_benchmark.evidence_min_coverage,
        "warmup_runs": config.retrieval_benchmark.warmup_runs,
        "top_k": {
            "bm25": config.retrievals.bm25.top_k,
            "dense": config.retrievals.dense.top_k,
            "dense_cross_encoder": config.retrievals.dense_cross_encoder.top_k,
            "dense_cross_encoder_candidate_k": (
                config.retrievals.dense_cross_encoder.candidate_k
            ),
        },
        "bm25_tokenizer": bm25_tokenizer_name,
        "reranker": {
            "model": reranker.model_name,
            "revision": reranker.revision,
            "requested_device": reranker.requested_device,
            "resolved_device": reranker.resolved_device,
        },
        "query_embeddings": vector_manifests,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "retrieval_configs": [item.__dict__ for item in configs],
    }
    storage.upload_bytes(
        bucket,
        f"{run_prefix}/config.json",
        json_bytes(run_config),
        content_type="application/json",
    )

    completed: list[str] = []
    run_manifest = {
        "run_id": run_id,
        "phase": "retrieval",
        "status": "running",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "benchmark_sha256": prep["benchmark_sha256"],
        "split": args.split,
        "configs_total": len(configs),
        "configs_completed": 0,
        "completed_config_ids": completed,
    }
    storage.upload_bytes(
        bucket,
        existing_manifest_key,
        json_bytes(run_manifest),
        content_type="application/json",
    )

    all_rank_frames: list[pd.DataFrame] = []
    all_query_frames: list[pd.DataFrame] = []

    try:
        total_configs = len(configs)
        for config_number, runtime in enumerate(configs, start=1):
            shard_base = f"{retrieval_prefix}/shards/{runtime.config_id}"
            rank_key = f"{shard_base}.results.parquet"
            query_key = f"{shard_base}.queries.parquet"

            shard_exists = storage.object_exists(
                bucket, rank_key
            ) and storage.object_exists(bucket, query_key)
            if shard_exists and args.resume and not args.force:
                print(
                    f"[{config_number}/{total_configs}] [resume] "
                    f"{runtime.config_id}"
                )
                rank_df = load_shard(storage, bucket, rank_key)
                query_df = load_shard(storage, bucket, query_key)
                all_rank_frames.append(rank_df)
                all_query_frames.append(query_df)
                completed.append(runtime.config_id)
                continue
            if shard_exists and not args.force:
                raise RuntimeError(
                    f"Shard já existe para {runtime.config_id}; "
                    "use --resume ou --force."
                )

            print(
                f"\n[{config_number}/{total_configs}] {runtime.config_id} "
                f"| queries={len(split_df) * len(query_variants)}"
            )

            # Warm-up descartado das métricas.
            if config.retrieval_benchmark.warmup_runs > 0:
                warm_row = next(split_df.itertuples(index=False))
                warm_variant = query_variants[0]
                warm_query = str(
                    getattr(
                        warm_row,
                        (
                            "query_manual"
                            if warm_variant == "manual"
                            else "query_rewritten"
                        ),
                    )
                )
                for _ in range(config.retrieval_benchmark.warmup_runs):
                    execute_one_query(
                        runtime=runtime,
                        row=warm_row,
                        query_variant=warm_variant,
                        query=warm_query,
                        config=config,
                        qdrant_client=qdrant_client,
                        corpus_cache=corpus_cache,
                        bm25_tokenizer=bm25_tokenizer,
                        vector_lookups=vector_lookups,
                        reranker=reranker,
                    )

            rank_rows_acc: list[dict[str, Any]] = []
            query_rows_acc: list[dict[str, Any]] = []

            done = 0
            total = len(split_df) * len(query_variants)
            for benchmark_row in split_df.itertuples(index=False):
                for query_variant in query_variants:
                    query_column = (
                        "query_manual"
                        if query_variant == "manual"
                        else "query_rewritten"
                    )
                    query = str(getattr(benchmark_row, query_column))

                    rank_rows, metrics = execute_one_query(
                        runtime=runtime,
                        row=benchmark_row,
                        query_variant=query_variant,
                        query=query,
                        config=config,
                        qdrant_client=qdrant_client,
                        corpus_cache=corpus_cache,
                        bm25_tokenizer=bm25_tokenizer,
                        vector_lookups=vector_lookups,
                        reranker=reranker,
                    )

                    base = {
                        "run_id": run_id,
                        "config_id": runtime.config_id,
                        "experiment_id": runtime.experiment_id,
                        "query_id": str(benchmark_row.query_id),
                        "split": str(benchmark_row.split),
                        "query_variant": query_variant,
                        "query": query,
                        "canonical_work": str(benchmark_row.canonical_work),
                        "difficulty": str(benchmark_row.difficulty),
                        "query_type": str(benchmark_row.query_type),
                        "gold_answer": str(benchmark_row.gold_answer),
                        "evidence_text": str(benchmark_row.evidence_text),
                        "chunk_strategy": runtime.chunk_strategy,
                        "chunk_size": runtime.chunk_size,
                        "overlap": runtime.overlap,
                        "retrieval_strategy": runtime.retrieval_strategy,
                        "embedder": runtime.embedder,
                        "silver_dataset_id": runtime.silver_dataset_id,
                        "index_id": runtime.index_id,
                    }

                    query_rows_acc.append({**base, **metrics})
                    for rank_row in rank_rows:
                        rank_rows_acc.append({**base, **rank_row})

                    done += 1
                    if done % 20 == 0 or done == total:
                        print(f"  progress {done}/{total}")

            rank_df = pd.DataFrame(rank_rows_acc)
            query_df = pd.DataFrame(query_rows_acc)
            upload_dataframe(storage, bucket, rank_key, rank_df)
            upload_dataframe(storage, bucket, query_key, query_df)
            all_rank_frames.append(rank_df)
            all_query_frames.append(query_df)
            completed.append(runtime.config_id)

            run_manifest.update(
                {
                    "updated_at_utc": utc_now(),
                    "configs_completed": len(completed),
                    "completed_config_ids": completed,
                }
            )
            storage.upload_bytes(
                bucket,
                existing_manifest_key,
                json_bytes(run_manifest),
                content_type="application/json",
            )

        results_df = pd.concat(all_rank_frames, ignore_index=True)
        queries_df = pd.concat(all_query_frames, ignore_index=True)
        summary_df = build_summary(queries_df)

        upload_dataframe(
            storage,
            bucket,
            f"{retrieval_prefix}/results.parquet",
            results_df,
        )
        upload_dataframe(
            storage,
            bucket,
            f"{retrieval_prefix}/query_metrics.parquet",
            queries_df,
        )
        upload_dataframe(
            storage,
            bucket,
            f"{retrieval_prefix}/summary.parquet",
            summary_df,
        )

        summary_payload = {
            "run_id": run_id,
            "split": args.split,
            "configs": len(configs),
            "base_queries": len(split_df),
            "query_variants": list(query_variants),
            "retrieval_executions": len(queries_df),
            "rank_rows": len(results_df),
            "summary_rows": summary_df.to_dict(orient="records"),
        }
        storage.upload_bytes(
            bucket,
            f"{retrieval_prefix}/summary.json",
            json_bytes(summary_payload),
            content_type="application/json",
        )

        run_manifest.update(
            {
                "status": "success",
                "updated_at_utc": utc_now(),
                "completed_at_utc": utc_now(),
                "configs_completed": len(completed),
                "completed_config_ids": completed,
                "outputs": {
                    "results": f"{retrieval_prefix}/results.parquet",
                    "query_metrics": f"{retrieval_prefix}/query_metrics.parquet",
                    "summary_parquet": f"{retrieval_prefix}/summary.parquet",
                    "summary_json": f"{retrieval_prefix}/summary.json",
                },
            }
        )
        storage.upload_bytes(
            bucket,
            existing_manifest_key,
            json_bytes(run_manifest),
            content_type="application/json",
        )

        print("\n=== RETRIEVAL BENCHMARK OK ===")
        print(f"run_id: {run_id}")
        print(f"split: {args.split}")
        print(f"configs: {len(configs)}")
        print(f"retrieval executions: {len(queries_df)}")
        print(f"rank rows: {len(results_df)}")
        print("results: " f"s3://{bucket}/{retrieval_prefix}/results.parquet")
        print("summary: " f"s3://{bucket}/{retrieval_prefix}/summary.parquet")
        print("manifest: " f"s3://{bucket}/{existing_manifest_key}")

    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "updated_at_utc": utc_now(),
                "configs_completed": len(completed),
                "completed_config_ids": completed,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
            }
        )
        storage.upload_bytes(
            bucket,
            existing_manifest_key,
            json_bytes(run_manifest),
            content_type="application/json",
        )
        raise
    finally:
        corpus_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
