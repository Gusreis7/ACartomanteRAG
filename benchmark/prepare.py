#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmark.matrix import (
    BENCHMARK_PREFIX,
    BENCHMARK_SOURCE_KEY,
    CHUNK_VARIANTS,
    DEV_RATIO,
    EMBEDDERS,
    HOLDOUT_RATIO,
    OLLAMA_MODELS,
    RETRIEVALS,
    SPLIT_SEED,
)
from src.config.config_manager import AppConfig, ConfigManager
from src.etl.preparar_camada_silver_chunking import (
    build_dataset_id,
    salvar_parquet,
)
from src.storage.qdrant_manager import QdrantManager
from src.storage.rustfs_manager import RustFSManager
from src.utils.chunking import HuggingFaceChunkTokenizer, build_sections

if TYPE_CHECKING:
    from src.indexing.embedding_index import EmbeddingIndex


BRONZE_CURRENT_KEY = "current.json"


REQUIRED_BENCHMARK_COLUMNS = {
    "query_id",
    "canonical_work",
    "query_type",
    "difficulty",
    "evidence_text",
    "gold_answer",
    "query_manual",
    "query_rewritten",
    "correct_option",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara os artefatos compartilhados do benchmark completo: "
            "dataset canônico, split, 6 Silvers, 12 índices Dense e "
            "embeddings das queries."
        )
    )
    parser.add_argument(
        "--benchmark-key",
        default=BENCHMARK_SOURCE_KEY,
        help="Objeto no bucket acartomante-benchmarks usado como fonte.",
    )
    parser.add_argument(
        "--skip-silver",
        action="store_true",
        help="Não resolve/cria as 6 Silvers.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Não resolve/cria os 12 índices Dense.",
    )
    parser.add_argument(
        "--skip-query-embeddings",
        action="store_true",
        help="Não pré-computa embeddings das queries.",
    )
    parser.add_argument(
        "--force-prepared",
        action="store_true",
        help=(
            "Permite substituir split/query embeddings preparados para um "
            "hash de benchmark diferente."
        ),
    )
    return parser.parse_args()


def validate_benchmark(df: pd.DataFrame) -> None:
    missing = REQUIRED_BENCHMARK_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Benchmark sem colunas obrigatórias: " + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("Benchmark vazio.")

    if df["query_id"].isna().any():
        raise ValueError("Benchmark possui query_id nulo.")

    if df["query_id"].duplicated().any():
        duplicated = df.loc[
            df["query_id"].duplicated(keep=False), "query_id"
        ].astype(str).tolist()
        raise ValueError(f"query_id duplicado: {duplicated[:10]}")


def stable_rank(seed: int, *parts: str) -> str:
    raw = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def allocate_holdout_by_difficulty(
    group: pd.DataFrame,
    target: int,
) -> dict[str, int]:
    counts = group["difficulty"].astype(str).value_counts().to_dict()
    total = len(group)

    raw = {
        key: target * count / total
        for key, count in counts.items()
    }
    allocation = {
        key: min(counts[key], int(math.floor(value)))
        for key, value in raw.items()
    }

    remaining = target - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(raw[key] - math.floor(raw[key])),
            -counts[key],
            key,
        ),
    )

    while remaining > 0:
        progressed = False
        for key in order:
            if allocation[key] < counts[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Não foi possível completar a alocação holdout.")

    return allocation


def build_split(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["split"] = "dev"

    for work, indexes in result.groupby("canonical_work", sort=True).groups.items():
        group = result.loc[list(indexes)].copy()
        target = int(round(len(group) * HOLDOUT_RATIO))
        target = max(1, min(len(group) - 1, target)) if len(group) > 1 else 0

        allocation = allocate_holdout_by_difficulty(group, target)

        selected: list[int] = []
        for difficulty, count in sorted(allocation.items()):
            if count <= 0:
                continue
            subset = group[group["difficulty"].astype(str).eq(difficulty)].copy()
            subset["_rank"] = [
                stable_rank(
                    SPLIT_SEED,
                    str(work),
                    str(difficulty),
                    str(query_id),
                )
                for query_id in subset["query_id"]
            ]
            selected.extend(
                subset.sort_values("_rank").head(count).index.tolist()
            )

        result.loc[selected, "split"] = "holdout"

    # Ajuste determinístico caso os arredondamentos por obra não fechem 40/60.
    target_holdout = int(round(len(result) * HOLDOUT_RATIO))
    current_holdout = int(result["split"].eq("holdout").sum())

    if current_holdout != target_holdout:
        if current_holdout < target_holdout:
            candidates = result[result["split"].eq("dev")].copy()
            direction = "holdout"
            amount = target_holdout - current_holdout
        else:
            candidates = result[result["split"].eq("holdout")].copy()
            direction = "dev"
            amount = current_holdout - target_holdout

        candidates["_rank"] = [
            stable_rank(SPLIT_SEED, "global-adjust", str(query_id))
            for query_id in candidates["query_id"]
        ]
        chosen = candidates.sort_values("_rank").head(amount).index
        result.loc[chosen, "split"] = direction

    return result


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "counts": df["split"].value_counts().sort_index().to_dict(),
        "by_work": (
            df.groupby(["canonical_work", "split"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
            .to_dict(orient="index")
        ),
        "by_difficulty": (
            df.groupby(["difficulty", "split"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
            .to_dict(orient="index")
        ),
    }


def load_json(storage: RustFSManager, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(storage.download_bytes(bucket, key).decode("utf-8"))


def ensure_split(
    storage: RustFSManager,
    bucket: str,
    benchmark_df: pd.DataFrame,
    benchmark_sha256: str,
    *,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_key = f"{BENCHMARK_PREFIX}/prepared/splits/benchmark_with_split.csv"
    manifest_key = f"{BENCHMARK_PREFIX}/prepared/splits/manifest.json"

    if storage.object_exists(bucket, manifest_key) and storage.object_exists(bucket, csv_key):
        old_manifest = load_json(storage, bucket, manifest_key)
        if (
            old_manifest.get("benchmark_sha256") == benchmark_sha256
            and old_manifest.get("seed") == SPLIT_SEED
        ):
            print("[reuse] split 60/40 já preparado.")
            split_df = pd.read_csv(io.BytesIO(storage.download_bytes(bucket, csv_key)))
            return split_df, old_manifest

        if not force:
            raise RuntimeError(
                "Já existe split preparado para outro benchmark/seed. "
                "Use --force-prepared apenas se a substituição for intencional."
            )

    split_df = build_split(benchmark_df)
    summary = summarize_split(split_df)
    manifest = {
        "created_at_utc": utc_now(),
        "benchmark_sha256": benchmark_sha256,
        "seed": SPLIT_SEED,
        "dev_ratio": DEV_RATIO,
        "holdout_ratio": HOLDOUT_RATIO,
        "method": "stratified_by_work_then_difficulty_with_deterministic_hash",
        "summary": summary,
    }

    storage.upload_bytes(
        bucket,
        csv_key,
        dataframe_csv_bytes(split_df),
        content_type="text/csv; charset=utf-8",
    )
    storage.upload_bytes(
        bucket,
        manifest_key,
        json_bytes(manifest),
        content_type="application/json",
    )
    print(
        "[create] split salvo: "
        f"dev={summary['counts'].get('dev', 0)} | "
        f"holdout={summary['counts'].get('holdout', 0)}"
    )
    return split_df, manifest


def config_for_chunk_variant(base: AppConfig, variant: dict[str, Any]) -> AppConfig:
    chunking = base.silver.chunking.model_copy(
        update={
            "strategy": variant["strategy"],
            "chunk_size": variant["chunk_size"],
            "overlap": variant["overlap"],
        }
    )
    silver = base.silver.model_copy(update={"chunking": chunking})
    return base.model_copy(update={"silver": silver}, deep=True)


def resolve_bronze_source(
    config: AppConfig,
    storage: RustFSManager,
) -> dict[str, Any]:
    bucket = config.storage.bronze_bucket

    current_bytes = storage.download_bytes(
        bucket,
        BRONZE_CURRENT_KEY,
    )

    try:
        current = json.loads(current_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "current.json da Bronze não é um JSON válido: "
            f"s3://{bucket}/{BRONZE_CURRENT_KEY}"
        ) from exc

    bronze_artifact_id = str(
        current.get("bronze_artifact_id", "")
    ).strip()
    manifest_key = str(
        current.get("manifest_key", "")
    ).strip()

    if not bronze_artifact_id or not manifest_key:
        raise RuntimeError(
            "current.json da Bronze inválido: "
            "bronze_artifact_id/manifest_key ausentes."
        )

    manifest_bytes = storage.download_bytes(
        bucket,
        manifest_key,
    )
    manifest_sha256 = sha256_bytes(manifest_bytes)
    expected_sha256 = str(
        current.get("manifest_sha256", "")
    ).strip()

    if expected_sha256 and manifest_sha256 != expected_sha256:
        raise RuntimeError(
            "Falha de integridade no manifest Bronze. "
            f"Esperado={expected_sha256}; "
            f"obtido={manifest_sha256}."
        )

    return {
        "bronze_artifact_id": bronze_artifact_id,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
        "manifest_bytes": manifest_bytes,
    }


def load_bronze_manifest(
    config: AppConfig,
    bronze_source: dict[str, Any],
) -> pd.DataFrame:
    bronze_bytes = bronze_source["manifest_bytes"]
    manifest_key = bronze_source["manifest_key"]
    bronze_artifact_id = bronze_source["bronze_artifact_id"]

    if not (
        len(bronze_bytes) >= 8
        and bronze_bytes[:4] == b"PAR1"
        and bronze_bytes[-4:] == b"PAR1"
    ):
        preview = bronze_bytes[:32]
        raise RuntimeError(
            "Manifest Bronze não é um Parquet válido. "
            f"Objeto=s3://{config.storage.bronze_bucket}/{manifest_key} | "
            f"bytes={len(bronze_bytes)} | início={preview!r}."
        )

    try:
        bronze_manifest = pd.read_parquet(io.BytesIO(bronze_bytes))
    except Exception as exc:
        raise RuntimeError(
            "Falha lendo o manifest Bronze como Parquet: "
            f"s3://{config.storage.bronze_bucket}/{manifest_key}"
        ) from exc

    if "bronze_artifact_id" not in bronze_manifest.columns:
        raise RuntimeError(
            "Manifest Bronze sem coluna bronze_artifact_id. "
            "O benchmark exige o contrato atual da camada Bronze."
        )

    manifest_artifact_ids = set(
        bronze_manifest["bronze_artifact_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if manifest_artifact_ids != {bronze_artifact_id}:
        raise RuntimeError(
            "bronze_artifact_id do manifest não corresponde ao current.json. "
            f"current={bronze_artifact_id}; "
            f"manifest={sorted(manifest_artifact_ids)}."
        )

    if "status_bronze" in bronze_manifest.columns:
        bronze_manifest = (
            bronze_manifest[bronze_manifest["status_bronze"].eq("success")]
            .copy()
            .reset_index(drop=True)
        )

    if bronze_manifest.empty:
        raise RuntimeError("Manifest Bronze não possui documentos processáveis.")

    return bronze_manifest


def silver_identity(
    config: AppConfig,
    storage: RustFSManager,
) -> tuple[
    str,
    str,
    HuggingFaceChunkTokenizer,
    dict[str, Any],
]:
    bronze_source = resolve_bronze_source(
        config,
        storage,
    )

    chunking = config.silver.chunking
    tokenizer_config = config.silver.tokenizers[chunking.tokenizer]
    tokenizer = HuggingFaceChunkTokenizer(
        model=tokenizer_config.model,
        revision=tokenizer_config.revision,
    )
    semantic = config.silver.semantic

    dataset_id, config_hash = build_dataset_id(
        pipeline_version=config.silver.pipeline_version,
        strategy=chunking.strategy,
        tokenizer_alias=chunking.tokenizer,
        chunk_size=chunking.chunk_size,
        overlap=chunking.overlap,
        tokenizer_model=tokenizer_config.model,
        tokenizer_revision=tokenizer.resolved_revision,
        bronze_artifact_id=bronze_source["bronze_artifact_id"],
        semantic_config={
            "min_capitulos": semantic.min_capitulos,
            "min_mediana_chars": semantic.min_mediana_chars,
        },
    )

    return (
        dataset_id,
        config_hash,
        tokenizer,
        bronze_source,
    )


def ensure_silver(
    config: AppConfig,
    storage: RustFSManager,
) -> dict[str, Any]:
    (
        dataset_id,
        config_hash,
        tokenizer,
        bronze_source,
    ) = silver_identity(config, storage)

    bucket = config.storage.silver_bucket
    prefix = f"datasets/{dataset_id}"
    chunks_key = f"{prefix}/chunks.parquet"
    manifest_key = f"{prefix}/manifest.parquet"

    chunks_exists = storage.object_exists(bucket, chunks_key)
    manifest_exists = storage.object_exists(bucket, manifest_key)

    if chunks_exists and manifest_exists:
        manifest = pd.read_parquet(
            io.BytesIO(
                storage.download_bytes(bucket, manifest_key)
            )
        )
        print(f"[reuse] Silver {dataset_id}")
        return {
            "status": "reused",
            "silver_dataset_id": dataset_id,
            "manifest": (
                manifest.iloc[0].to_dict()
                if not manifest.empty
                else {}
            ),
        }

    if chunks_exists != manifest_exists:
        raise RuntimeError(
            "Silver inconsistente: apenas um dos artefatos existe. "
            f"dataset_id={dataset_id}"
        )

    print(f"[create] Silver ausente; gerando {dataset_id}")

    bronze_manifest = load_bronze_manifest(
        config,
        bronze_source,
    )
    chunking = config.silver.chunking
    tokenizer_config = config.silver.tokenizers[chunking.tokenizer]
    semantic = config.silver.semantic
    chunks: list[dict[str, Any]] = []

    for row_index, row in bronze_manifest.iterrows():
        title = str(row.get("titulo", ""))
        category = str(row.get("categoria", ""))
        bronze_object_key = row.get("bronze_object_key")

        if pd.isna(bronze_object_key) or not bronze_object_key:
            print(f"[warn] Bronze sem object key: {title}")
            continue

        text = storage.download_bytes(
            config.storage.bronze_bucket,
            str(bronze_object_key),
        ).decode("utf-8", errors="replace")

        if not text.strip():
            print(f"[warn] Bronze vazio: {title}")
            continue

        sections, effective_strategy = build_sections(
            text=text,
            title=title,
            strategy=chunking.strategy,
            min_capitulos=semantic.min_capitulos,
            min_mediana_chars=semantic.min_mediana_chars,
        )

        source_document_id = row.get("source_document_id")
        if pd.isna(source_document_id):
            source_document_id = ""
        elif (
            isinstance(source_document_id, float)
            and source_document_id.is_integer()
        ):
            source_document_id = str(int(source_document_id))
        else:
            source_document_id = str(source_document_id)

        chunk_index = 0
        for section in sections:
            section_chunks = tokenizer.split(
                text=section["text"],
                chunk_size=chunking.chunk_size,
                overlap=chunking.overlap,
            )
            for chunk_index_in_section, chunk in enumerate(section_chunks):
                start_char = section["start_char"] + chunk["char_start"]
                end_char = section["start_char"] + chunk["char_end"]
                identity = (
                    f"{dataset_id}|{source_document_id}|"
                    f"{section['section_index']}|{chunk_index}|{chunk['text']}"
                )
                chunk_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:24]

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "silver_dataset_id": dataset_id,
                        "source_document_id": source_document_id,
                        "titulo": title,
                        "autor": row.get("autor"),
                        "categoria": category,
                        "source_bronze_bucket": config.storage.bronze_bucket,
                        "source_bronze_object_key": bronze_object_key,
                        "requested_strategy": chunking.strategy,
                        "effective_strategy": effective_strategy,
                        "section_index": section["section_index"],
                        "section_type": section["section_type"],
                        "section_number": section["section_number"],
                        "section_header": section["section_header"],
                        "section_title": section["section_title"],
                        "section_title_rule": section["section_title_rule"],
                        "chapter_rule": section["chapter_rule"],
                        "chunk_index": chunk_index,
                        "chunk_index_in_section": chunk_index_in_section,
                        "chunk_tokenizer": chunking.tokenizer,
                        "chunk_tokenizer_model": tokenizer_config.model,
                        "chunk_tokenizer_revision": tokenizer.resolved_revision,
                        "chunk_size": chunking.chunk_size,
                        "overlap": chunking.overlap,
                        "token_count": chunk["token_count"],
                        "start_char": start_char,
                        "end_char": end_char,
                        "text": chunk["text"],
                    }
                )
                chunk_index += 1

        print(
            f"  [{row_index + 1}/{len(bronze_manifest)}] "
            f"{title}: {chunk_index} chunks | {effective_strategy}"
        )

    if not chunks:
        raise RuntimeError(f"Nenhum chunk gerado para {dataset_id}.")

    chunks_df = pd.DataFrame(chunks)
    salvar_parquet(storage, bucket, chunks_key, chunks_df)

    manifest_df = pd.DataFrame(
        [
            {
                "silver_dataset_id": dataset_id,
                "pipeline_version": config.silver.pipeline_version,
                "config_hash": config_hash,
                "created_at_utc": utc_now(),
                "source_bronze_bucket": config.storage.bronze_bucket,
                "source_bronze_artifact_id": bronze_source[
                    "bronze_artifact_id"
                ],
                "source_bronze_manifest_key": bronze_source[
                    "manifest_key"
                ],
                "source_bronze_manifest_sha256": bronze_source[
                    "manifest_sha256"
                ],
                "strategy": chunking.strategy,
                "chunk_size": chunking.chunk_size,
                "overlap": chunking.overlap,
                "chunk_tokenizer": chunking.tokenizer,
                "chunk_tokenizer_model": tokenizer_config.model,
                "chunk_tokenizer_revision": tokenizer.resolved_revision,
                "min_capitulos": semantic.min_capitulos,
                "min_mediana_chars": semantic.min_mediana_chars,
                "documents": chunks_df["source_document_id"].nunique(),
                "chunks": len(chunks_df),
            }
        ]
    )
    salvar_parquet(storage, bucket, manifest_key, manifest_df)

    print(f"[created] Silver {dataset_id} | chunks={len(chunks_df)}")
    return {
        "status": "created",
        "silver_dataset_id": dataset_id,
        "manifest": manifest_df.iloc[0].to_dict(),
    }

def make_qdrant_manager(config: AppConfig) -> QdrantManager:
    qcfg = config.environment.qdrant
    factory = getattr(QdrantManager, "from_config", None)
    if callable(factory):
        return factory(qcfg)

    try:
        return QdrantManager(url=qcfg.url, api_key=qcfg.api_key)
    except TypeError:
        return QdrantManager(qcfg.url, qcfg.api_key)


class ReuseRegistry:
    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self.embedder


def index_identity(indexer: EmbeddingIndex, silver_dataset_id: str, embedder: Any) -> dict[str, Any]:
    info = embedder.info
    signature = indexer._build_index_signature(silver_dataset_id, info)
    index_hash = hashlib.sha256(
        json.dumps(
            signature,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    index_id = indexer._build_index_id(info.model, index_hash)
    return {
        "index_id": index_id,
        "index_hash": index_hash,
        "collection": index_id,
        "manifest_key": f"{index_id}/manifest.json",
        "signature": signature,
    }


def embed_queries(embedder: Any, texts: list[str]) -> np.ndarray:
    batch_method = getattr(embedder, "embed_queries", None)
    if callable(batch_method):
        vectors = batch_method(texts)
    else:
        single_method = getattr(embedder, "embed_query", None)
        if not callable(single_method):
            raise AttributeError(
                "Embedder não expõe embed_queries() nem embed_query(). "
                "Não farei fallback para embed_documents(), pois isso pode "
                "usar o prompt errado para queries."
            )
        rows = []
        for text in texts:
            value = np.asarray(single_method(text), dtype=np.float32)
            if value.ndim == 2 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 1:
                raise ValueError(f"Shape inesperado em embed_query(): {value.shape}")
            rows.append(value)
        vectors = np.stack(rows)

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(
            f"Embeddings de query com shape inválido: {vectors.shape}; "
            f"esperado ({len(texts)}, dim)."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings de query contêm NaN ou infinito.")
    return vectors


def ensure_query_embeddings(
    *,
    config: AppConfig,
    storage: RustFSManager,
    benchmark_bucket: str,
    benchmark_df: pd.DataFrame,
    benchmark_sha256: str,
    alias: str,
    embedder: Any,
    force: bool,
) -> dict[str, Any]:
    prefix = f"{BENCHMARK_PREFIX}/prepared/query_embeddings/{alias}"
    meta_key = f"{prefix}/queries.parquet"
    vectors_key = f"{prefix}/vectors.npz"
    manifest_key = f"{prefix}/manifest.json"

    info = embedder.info
    expected_identity = {
        "benchmark_sha256": benchmark_sha256,
        "model": info.model,
        "resolved_revision": info.resolved_revision,
        "dimension": info.dimension,
        "query_prompt_name": info.query_prompt_name,
    }

    if all(
        storage.object_exists(benchmark_bucket, key)
        for key in (meta_key, vectors_key, manifest_key)
    ):
        old = load_json(storage, benchmark_bucket, manifest_key)
        if all(old.get(key) == value for key, value in expected_identity.items()):
            print(f"[reuse] query embeddings {alias}")
            return old
        if not force:
            raise RuntimeError(
                f"Query embeddings {alias} já existem com identidade diferente. "
                "Use --force-prepared apenas se a substituição for intencional."
            )

    metadata_rows: list[dict[str, str]] = []
    texts: list[str] = []
    for row in benchmark_df.itertuples(index=False):
        for variant, column in (
            ("manual", "query_manual"),
            ("rewritten", "query_rewritten"),
        ):
            metadata_rows.append(
                {
                    "query_id": str(row.query_id),
                    "query_variant": variant,
                    "query": str(getattr(row, column)),
                }
            )
            texts.append(str(getattr(row, column)))

    vectors = embed_queries(embedder, texts)
    metadata = pd.DataFrame(metadata_rows)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, vectors=vectors)

    manifest = {
        "created_at_utc": utc_now(),
        **expected_identity,
        "model_alias": alias,
        "rows": len(metadata),
        "dtype": str(vectors.dtype),
        "shape": list(vectors.shape),
        "metadata_key": meta_key,
        "vectors_key": vectors_key,
    }
    storage.upload_bytes(
        benchmark_bucket,
        meta_key,
        dataframe_parquet_bytes(metadata),
        content_type="application/vnd.apache.parquet",
    )
    storage.upload_bytes(
        benchmark_bucket,
        vectors_key,
        buffer.getvalue(),
        content_type="application/octet-stream",
    )
    storage.upload_bytes(
        benchmark_bucket,
        manifest_key,
        json_bytes(manifest),
        content_type="application/json",
    )
    print(f"[created] query embeddings {alias} | shape={vectors.shape}")
    return manifest


def release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_matrix_manifest(
    *,
    benchmark_sha256: str,
    silver_results: list[dict[str, Any]],
    index_results: list[dict[str, Any]],
) -> dict[str, Any]:
    retrieval_configs = []
    for silver in silver_results:
        sid = silver["silver_dataset_id"]
        manifest = silver.get("manifest", {})
        for retrieval in RETRIEVALS:
            retrieval_configs.append(
                {
                    "silver_dataset_id": sid,
                    "chunk_strategy": manifest.get("strategy"),
                    "chunk_size": manifest.get("chunk_size"),
                    "overlap": manifest.get("overlap"),
                    **retrieval,
                }
            )

    return {
        "created_at_utc": utc_now(),
        "benchmark_sha256": benchmark_sha256,
        "split": {
            "seed": SPLIT_SEED,
            "dev_ratio": DEV_RATIO,
            "holdout_ratio": HOLDOUT_RATIO,
        },
        "chunk_variants": list(CHUNK_VARIANTS),
        "embedders": list(EMBEDDERS),
        "retrievals": list(RETRIEVALS),
        "llms": list(OLLAMA_MODELS),
        "expected": {
            "silvers": 6,
            "dense_indexes": 12,
            "retrieval_configs": 30,
            "query_embeddings": 400,
        },
        "resolved": {
            "silvers": silver_results,
            "indexes": index_results,
            "retrieval_configs": retrieval_configs,
        },
    }


def main() -> None:
    from src.embeddings.registry import EmbedderRegistry
    from src.indexing.embedding_index import EmbeddingIndex

    args = parse_args()
    base_config = ConfigManager.load()
    storage = RustFSManager.from_config(base_config.environment.rustfs)
    benchmark_bucket = base_config.storage.benchmarks_bucket
    storage.create_bucket(benchmark_bucket)

    if not storage.object_exists(benchmark_bucket, args.benchmark_key):
        raise FileNotFoundError(
            f"Benchmark não encontrado: s3://{benchmark_bucket}/{args.benchmark_key}"
        )

    benchmark_bytes = storage.download_bytes(benchmark_bucket, args.benchmark_key)
    benchmark_sha = sha256_bytes(benchmark_bytes)
    benchmark_df = pd.read_csv(io.BytesIO(benchmark_bytes))
    validate_benchmark(benchmark_df)

    canonical_key = f"{BENCHMARK_PREFIX}/datasets/benchmark_questions_final.csv"
    storage.upload_bytes(
        benchmark_bucket,
        canonical_key,
        benchmark_bytes,
        content_type="text/csv; charset=utf-8",
    )
    print(
        f"[benchmark] rows={len(benchmark_df)} | sha256={benchmark_sha[:12]}... | "
        f"s3://{benchmark_bucket}/{canonical_key}"
    )

    split_df, split_manifest = ensure_split(
        storage,
        benchmark_bucket,
        benchmark_df,
        benchmark_sha,
        force=args.force_prepared,
    )

    silver_results: list[dict[str, Any]] = []
    if not args.skip_silver:
        for variant in CHUNK_VARIANTS:
            variant_config = config_for_chunk_variant(base_config, variant)
            result = ensure_silver(variant_config, storage)
            silver_results.append(result)
    else:
        print("[skip] Silvers")

    index_results: list[dict[str, Any]] = []
    query_embedding_results: list[dict[str, Any]] = []

    if (not args.skip_indexes or not args.skip_query_embeddings) and not silver_results:
        # Resolve as Silvers sem criá-las. Se alguma não existir, falha com mensagem clara.
        for variant in CHUNK_VARIANTS:
            variant_config = config_for_chunk_variant(base_config, variant)
            sid, _, _, _, _ = silver_identity(variant_config, storage)
            manifest_key = f"datasets/{sid}/manifest.parquet"
            chunks_key = f"datasets/{sid}/chunks.parquet"
            if not (
                storage.object_exists(base_config.storage.silver_bucket, manifest_key)
                and storage.object_exists(base_config.storage.silver_bucket, chunks_key)
            ):
                raise RuntimeError(
                    f"Silver {sid} ausente. Rode sem --skip-silver primeiro."
                )
            manifest = pd.read_parquet(
                io.BytesIO(
                    storage.download_bytes(base_config.storage.silver_bucket, manifest_key)
                )
            )
            silver_results.append(
                {
                    "status": "resolved",
                    "silver_dataset_id": sid,
                    "manifest": manifest.iloc[0].to_dict(),
                }
            )

    if not args.skip_indexes or not args.skip_query_embeddings:
        qdrant = make_qdrant_manager(base_config) if not args.skip_indexes else None
        registry = EmbedderRegistry()

        for alias in EMBEDDERS:
            print(f"\n=== Embedder: {alias} ===")
            model_config = base_config.embedding.models[alias]
            embedder = registry.create(
                model_config,
                normalize_embeddings=base_config.embedding.index.normalize_embeddings,
                device=base_config.embedding.execution.device,
                batch_size=base_config.embedding.execution.batch_size,
            )
            print(
                f"[model] {embedder.info.model} | "
                f"revision={embedder.info.resolved_revision} | "
                f"dim={embedder.info.dimension}"
            )

            if not args.skip_query_embeddings:
                query_embedding_results.append(
                    ensure_query_embeddings(
                        config=base_config,
                        storage=storage,
                        benchmark_bucket=benchmark_bucket,
                        benchmark_df=benchmark_df,
                        benchmark_sha256=benchmark_sha,
                        alias=alias,
                        embedder=embedder,
                        force=args.force_prepared,
                    )
                )

            if not args.skip_indexes:
                assert qdrant is not None
                reuse_registry = ReuseRegistry(embedder)
                indexer = EmbeddingIndex(
                    base_config,
                    storage,
                    qdrant,
                    reuse_registry,
                )
                for silver in silver_results:
                    sid = silver["silver_dataset_id"]
                    identity = index_identity(indexer, sid, embedder)
                    index_bucket = base_config.storage.embedding_indexes_bucket
                    collection_exists = qdrant.collection_exists(identity["collection"])
                    manifest_exists = storage.object_exists(
                        index_bucket,
                        identity["manifest_key"],
                    )

                    if collection_exists and manifest_exists:
                        manifest = load_json(
                            storage,
                            index_bucket,
                            identity["manifest_key"],
                        )
                        print(f"[reuse] index {identity['index_id']} | silver={sid}")
                        index_results.append(
                            {
                                "status": "reused",
                                "model_alias": alias,
                                **identity,
                                "manifest": manifest,
                            }
                        )
                    elif collection_exists != manifest_exists:
                        raise RuntimeError(
                            "Índice inconsistente entre Qdrant e RustFS: "
                            f"{identity['index_id']}"
                        )
                    else:
                        print(
                            f"[create] index ausente {identity['index_id']} | silver={sid}"
                        )
                        manifest = indexer.build(sid, model_alias=alias)
                        index_results.append(
                            {
                                "status": "created",
                                "model_alias": alias,
                                **identity,
                                "manifest": manifest,
                            }
                        )

            del embedder
            release_gpu()
            print(f"[unload] {alias}")

    matrix_manifest = build_matrix_manifest(
        benchmark_sha256=benchmark_sha,
        silver_results=silver_results,
        index_results=index_results,
    )
    matrix_manifest["prepared"] = {
        "split": split_manifest,
        "query_embeddings": query_embedding_results,
    }

    matrix_key = f"{BENCHMARK_PREFIX}/prepared/matrix.json"
    prep_manifest_key = f"{BENCHMARK_PREFIX}/prepared/preparation_manifest.json"
    storage.upload_bytes(
        benchmark_bucket,
        matrix_key,
        json_bytes(matrix_manifest),
        content_type="application/json",
    )
    storage.upload_bytes(
        benchmark_bucket,
        prep_manifest_key,
        json_bytes(
            {
                "created_at_utc": utc_now(),
                "status": "success",
                "benchmark_sha256": benchmark_sha,
                "benchmark_source_key": args.benchmark_key,
                "canonical_benchmark_key": canonical_key,
                "matrix_key": matrix_key,
                "split_counts": split_df["split"].value_counts().to_dict(),
                "silvers": len(silver_results),
                "indexes": len(index_results),
                "query_embedding_models": len(query_embedding_results),
            }
        ),
        content_type="application/json",
    )

    print("\n=== PREPARATION OK ===")
    print(f"benchmark sha256: {benchmark_sha}")
    print(f"split: {split_df['split'].value_counts().to_dict()}")
    print(f"silvers resolved: {len(silver_results)}")
    print(f"dense indexes resolved: {len(index_results)}")
    print(f"query embedding sets: {len(query_embedding_results)}")
    print(f"manifest: s3://{benchmark_bucket}/{prep_manifest_key}")


if __name__ == "__main__":
    main()
