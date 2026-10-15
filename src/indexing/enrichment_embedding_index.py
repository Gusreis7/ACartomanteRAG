from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config.config_manager import AppConfig, EmbeddingModelConfig
from src.embeddings.base import EmbedderInfo
from src.embeddings.registry import EmbedderRegistry
from src.storage.qdrant_manager import QdrantManager
from src.storage.rustfs_manager import RustFSManager


ENRICHMENT_CURRENT_KEY = os.getenv("ENRICHMENT_DATASET_POINTER", "current.json")
ENRICHMENT_INDEX_CURRENT_KEY = "enrichment/current.json"
ENRICHMENT_DOCUMENT_SCHEMA_VERSION = "summary_units_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_qdrant_manager(config: AppConfig) -> QdrantManager:
    qcfg = config.environment.qdrant
    factory = getattr(QdrantManager, "from_config", None)
    if callable(factory):
        return factory(qcfg)
    try:
        return QdrantManager(url=qcfg.url, api_key=qcfg.api_key)
    except TypeError:
        return QdrantManager(qcfg.url, qcfg.api_key)


def json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-").lower()
    return value or "model"


class EnrichmentEmbeddingIndex:
    """Cria uma collection Qdrant para um dataset de enrichment e um embedder.

    Unidade vetorial:
    - 1 vetor para `summary.resumo_geral` de cada obra;
    - 1 vetor para `blocks[].resumo` de cada capítulo/bloco;
    - nunca rechunka o resumo.
    """

    def __init__(
        self,
        config: AppConfig,
        rustfs: RustFSManager,
        qdrant: QdrantManager,
        registry: EmbedderRegistry,
    ) -> None:
        self.config = config
        self.rustfs = rustfs
        self.qdrant = qdrant
        self.registry = registry

    def resolve_current_dataset(self) -> tuple[dict[str, Any], pd.DataFrame, str]:
        bucket = self.config.storage.enrichment_bucket
        if not self.rustfs.bucket_exists(bucket):
            raise RuntimeError(f"Bucket enrichment não encontrado: {bucket}")
        if not self.rustfs.object_exists(bucket, ENRICHMENT_CURRENT_KEY):
            raise FileNotFoundError(
                f"Enrichment current ausente: s3://{bucket}/{ENRICHMENT_CURRENT_KEY}"
            )

        current = json.loads(
            self.rustfs.download_bytes(bucket, ENRICHMENT_CURRENT_KEY).decode("utf-8")
        )
        dataset_id = str(current.get("enrichment_dataset_id") or "").strip()
        manifest_key = str(current.get("manifest_key") or "").strip()
        if not dataset_id or not manifest_key:
            raise RuntimeError("current.json do enrichment sem dataset_id/manifest_key.")

        manifest_bytes = self.rustfs.download_bytes(bucket, manifest_key)
        manifest_sha = sha256_bytes(manifest_bytes)
        expected_sha = str(current.get("manifest_sha256") or "").strip()
        if expected_sha and expected_sha != manifest_sha:
            raise RuntimeError(
                "SHA do manifest enrichment divergente. "
                f"esperado={expected_sha}; obtido={manifest_sha}."
            )
        frame = pd.read_parquet(io.BytesIO(manifest_bytes))
        if frame.empty:
            raise RuntimeError("Manifest enrichment vazio.")
        if "status_enrichment" in frame.columns:
            failed = frame[~frame["status_enrichment"].astype(str).eq("success")]
            if len(failed):
                raise RuntimeError(
                    "Enrichment current contém obras não-success; "
                    f"quantidade={len(failed)}."
                )
        return current, frame.reset_index(drop=True), manifest_sha

    def load_documents(
        self,
        current: dict[str, Any],
        manifest: pd.DataFrame,
    ) -> pd.DataFrame:
        bucket = self.config.storage.enrichment_bucket
        dataset_id = str(current["enrichment_dataset_id"])
        rows: list[dict[str, Any]] = []

        for item in manifest.itertuples(index=False):
            object_key = str(getattr(item, "enrichment_object_key"))
            payload = json.loads(
                self.rustfs.download_bytes(bucket, object_key).decode("utf-8")
            )
            source = payload.get("source") or {}
            plan = payload.get("plan") or {}
            title = str(source.get("titulo") or getattr(item, "titulo", ""))
            author = str(source.get("autor") or getattr(item, "autor", "") or "")
            category = str(source.get("categoria") or getattr(item, "categoria", "") or "")
            source_document_id = str(
                source.get("source_document_id")
                or getattr(item, "source_document_id", "")
            )

            overall = payload.get("summary") or {}
            overall_text = str(overall.get("resumo_geral") or "").strip()
            if overall_text:
                identity = f"{dataset_id}|{source_document_id}|work_summary"
                rows.append(
                    {
                        "enrichment_document_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                        "enrichment_dataset_id": dataset_id,
                        "source_document_id": source_document_id,
                        "titulo": title,
                        "autor": author,
                        "categoria": category,
                        "document_type": "work_summary",
                        "unit_id": None,
                        "section_index": None,
                        "section_label": None,
                        "section_type": None,
                        "part_index": None,
                        "part_count": None,
                        "start_char": None,
                        "end_char": None,
                        "plan_mode": plan.get("plan_mode"),
                        "effective_strategy": plan.get("effective_strategy_from_chunker"),
                        "text": overall_text,
                        "events": list(overall.get("eventos_principais") or []),
                        "source_enrichment_bucket": bucket,
                        "source_enrichment_object_key": object_key,
                    }
                )

            for block in payload.get("blocks") or []:
                text = str(block.get("resumo") or "").strip()
                if not text:
                    continue
                unit_id = str(block.get("unit_id") or "")
                identity = (
                    f"{dataset_id}|{source_document_id}|section_summary|{unit_id}"
                )
                rows.append(
                    {
                        "enrichment_document_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                        "enrichment_dataset_id": dataset_id,
                        "source_document_id": source_document_id,
                        "titulo": title,
                        "autor": author,
                        "categoria": category,
                        "document_type": "section_summary",
                        "unit_id": unit_id,
                        "section_index": block.get("section_index"),
                        "section_label": block.get("section_label"),
                        "section_type": block.get("section_type"),
                        "part_index": block.get("part_index"),
                        "part_count": block.get("part_count"),
                        "start_char": block.get("start_char"),
                        "end_char": block.get("end_char"),
                        "plan_mode": plan.get("plan_mode"),
                        "effective_strategy": plan.get("effective_strategy_from_chunker"),
                        "text": text,
                        "events": list(block.get("eventos") or []),
                        "source_enrichment_bucket": bucket,
                        "source_enrichment_object_key": object_key,
                    }
                )

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("Nenhum resumo indexável encontrado no enrichment.")
        if frame["enrichment_document_id"].duplicated().any():
            raise RuntimeError("enrichment_document_id duplicado.")
        return frame.reset_index(drop=True)

    def build(self, *, model_alias: str, resume: bool = False) -> dict[str, Any]:
        if model_alias not in self.config.embedding.models:
            raise ValueError(
                f"Embedder '{model_alias}' não existe em embedding.models."
            )

        current, source_manifest, source_manifest_sha = self.resolve_current_dataset()
        documents = self.load_documents(current, source_manifest)
        model_config = self.config.embedding.models[model_alias]
        embedder = self.registry.create(
            model_config,
            normalize_embeddings=self.config.embedding.index.normalize_embeddings,
            device=self.config.embedding.execution.device,
            batch_size=self.config.embedding.execution.batch_size,
        )
        info = embedder.info
        self._validate_embedder_info(info, model_config)

        signature = {
            "pipeline_version": self.config.embedding.pipeline_version,
            "document_schema_version": ENRICHMENT_DOCUMENT_SCHEMA_VERSION,
            "source_enrichment_dataset_id": str(current["enrichment_dataset_id"]),
            "source_enrichment_manifest_sha256": source_manifest_sha,
            "embedding": {
                "model_alias": model_alias,
                "backend": info.backend,
                "model": info.model,
                "resolved_revision": info.resolved_revision,
                "dimension": info.dimension,
                "inference_dtype": info.inference_dtype,
                "pooling": info.pooling,
                "truncate_dim": info.truncate_dim,
                "normalize_embeddings": info.normalize_embeddings,
                "document_prompt_name": info.document_prompt_name,
                "query_prompt_name": info.query_prompt_name,
            },
            "distance": self.config.embedding.index.distance,
        }
        index_hash = hashlib.sha256(
            json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        index_id = f"enr_idx_{slug(model_alias)}_{index_hash[:12]}"
        collection = index_id
        indexes_bucket = self.config.storage.embedding_indexes_bucket
        manifest_key = f"enrichment/{index_id}/manifest.json"

        if not self.rustfs.bucket_exists(indexes_bucket):
            self.rustfs.create_bucket(indexes_bucket)

        collection_exists = self.qdrant.collection_exists(collection)
        manifest_exists = self.rustfs.object_exists(indexes_bucket, manifest_key)
        building_key = f"enrichment/{index_id}/building.json"
        if manifest_exists:
            old = json.loads(
                self.rustfs.download_bytes(indexes_bucket, manifest_key).decode("utf-8")
            )
            if old.get("status") != "success" or old.get("index_hash") != index_hash:
                raise RuntimeError(f"Índice enrichment existente inválido: {index_id}")
            if collection_exists and self.qdrant.count(collection_name=collection, exact=True) == len(documents):
                print(f"[reuse] {model_alias} -> {collection}")
                return old
            if not resume:
                raise RuntimeError("Collection enrichment incompleta; use retomada explícita.")
        elif collection_exists:
            if not resume or not self.rustfs.object_exists(indexes_bucket, building_key):
                raise RuntimeError("Collection sem manifest nem checkpoint de construção.")
            marker = json.loads(self.rustfs.download_bytes(indexes_bucket, building_key))
            if marker.get("index_hash") != index_hash:
                raise ValueError("Collection parcial de enrichment diverge da assinatura.")

        first = embedder.embed_documents([str(documents.iloc[0]["text"])])
        self._validate_vectors(first, 1, info.dimension)
        self.rustfs.upload_bytes(indexes_bucket, building_key, json.dumps({"index_hash": index_hash}).encode(), "application/json")
        if not collection_exists:
            self.qdrant.create_collection(
                collection_name=collection,
                vector_size=info.dimension,
                distance=self.config.embedding.index.distance,
            )

        batch_size = int(self.config.embedding.execution.batch_size)
        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents.iloc[start : start + batch_size]
            vectors = embedder.embed_documents(batch["text"].astype(str).tolist())
            self._validate_vectors(vectors, len(batch), info.dimension)
            ids = [
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"acartomante:enrichment:{index_id}:{doc_id}",
                    )
                )
                for doc_id in batch["enrichment_document_id"].astype(str).tolist()
            ]
            payloads = [self._payload(row) for _, row in batch.iterrows()]
            self.qdrant.upsert_points(
                collection_name=collection,
                ids=ids,
                vectors=vectors.tolist(),
                payloads=payloads,
                wait=True,
            )
            total += len(batch)
            print(f"[embed] {model_alias}: {total}/{len(documents)}")

        count = self.qdrant.count(collection_name=collection, exact=True)
        if count != len(documents):
            raise RuntimeError(
                f"Qdrant count divergente: docs={len(documents)} qdrant={count}."
            )

        final_info = embedder.info
        manifest = {
            "manifest_version": 1,
            "status": "success",
            "index_id": index_id,
            "index_hash": index_hash,
            "created_at_utc": utc_now(),
            "source": {
                "enrichment_bucket": self.config.storage.enrichment_bucket,
                "enrichment_dataset_id": str(current["enrichment_dataset_id"]),
                "enrichment_manifest_key": str(current["manifest_key"]),
                "enrichment_manifest_sha256": source_manifest_sha,
            },
            "documents": {
                "schema_version": ENRICHMENT_DOCUMENT_SCHEMA_VERSION,
                "total": len(documents),
                "work_summaries": int(documents["document_type"].eq("work_summary").sum()),
                "section_summaries": int(documents["document_type"].eq("section_summary").sum()),
                "embedding_field": "text",
                "rechunked": False,
            },
            "embedding": {
                "pipeline_version": self.config.embedding.pipeline_version,
                "model_alias": model_alias,
                "backend": final_info.backend,
                "model": final_info.model,
                "configured_revision": final_info.configured_revision,
                "resolved_revision": final_info.resolved_revision,
                "dimension": final_info.dimension,
                "output_dtype": final_info.output_dtype,
                "inference_dtype": final_info.inference_dtype,
                "pooling": final_info.pooling,
                "truncate_dim": final_info.truncate_dim,
                "normalize_embeddings": final_info.normalize_embeddings,
                "distance": self.config.embedding.index.distance,
                "document_prompt_name": final_info.document_prompt_name,
                "query_prompt_name": final_info.query_prompt_name,
            },
            "execution": {
                "batch_size_requested": final_info.batch_size_requested,
                "batch_size_effective": final_info.batch_size_effective,
                "requested_device": final_info.requested_device,
                "resolved_device": final_info.resolved_device,
            },
            "qdrant": {"collection": collection},
            "validation": {
                "generated_vectors": total,
                "qdrant_vectors": count,
            },
            "index_signature": signature,
        }
        self.rustfs.upload_bytes(
            indexes_bucket,
            manifest_key,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
        )
        print(f"[created] {model_alias} -> {collection} | vectors={count}")
        return manifest

    @staticmethod
    def _payload(row: pd.Series) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key == "events":
                out[key] = [str(item) for item in (value or [])]
            else:
                out[key] = json_scalar(value)
        return out

    @staticmethod
    def _validate_vectors(vectors: np.ndarray, rows: int, dim: int) -> None:
        if not isinstance(vectors, np.ndarray):
            raise TypeError("Embedder deve retornar np.ndarray.")
        if vectors.shape != (rows, dim):
            raise ValueError(
                f"Shape inválido: esperado={(rows, dim)} recebido={vectors.shape}."
            )
        if vectors.dtype != np.float32:
            raise TypeError(f"dtype inválido: {vectors.dtype}; esperado=float32.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contêm NaN/infinito.")

    @staticmethod
    def _validate_embedder_info(info: EmbedderInfo, cfg: EmbeddingModelConfig) -> None:
        if info.dimension <= 0:
            raise ValueError("Dimensão inválida do embedder.")
        if info.output_dtype != "float32":
            raise ValueError("Contrato do embedder exige output_dtype=float32.")
        if info.configured_revision != cfg.revision:
            raise ValueError("Revision do embedder diverge da configuração.")
        if info.inference_dtype != cfg.inference_dtype:
            raise ValueError("inference_dtype do embedder diverge da configuração.")
        if info.pooling != cfg.pooling:
            raise ValueError("pooling do embedder diverge da configuração.")


def release_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
