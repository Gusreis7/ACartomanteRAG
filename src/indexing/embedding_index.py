from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config.config_manager import AppConfig, EmbeddingModelConfig
from src.embeddings.base import EmbedderInfo
from src.embeddings.registry import EmbedderRegistry
from src.indexing.release_manager import ReleaseManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.rustfs_manager import RustFSManager


class EmbeddingIndex:
    REQUIRED_CHUNK_COLUMNS = {
        "chunk_id",
        "silver_dataset_id",
        "text",
    }

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

    def build(
        self,
        silver_dataset_id: str,
        model_alias: str | None = None,
        *,
        resume: bool = False,
        publish_default: bool = True,
    ) -> dict[str, Any]:
        alias = (
            model_alias
            or self.config.embedding.selected_model
        )

        if alias not in self.config.embedding.models:
            available = ", ".join(
                sorted(
                    self.config.embedding.models
                )
            )

            raise ValueError(
                f"Embedding model '{alias}' "
                "não configurado. "
                f"Disponíveis: {available}."
            )

        model_config = (
            self.config.embedding.models[
                alias
            ]
        )

        embedder = self.registry.create(
            model_config,
            normalize_embeddings=(
                self.config.embedding
                .index
                .normalize_embeddings
            ),
            device=(
                self.config.embedding
                .execution
                .device
            ),
            batch_size=(
                self.config.embedding
                .execution
                .batch_size
            ),
        )

        info = embedder.info

        self._validate_embedder_info(
            info,
            model_config,
        )

        signature = (
            self._build_index_signature(
                silver_dataset_id=(
                    silver_dataset_id
                ),
                info=info,
            )
        )

        index_hash = hashlib.sha256(
            json.dumps(
                signature,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        index_id = self._build_index_id(
            model=info.model,
            index_hash=index_hash,
        )

        collection_name = index_id
        manifest_key = (
            f"{index_id}/manifest.json"
        )

        indexes_bucket = (
            self.config.storage
            .embedding_indexes_bucket
        )

        if not self.rustfs.bucket_exists(
            indexes_bucket
        ):
            self.rustfs.create_bucket(
                indexes_bucket
            )

        collection_exists = (
            self.qdrant.collection_exists(
                collection_name
            )
        )

        manifest_exists = (
            self.rustfs.object_exists(
                indexes_bucket,
                manifest_key,
            )
        )

        building_key = f"{index_id}/building.json"
        if resume and manifest_exists:
            previous = json.loads(self.rustfs.download_bytes(indexes_bucket, manifest_key))
            if previous.get("status") != "success" or previous.get("index_hash") != index_hash:
                raise ValueError("Manifest existente diverge da assinatura do índice.")
            if collection_exists:
                count = self.qdrant.count(collection_name=collection_name, exact=True)
                if count == previous["validation"]["silver_chunks"]:
                    return previous
                # Reenvia IDs determinísticos de um índice pertencente à assinatura.
        elif resume and collection_exists and self.rustfs.object_exists(indexes_bucket, building_key):
            marker = json.loads(self.rustfs.download_bytes(indexes_bucket, building_key))
            if marker.get("index_hash") != index_hash:
                raise ValueError("Collection parcial não pertence a esta execução.")
        elif collection_exists or manifest_exists:
            raise FileExistsError(
                "Embedding index já existe "
                "ou está em estado inconsistente. "
                f"index_id={index_id}, "
                "collection_exists="
                f"{collection_exists}, "
                "manifest_exists="
                f"{manifest_exists}."
            )

        chunks = self._load_silver_chunks(
            silver_dataset_id
        )

        if chunks.empty:
            raise ValueError(
                "O dataset Silver "
                "não possui chunks."
            )

        first_vector = (
            embedder.embed_documents(
                [
                    str(
                        chunks.iloc[0][
                            "text"
                        ]
                    )
                ]
            )
        )

        self._validate_vectors(
            first_vector,
            expected_rows=1,
            expected_dimension=(
                info.dimension
            ),
        )

        self.rustfs.upload_bytes(indexes_bucket, building_key, json.dumps({"index_hash": index_hash}).encode(), "application/json")
        if not collection_exists:
            self.qdrant.create_collection(
                collection_name=(
                    collection_name
                ),
                vector_size=info.dimension,
                distance=(
                    self.config.embedding
                    .index
                    .distance
                ),
            )

        configured_batch_size = (
            self.config.embedding
            .execution
            .batch_size
        )

        total_vectors = 0

        with tqdm(
            total=len(chunks),
            desc="Indexando chunks",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            for start in range(
                0,
                len(chunks),
                configured_batch_size,
            ):
                batch = chunks.iloc[
                    start :
                    start
                    + configured_batch_size
                ]

                texts = (
                    batch["text"]
                    .astype(str)
                    .tolist()
                )

                vectors = (
                    embedder.embed_documents(
                        texts
                    )
                )

                self._validate_vectors(
                    vectors,
                    expected_rows=len(batch),
                    expected_dimension=(
                        info.dimension
                    ),
                )

                ids = [
                    str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            (
                                "acartomante:"
                                f"{index_id}:"
                                f"{chunk_id}"
                            ),
                        )
                    )
                    for chunk_id
                    in batch[
                        "chunk_id"
                    ].tolist()
                ]

                payloads = [
                    self._build_payload(row)
                    for _, row
                    in batch.iterrows()
                ]

                self.qdrant.upsert_points(
                    collection_name=(
                        collection_name
                    ),
                    ids=ids,
                    vectors=vectors.tolist(),
                    payloads=payloads,
                    wait=True,
                )

                total_vectors += len(batch)

                progress.update(len(batch))
                progress.set_postfix(
                    batch=(
                        embedder.info
                        .batch_size_effective
                    ),
                    vectors=total_vectors,
                )

        qdrant_count = (
            self.qdrant.count(
                collection_name=(
                    collection_name
                ),
                exact=True,
            )
        )

        if qdrant_count != len(chunks):
            raise RuntimeError(
                "Falha na validação do "
                "índice: "
                f"Silver chunks={len(chunks)}, "
                "Qdrant vectors="
                f"{qdrant_count}."
            )

        # Releitura importante: batch_size_effective
        # pode ter mudado por fallback de OOM.
        final_info = embedder.info

        manifest = {
            "manifest_version": 1,
            "status": "success",
            "index_id": index_id,
            "index_hash": index_hash,
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "source": {
                "silver_bucket": (
                    self.config.storage
                    .silver_bucket
                ),
                "silver_dataset_id": (
                    silver_dataset_id
                ),
                "chunks_key": (
                    "datasets/"
                    f"{silver_dataset_id}/"
                    "chunks.parquet"
                ),
                "manifest_key": (
                    "datasets/"
                    f"{silver_dataset_id}/"
                    "manifest.parquet"
                ),
            },
            "embedding": {
                "pipeline_version": (
                    self.config.embedding
                    .pipeline_version
                ),
                "model_alias": alias,
                "backend": (
                    final_info.backend
                ),
                "model": (
                    final_info.model
                ),
                "configured_revision": (
                    final_info
                    .configured_revision
                ),
                "resolved_revision": (
                    final_info
                    .resolved_revision
                ),
                "dimension": (
                    final_info.dimension
                ),
                "output_dtype": (
                    final_info.output_dtype
                ),
                "inference_dtype": (
                    final_info
                    .inference_dtype
                ),
                "pooling": (
                    final_info.pooling
                ),
                "truncate_dim": (
                    final_info.truncate_dim
                ),
                "normalize_embeddings": (
                    final_info
                    .normalize_embeddings
                ),
                "distance": (
                    self.config.embedding
                    .index
                    .distance
                ),
                "configured_document_prompt_name": (
                    final_info
                    .configured_document_prompt_name
                ),
                "configured_query_prompt_name": (
                    final_info
                    .configured_query_prompt_name
                ),
                "document_prompt_name": (
                    final_info
                    .document_prompt_name
                ),
                "query_prompt_name": (
                    final_info
                    .query_prompt_name
                ),
            },
            "execution": {
                "batch_size_requested": (
                    final_info
                    .batch_size_requested
                ),
                "batch_size_effective": (
                    final_info
                    .batch_size_effective
                ),
                "requested_device": (
                    final_info
                    .requested_device
                ),
                "resolved_device": (
                    final_info
                    .resolved_device
                ),
            },
            "qdrant": {
                "collection": (
                    collection_name
                ),
            },
            "validation": {
                "silver_chunks": len(
                    chunks
                ),
                "generated_vectors": (
                    total_vectors
                ),
                "qdrant_vectors": (
                    qdrant_count
                ),
            },
            "index_signature": (
                signature
            ),
        }

        self.rustfs.upload_bytes(
            indexes_bucket,
            manifest_key,
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            content_type=(
                "application/json"
            ),
        )

        release_manager = ReleaseManager(
            config=self.config,
            rustfs=self.rustfs,
        )

        bootstrapped = publish_default and (
            release_manager
            .bootstrap_if_missing(
                index_id=index_id,
                release="default",
            )
        )

        if bootstrapped:
            print(
                "[+] Bootstrap release: "
                f"default -> {index_id}"
            )

        return manifest

    def _load_silver_chunks(
        self,
        silver_dataset_id: str,
    ) -> pd.DataFrame:
        chunks_key = (
            "datasets/"
            f"{silver_dataset_id}/"
            "chunks.parquet"
        )

        if not self.rustfs.object_exists(
            self.config.storage.silver_bucket,
            chunks_key,
        ):
            raise FileNotFoundError(
                "Dataset Silver não "
                "encontrado: "
                "s3://"
                f"{self.config.storage.silver_bucket}/"
                f"{chunks_key}"
            )

        payload = (
            self.rustfs.download_bytes(
                self.config.storage
                .silver_bucket,
                chunks_key,
            )
        )

        chunks = pd.read_parquet(
            BytesIO(payload)
        )

        missing_columns = (
            self.REQUIRED_CHUNK_COLUMNS
            - set(chunks.columns)
        )

        if missing_columns:
            raise ValueError(
                "Silver chunks sem "
                "colunas obrigatórias: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        dataset_ids = set(
            chunks[
                "silver_dataset_id"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        if dataset_ids != {
            silver_dataset_id
        }:
            raise ValueError(
                "silver_dataset_id interno "
                "não corresponde ao dataset "
                "solicitado. "
                f"Esperado={silver_dataset_id}, "
                "encontrado="
                f"{sorted(dataset_ids)}."
            )

        return chunks

    def _build_index_signature(
        self,
        silver_dataset_id: str,
        info: EmbedderInfo,
    ) -> dict[str, Any]:
        return {
            "pipeline_version": (
                self.config.embedding
                .pipeline_version
            ),
            "silver_dataset_id": (
                silver_dataset_id
            ),
            "embedding": {
                "backend": info.backend,
                "model": info.model,
                "resolved_revision": (
                    info.resolved_revision
                ),
                "dimension": (
                    info.dimension
                ),
                "inference_dtype": (
                    info.inference_dtype
                ),
                "pooling": info.pooling,
                "truncate_dim": (
                    info.truncate_dim
                ),
                "normalize_embeddings": (
                    info.normalize_embeddings
                ),
                "document_prompt_name": (
                    info.document_prompt_name
                ),
                "query_prompt_name": (
                    info.query_prompt_name
                ),
            },
            "distance": (
                self.config.embedding
                .index
                .distance
            ),
        }

    @staticmethod
    def _build_index_id(
        model: str,
        index_hash: str,
    ) -> str:
        model_name = (
            model.rsplit("/", 1)[-1]
        )

        slug = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "-",
            model_name,
        ).strip("-").lower()

        return (
            f"idx_{slug}_"
            f"{index_hash[:12]}"
        )

    @staticmethod
    def _validate_embedder_info(
        info: EmbedderInfo,
        model_config: EmbeddingModelConfig,
    ) -> None:
        if info.dimension <= 0:
            raise ValueError(
                "Embedder informou "
                "dimensão inválida."
            )

        if (
            info.output_dtype
            != "float32"
        ):
            raise ValueError(
                "Contrato do Embedder "
                "exige "
                "output_dtype=float32."
            )

        if (
            info.configured_revision
            != model_config.revision
        ):
            raise ValueError(
                "Revision reportada pelo "
                "Embedder não corresponde "
                "à configuração."
            )

        if (
            info.inference_dtype
            != model_config
            .inference_dtype
        ):
            raise ValueError(
                "inference_dtype "
                "reportado pelo Embedder "
                "não corresponde à "
                "configuração."
            )

        if (
            info.pooling
            != model_config.pooling
        ):
            raise ValueError(
                "pooling reportado pelo "
                "Embedder não corresponde "
                "à configuração."
            )

    @staticmethod
    def _validate_vectors(
        vectors: np.ndarray,
        expected_rows: int,
        expected_dimension: int,
    ) -> None:
        if not isinstance(
            vectors,
            np.ndarray,
        ):
            raise TypeError(
                "Embedder deve retornar "
                "np.ndarray."
            )

        expected_shape = (
            expected_rows,
            expected_dimension,
        )

        if (
            vectors.shape
            != expected_shape
        ):
            raise ValueError(
                "Shape inválido retornado "
                "pelo Embedder. "
                f"Esperado={expected_shape}, "
                f"recebido={vectors.shape}."
            )

        if (
            vectors.dtype
            != np.float32
        ):
            raise TypeError(
                "Embedder deve retornar "
                "np.ndarray "
                "dtype=float32."
            )

        if not np.isfinite(
            vectors
        ).all():
            raise ValueError(
                "Embedder retornou "
                "NaN ou infinito."
            )

    @staticmethod
    def _build_payload(
        row: pd.Series,
    ) -> dict[str, Any]:
        payload: dict[
            str,
            Any,
        ] = {}

        for key, value in row.items():
            if pd.isna(value):
                payload[key] = None

            elif isinstance(
                value,
                np.generic,
            ):
                payload[key] = (
                    value.item()
                )

            else:
                payload[key] = value

        return payload
