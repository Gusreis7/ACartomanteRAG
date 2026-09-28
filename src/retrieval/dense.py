from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.retrieval.types import RetrievalHit, RetrievalResponse


class DenseRetriever:
    def __init__(self, config, resolved_index) -> None:
        self.config = config
        self.resolved = resolved_index

        embedding = resolved_index.manifest.get("embedding") or {}
        model_name = embedding.get("model") or resolved_index.model
        resolved_revision = (
            embedding.get("resolved_revision")
            or resolved_index.resolved_revision
        )

        if not model_name:
            raise ValueError("Manifest do índice sem embedding.model.")
        if not resolved_revision:
            raise ValueError(
                "Manifest do índice sem embedding.resolved_revision."
            )

        configured_device = config.embedding.execution.device
        device = (
            None
            if configured_device in {None, "", "auto"}
            else configured_device
        )

        truncate_dim = embedding.get("truncate_dim")
        model_kwargs: dict[str, Any] = {
            "revision": resolved_revision,
            "device": device,
        }
        if truncate_dim is not None:
            model_kwargs["truncate_dim"] = int(truncate_dim)

        self.model = SentenceTransformer(model_name, **model_kwargs)
        self.query_prompt_name = embedding.get("query_prompt_name")
        self.normalize_embeddings = bool(
            embedding.get("normalize_embeddings", False)
        )
        self.expected_dimension = int(
            embedding.get("dimension") or resolved_index.dimension
        )

        qdrant_cfg = config.environment.qdrant
        kwargs: dict[str, Any] = {"url": qdrant_cfg.url}
        api_key = getattr(qdrant_cfg, "api_key", None)
        if api_key:
            kwargs["api_key"] = api_key
        self.qdrant = QdrantClient(**kwargs)

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        method: str = "dense",
    ) -> RetrievalResponse:
        total_started = perf_counter()

        embedding_started = perf_counter()
        encode_kwargs: dict[str, Any] = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress_bar": False,
        }
        if self.query_prompt_name:
            encode_kwargs["prompt_name"] = self.query_prompt_name

        vector = self.model.encode([query], **encode_kwargs)[0]
        vector = np.asarray(vector, dtype=np.float32)
        embedding_ms = (perf_counter() - embedding_started) * 1000.0

        if vector.ndim != 1:
            raise ValueError("Embedding da query não é um vetor 1-D.")
        if len(vector) != self.expected_dimension:
            raise ValueError(
                "Dimensão da query não corresponde ao índice. "
                f"Esperado={self.expected_dimension}; recebido={len(vector)}."
            )

        qdrant_started = perf_counter()
        response = self.qdrant.query_points(
            collection_name=self.resolved.collection,
            query=vector.tolist(),
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
        )
        points = response.points
        qdrant_ms = (perf_counter() - qdrant_started) * 1000.0

        hits = []
        for rank, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            chunk_id = str(payload.get("chunk_id", point.id))
            text = str(payload.get("text", ""))
            hits.append(
                RetrievalHit(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=float(point.score),
                    text=text,
                    metadata={
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                )
            )

        total_ms = (perf_counter() - total_started) * 1000.0
        return RetrievalResponse(
            method=method,
            query=query,
            hits=hits,
            timings_ms={
                "embedding_ms": embedding_ms,
                "qdrant_ms": qdrant_ms,
                "total_ms": total_ms,
            },
            silver_dataset_id=self.resolved.silver_dataset_id,
            index_id=self.resolved.index_id,
            collection=self.resolved.collection,
        )
