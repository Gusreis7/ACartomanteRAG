from __future__ import annotations

from time import perf_counter

import numpy as np
from sentence_transformers import CrossEncoder

from src.retrieval.types import RetrievalHit, RetrievalResponse


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str = "main",
        device: str | None = None,
    ) -> None:
        if device in {"", "auto"}:
            device = None
        self.model_name = model_name
        self.revision = revision
        self.model = CrossEncoder(
            model_name,
            revision=revision,
            device=device,
        )

    def rerank(
        self,
        query: str,
        dense_response: RetrievalResponse,
        top_k: int = 10,
    ) -> RetrievalResponse:
        started = perf_counter()

        if not dense_response.hits:
            return RetrievalResponse(
                method="dense_cross_encoder",
                query=query,
                hits=[],
                timings_ms={
                    **dense_response.timings_ms,
                    "rerank_ms": 0.0,
                    "total_ms": dense_response.timings_ms.get(
                        "total_ms", 0.0
                    ),
                },
                silver_dataset_id=dense_response.silver_dataset_id,
                index_id=dense_response.index_id,
                collection=dense_response.collection,
            )

        pairs = [(query, hit.text) for hit in dense_response.hits]
        scores = np.asarray(
            self.model.predict(pairs, show_progress_bar=False)
        ).reshape(-1)
        ordered = np.argsort(-scores)
        top_k = min(int(top_k), len(ordered))

        hits = []
        for new_rank, source_idx in enumerate(ordered[:top_k], start=1):
            original = dense_response.hits[int(source_idx)]
            metadata = dict(original.metadata)
            metadata["dense_rank"] = original.rank
            metadata["dense_score"] = original.score
            hits.append(
                RetrievalHit(
                    rank=new_rank,
                    chunk_id=original.chunk_id,
                    score=float(scores[int(source_idx)]),
                    text=original.text,
                    metadata=metadata,
                )
            )

        rerank_ms = (perf_counter() - started) * 1000.0
        base_total = float(dense_response.timings_ms.get("total_ms", 0.0))

        return RetrievalResponse(
            method="dense_cross_encoder",
            query=query,
            hits=hits,
            timings_ms={
                "embedding_ms": float(
                    dense_response.timings_ms.get("embedding_ms", 0.0)
                ),
                "qdrant_ms": float(
                    dense_response.timings_ms.get("qdrant_ms", 0.0)
                ),
                "rerank_ms": rerank_ms,
                "total_ms": base_total + rerank_ms,
            },
            silver_dataset_id=dense_response.silver_dataset_id,
            index_id=dense_response.index_id,
            collection=dense_response.collection,
        )
