from __future__ import annotations

from time import perf_counter

from src.indexing.index_resolver import IndexResolver
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.corpus import load_resolved_silver_chunks
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.storage.rustfs_manager import RustFSManager


class RetrievalService:
    """Interface comum para BM25, Dense e Dense + CrossEncoder."""

    def __init__(self, config, *, index_id: str | None = None) -> None:
        started = perf_counter()
        self.config = config
        self.rustfs = RustFSManager.from_config(config.environment.rustfs)
        self.resolved = IndexResolver(config, self.rustfs).resolve(
            index_id=index_id
        )
        self.chunks = load_resolved_silver_chunks(
            config,
            self.rustfs,
            self.resolved,
        )

        self.bm25 = BM25Retriever(
            self.chunks,
            silver_dataset_id=self.resolved.silver_dataset_id,
            index_id=self.resolved.index_id,
            collection=self.resolved.collection,
        )

        # Lazy: BM25 não deve carregar embedding model / CE sem necessidade.
        self._dense = None
        self._reranker = None
        self.dense_init_ms: float | None = None
        self.reranker_init_ms: float | None = None
        self.init_ms = (perf_counter() - started) * 1000.0

    @property
    def dense(self) -> DenseRetriever:
        if self._dense is None:
            started = perf_counter()
            self._dense = DenseRetriever(self.config, self.resolved)
            self.dense_init_ms = (perf_counter() - started) * 1000.0
        return self._dense

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            cfg = self.config.retrievals.dense_cross_encoder
            started = perf_counter()
            self._reranker = CrossEncoderReranker(
                model_name=cfg.model,
                revision=cfg.revision,
                device=self.config.embedding.execution.device,
            )
            self.reranker_init_ms = (perf_counter() - started) * 1000.0
        return self._reranker

    def warmup(
        self,
        *,
        runs: int | None = None,
        query: str = "Consulta de aquecimento do sistema de recuperação.",
    ) -> dict:
        """Aquece Dense + CrossEncoder fora das métricas do benchmark."""
        if runs is None:
            runs = int(self.config.retrieval_benchmark.warmup_runs)

        runs = max(int(runs), 0)
        started = perf_counter()
        run_timings = []
        cfg = self.config.retrievals.dense_cross_encoder
        candidate_k = int(cfg.candidate_k)
        final_k = min(int(cfg.top_k), candidate_k)

        for run_index in range(runs):
            dense_response = self.dense.search(
                query,
                top_k=candidate_k,
                method="warmup_dense",
            )
            reranked = self.reranker.rerank(
                query,
                dense_response,
                top_k=final_k,
            )
            run_timings.append(
                {
                    "run": run_index + 1,
                    **reranked.timings_ms,
                }
            )

        return {
            "runs": runs,
            "query": query,
            "total_ms": (perf_counter() - started) * 1000.0,
            "run_timings": run_timings,
        }

    def search(
        self,
        query: str,
        *,
        method: str,
        top_k: int | None = None,
    ):
        method = str(method).strip().lower()

        if method == "bm25":
            cfg = self.config.retrievals.bm25
            final_k = int(top_k) if top_k is not None else int(cfg.top_k)
            return self.bm25.search(query, top_k=final_k)

        if method == "dense":
            cfg = self.config.retrievals.dense
            final_k = int(top_k) if top_k is not None else int(cfg.top_k)
            return self.dense.search(query, top_k=final_k)

        if method in {"dense_ce", "dense_cross_encoder"}:
            cfg = self.config.retrievals.dense_cross_encoder
            final_k = int(top_k) if top_k is not None else int(cfg.top_k)
            candidate_k = max(int(cfg.candidate_k), final_k)
            dense_response = self.dense.search(
                query,
                top_k=candidate_k,
                method="dense_candidates",
            )
            return self.reranker.rerank(
                query,
                dense_response,
                top_k=final_k,
            )

        raise ValueError(
            f"Método de retrieval inválido: {method}. "
            "Use bm25, dense ou dense_cross_encoder."
        )
