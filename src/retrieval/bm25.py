from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval.text import tokenize
from src.retrieval.types import RetrievalHit, RetrievalResponse


class BM25Retriever:
    """BM25 em memória sobre a Silver resolvida."""

    def __init__(
        self,
        chunks: pd.DataFrame,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        silver_dataset_id: str,
        index_id: str | None,
        collection: str | None,
    ) -> None:
        started = perf_counter()
        self.chunks = chunks.reset_index(drop=True)
        self.k1 = float(k1)
        self.b = float(b)
        self.silver_dataset_id = silver_dataset_id
        self.index_id = index_id
        self.collection = collection
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len = np.zeros(len(self.chunks), dtype=np.float32)
        document_frequency: Counter[str] = Counter()

        for idx, text in enumerate(self.chunks["text"].fillna("").astype(str)):
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._doc_len[idx] = len(tokens)
            for token, tf in counts.items():
                self._postings[token].append((idx, int(tf)))
                document_frequency[token] += 1

        self._n_docs = len(self.chunks)
        self._avgdl = float(self._doc_len.mean()) if self._n_docs else 0.0
        self._idf = {
            token: log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }
        self.build_ms = (perf_counter() - started) * 1000.0

    def search(self, query: str, top_k: int = 10) -> RetrievalResponse:
        started = perf_counter()
        query_tokens = tokenize(query)
        scores = np.zeros(self._n_docs, dtype=np.float64)

        if query_tokens and self._n_docs and self._avgdl > 0:
            for token in set(query_tokens):
                idf = self._idf.get(token)
                if idf is None:
                    continue
                for doc_idx, tf in self._postings.get(token, []):
                    dl = float(self._doc_len[doc_idx])
                    denominator = tf + self.k1 * (
                        1.0 - self.b + self.b * dl / self._avgdl
                    )
                    scores[doc_idx] += idf * tf * (self.k1 + 1.0) / denominator

        k = min(max(int(top_k), 0), self._n_docs)
        if k == 0:
            ordered: list[int] = []
        elif k == self._n_docs:
            ordered = np.argsort(-scores).tolist()
        else:
            candidate = np.argpartition(-scores, k - 1)[:k]
            ordered = candidate[np.argsort(-scores[candidate])].tolist()

        hits = []
        for rank, doc_idx in enumerate(ordered, start=1):
            row = self.chunks.iloc[doc_idx]
            metadata = {
                key: _json_value(value) for key, value in row.items() if key != "text"
            }
            hits.append(
                RetrievalHit(
                    rank=rank,
                    chunk_id=str(row["chunk_id"]),
                    score=float(scores[doc_idx]),
                    text=str(row["text"]),
                    metadata=metadata,
                )
            )

        total_ms = (perf_counter() - started) * 1000.0
        return RetrievalResponse(
            method="bm25",
            query=query,
            hits=hits,
            timings_ms={"search_ms": total_ms, "total_ms": total_ms},
            silver_dataset_id=self.silver_dataset_id,
            index_id=self.index_id,
            collection=self.collection,
        )


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
