from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalHit:
    rank: int
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResponse:
    method: str
    query: str
    hits: list[RetrievalHit]
    timings_ms: dict[str, float]
    silver_dataset_id: str
    index_id: str | None = None
    collection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "timings_ms": self.timings_ms,
            "silver_dataset_id": self.silver_dataset_id,
            "index_id": self.index_id,
            "collection": self.collection,
        }
