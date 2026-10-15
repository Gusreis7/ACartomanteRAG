"""Pipeline de enrichment semântico do corpus."""

from .service import EnrichmentResult, WorkEnrichmentService, WorkSource

__all__ = [
    "EnrichmentResult",
    "WorkEnrichmentService",
    "WorkSource",
]
