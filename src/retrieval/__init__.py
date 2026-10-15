from src.retrieval.types import RetrievalHit, RetrievalResponse

__all__ = ["RetrievalHit", "RetrievalResponse", "RetrievalService"]


def __getattr__(name):
    if name == "RetrievalService":
        from src.retrieval.service import RetrievalService
        return RetrievalService
    raise AttributeError(name)
