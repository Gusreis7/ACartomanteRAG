"""Export público preservado, sem carregar embeddings ao importar um provider."""

__all__ = ["MachadoRAG"]


def __getattr__(name):
    if name == "MachadoRAG":
        from src.generation.graph import MachadoRAG

        return MachadoRAG
    raise AttributeError(name)
