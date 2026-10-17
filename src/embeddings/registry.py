from __future__ import annotations

from collections.abc import Callable

from src.config.config_manager import EmbeddingModelConfig
from src.embeddings.base import Embedder
from src.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)

EmbedderFactory = Callable[..., Embedder]


class EmbedderRegistry:
    def __init__(self) -> None:
        self._backends: dict[
            str,
            EmbedderFactory,
        ] = {}

        self.register(
            SentenceTransformerEmbedder.BACKEND,
            SentenceTransformerEmbedder,
        )

    def register(
        self,
        backend: str,
        factory: EmbedderFactory,
    ) -> None:
        if not backend:
            raise ValueError("backend não pode ser vazio.")

        if backend in self._backends:
            raise ValueError("Backend já registrado: " f"{backend}")

        self._backends[backend] = factory

    def create(
        self,
        model_config: EmbeddingModelConfig,
        *,
        normalize_embeddings: bool,
        device: str,
        batch_size: int,
    ) -> Embedder:
        factory = self._backends.get(model_config.backend)

        if factory is None:
            available = ", ".join(sorted(self._backends))

            raise ValueError(
                "Backend de embedding não "
                f"registrado: "
                f"{model_config.backend}. "
                f"Disponíveis: {available}."
            )

        return factory(
            model_config=model_config,
            normalize_embeddings=(normalize_embeddings),
            device=device,
            batch_size=batch_size,
        )

    def available_backends(
        self,
    ) -> tuple[str, ...]:
        return tuple(sorted(self._backends))
