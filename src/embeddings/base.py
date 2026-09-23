from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbedderInfo:
    backend: str
    model: str

    configured_revision: str
    resolved_revision: str

    dimension: int

    output_dtype: str
    inference_dtype: str

    normalize_embeddings: bool
    pooling: str
    truncate_dim: int | None

    configured_document_prompt_name: str | None
    configured_query_prompt_name: str | None

    document_prompt_name: str | None
    query_prompt_name: str | None

    requested_device: str
    resolved_device: str

    batch_size_requested: int
    batch_size_effective: int


class Embedder(ABC):
    @property
    @abstractmethod
    def info(self) -> EmbedderInfo:
        ...

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Retorna matriz float32 com shape (N, D).
        """

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Retorna vetor float32 com shape (D,).
        """
