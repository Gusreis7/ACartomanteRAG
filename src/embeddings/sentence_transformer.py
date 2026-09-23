from __future__ import annotations

import warnings

import numpy as np
import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

from src.config.config_manager import EmbeddingModelConfig
from src.embeddings.base import Embedder, EmbedderInfo


class SentenceTransformerEmbedder(Embedder):
    BACKEND = "sentence_transformers"

    def __init__(
        self,
        model_config: EmbeddingModelConfig,
        *,
        normalize_embeddings: bool,
        device: str,
        batch_size: int,
    ) -> None:
        if model_config.pooling != "model":
            raise ValueError(
                "SentenceTransformerEmbedder respeita "
                "o pooling empacotado pelo modelo. "
                "Use pooling='model'."
            )

        if model_config.inference_dtype == "float16":
            raise ValueError(
                "SentenceTransformerEmbedder não oferece "
                "float16 nesta etapa. Use float32 ou bfloat16."
            )

        self.model_config = model_config
        self.normalize_embeddings = normalize_embeddings

        self.requested_device = device
        self.resolved_device = self._resolve_device(
            device
        )

        self.batch_size_requested = batch_size
        self.batch_size_effective = batch_size

        self.resolved_revision = (
            self._resolve_revision(
                model_config.model,
                model_config.revision,
            )
        )

        torch_dtype = self._resolve_torch_dtype(
            model_config.inference_dtype
        )

        self._validate_dtype_device(
            dtype=torch_dtype,
            device=self.resolved_device,
        )

        # Carrega pelo SHA resolvido, e não por "main".
        # model_kwargs controla a precisão real dos pesos/cálculo
        # do backbone. O output externo continua float32.
        self.model = SentenceTransformer(
            model_config.model,
            revision=self.resolved_revision,
            device=self.resolved_device,
            backend="torch",
            truncate_dim=model_config.truncate_dim,
            model_kwargs={
                "torch_dtype": torch_dtype,
            },
        )

        (
            self.document_prompt_name,
            self.query_prompt_name,
        ) = self._resolve_prompts()

        self.dimension = (
            self._resolve_dimension()
        )

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            backend=self.BACKEND,
            model=self.model_config.model,
            configured_revision=(
                self.model_config.revision
            ),
            resolved_revision=(
                self.resolved_revision
            ),
            dimension=self.dimension,
            output_dtype="float32",
            inference_dtype=(
                self.model_config.inference_dtype
            ),
            normalize_embeddings=(
                self.normalize_embeddings
            ),
            pooling=self.model_config.pooling,
            truncate_dim=(
                self.model_config.truncate_dim
            ),
            configured_document_prompt_name=(
                self.model_config.document_prompt_name
            ),
            configured_query_prompt_name=(
                self.model_config.query_prompt_name
            ),
            document_prompt_name=(
                self.document_prompt_name
            ),
            query_prompt_name=(
                self.query_prompt_name
            ),
            requested_device=(
                self.requested_device
            ),
            resolved_device=(
                self.resolved_device
            ),
            batch_size_requested=(
                self.batch_size_requested
            ),
            batch_size_effective=(
                self.batch_size_effective
            ),
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        return self._encode(
            texts,
            prompt_name=(
                self.document_prompt_name
            ),
        )

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(
                "text deve ser str."
            )

        vectors = self._encode(
            [text],
            prompt_name=(
                self.query_prompt_name
            ),
        )

        return vectors[0]

    def _encode(
        self,
        texts: list[str],
        *,
        prompt_name: str | None,
    ) -> np.ndarray:
        batch_size = min(
            self.batch_size_effective,
            max(1, len(texts)),
        )

        while True:
            try:
                vectors = self.model.encode(
                    texts,
                    prompt_name=prompt_name,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    # Precisão do vetor devolvido pelo SentenceTransformers.
                    # A precisão de inferência do backbone é configurada
                    # separadamente via model_kwargs["torch_dtype"].
                    precision="float32",
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=(
                        self.normalize_embeddings
                    ),
                    truncate_dim=(
                        self.model_config.truncate_dim
                    ),
                )

                vectors = np.asarray(
                    vectors,
                    dtype=np.float32,
                )

                # Uma chamada pequena não reduz permanentemente
                # o batch efetivo. Este campo só muda em OOM real.
                return vectors

            except Exception as exc:
                if not self._is_cuda_oom(exc):
                    raise

                if batch_size <= 1:
                    raise RuntimeError(
                        "CUDA sem memória mesmo com "
                        "batch_size=1."
                    ) from exc

                next_batch_size = max(
                    1,
                    batch_size // 2,
                )

                warnings.warn(
                    "CUDA OOM durante geração de embeddings. "
                    "Reduzindo batch_size "
                    f"de {batch_size} para "
                    f"{next_batch_size}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

                batch_size = next_batch_size
                self.batch_size_effective = min(
                    self.batch_size_effective,
                    batch_size,
                )

                torch.cuda.empty_cache()

    def _resolve_prompts(
        self,
    ) -> tuple[str | None, str | None]:
        prompts = (
            getattr(
                self.model,
                "prompts",
                None,
            )
            or {}
        )

        document_prompt = self._resolve_prompt(
            configured_name=(
                self.model_config.document_prompt_name
            ),
            role="document",
            prompts=prompts,
        )

        query_prompt = self._resolve_prompt(
            configured_name=(
                self.model_config.query_prompt_name
            ),
            role="query",
            prompts=prompts,
        )

        return (
            document_prompt,
            query_prompt,
        )

    @staticmethod
    def _resolve_prompt(
        *,
        configured_name: str | None,
        role: str,
        prompts: dict,
    ) -> str | None:
        if configured_name is None:
            return None

        if configured_name in prompts:
            return configured_name

        warnings.warn(
            f"Prompt '{configured_name}' configurado para "
            f"{role} não existe no modelo. "
            f"{role} será codificado sem prompt.",
            RuntimeWarning,
            stacklevel=2,
        )

        return None

    def _resolve_dimension(
        self,
    ) -> int:
        if hasattr(
            self.model,
            "get_embedding_dimension",
        ):
            dimension = (
                self.model
                .get_embedding_dimension()
            )
        else:
            dimension = (
                self.model
                .get_sentence_embedding_dimension()
            )

        if dimension is None:
            raise ValueError(
                "Não foi possível determinar a "
                "dimensão do embedding."
            )

        dimension = int(dimension)

        if (
            self.model_config.truncate_dim
            is not None
        ):
            dimension = min(
                dimension,
                self.model_config.truncate_dim,
            )

        if dimension <= 0:
            raise ValueError(
                "Dimensão de embedding inválida."
            )

        return dimension

    @staticmethod
    def _resolve_revision(
        model: str,
        revision: str,
    ) -> str:
        info = HfApi().model_info(
            repo_id=model,
            revision=revision,
        )

        if not info.sha:
            raise ValueError(
                "Hugging Face não retornou SHA "
                f"para {model}@{revision}."
            )

        return info.sha

    @staticmethod
    def _resolve_device(
        requested_device: str,
    ) -> str:
        if requested_device != "auto":
            return requested_device

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    @staticmethod
    def _resolve_torch_dtype(
        inference_dtype: str,
    ) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        try:
            return mapping[inference_dtype]
        except KeyError as exc:
            raise ValueError(
                "inference_dtype não suportado pelo "
                "SentenceTransformerEmbedder: "
                f"{inference_dtype}"
            ) from exc

    @staticmethod
    def _validate_dtype_device(
        *,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if (
            dtype == torch.bfloat16
            and device.startswith("cuda")
            and hasattr(
                torch.cuda,
                "is_bf16_supported",
            )
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "A GPU CUDA selecionada não suporta "
                "bfloat16 no PyTorch atual."
            )

    def _is_cuda_oom(
        self,
        exc: Exception,
    ) -> bool:
        if not self.resolved_device.startswith(
            "cuda"
        ):
            return False

        if isinstance(
            exc,
            torch.cuda.OutOfMemoryError,
        ):
            return True

        return (
            "out of memory"
            in str(exc).lower()
        )
