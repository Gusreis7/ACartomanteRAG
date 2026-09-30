from __future__ import annotations

from abc import ABC, abstractmethod
from threading import BoundedSemaphore
from typing import Any

from langchain_ollama import ChatOllama


class GenerationCapacityError(RuntimeError):
    pass


class GenerationServiceError(RuntimeError):
    pass


class GenerationProvider(ABC):
    @abstractmethod
    def invoke(self, messages: list) -> Any:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> None:
        raise NotImplementedError


class OllamaGenerationProvider(GenerationProvider):
    def __init__(self, config) -> None:
        self.config = config
        self._slots = BoundedSemaphore(value=config.max_parallel)

        self.client = ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            top_p=config.top_p,
            num_predict=config.max_tokens,
            num_ctx=config.num_ctx,
            keep_alive=config.keep_alive,
            validate_model_on_init=True,
            client_kwargs={"timeout": config.timeout_seconds},
        )

    def invoke(self, messages: list) -> Any:
        acquired = self._slots.acquire(
            timeout=self.config.acquire_timeout_seconds
        )
        if not acquired:
            raise GenerationCapacityError(
                "Serviço de geração ocupado. Tente novamente."
            )

        try:
            return self.client.invoke(messages)
        except TimeoutError:
            raise
        except Exception as exc:
            raise GenerationServiceError(
                "Serviço de geração indisponível."
            ) from exc
        finally:
            self._slots.release()

    def warmup(self) -> None:
        self.invoke([
            ("system", "Responda somente com OK."),
            ("human", "warmup"),
        ])

    def healthcheck(self) -> None:
        self.client.invoke([
            ("human", "Responda somente com OK."),
        ])
