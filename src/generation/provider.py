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

    def invoke_structured(
        self,
        messages: list,
        schema: Any,
        *,
        include_raw: bool = False,
    ) -> Any:
        """Invoca o provider exigindo uma saída estruturada.

        Providers que não oferecem structured output podem sobrescrever este
        método quando forem adicionados. A v1 possui apenas Ollama.
        """
        raise NotImplementedError(
            f"{type(self).__name__} não implementa structured output."
        )

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> None:
        raise NotImplementedError


class OllamaGenerationProvider(GenerationProvider):
    def __init__(
        self,
        config,
        *,
        reasoning: bool | str | None = False,
        num_batch: int | None = None,
    ) -> None:
        self.config = config
        self._slots = BoundedSemaphore(value=config.max_parallel)
        self._structured_clients: dict[tuple[Any, bool], Any] = {}
        self._runtime_options: dict[str, Any] | None = None

        if num_batch is not None:
            num_batch = int(num_batch)
            if num_batch <= 0:
                raise ValueError("num_batch deve ser > 0.")
            # Quando `options` é enviado por chamada, ChatOllama substitui o
            # conjunto de opções em vez de mesclá-lo. Por isso repetimos aqui
            # os parâmetros do config e acrescentamos apenas `num_batch`.
            self._runtime_options = {
                "num_ctx": int(config.num_ctx),
                "num_predict": int(config.max_tokens),
                "temperature": float(config.temperature),
                "top_p": float(config.top_p),
                "num_batch": num_batch,
            }

        self.client = ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            top_p=config.top_p,
            num_predict=config.max_tokens,
            num_ctx=config.num_ctx,
            keep_alive=config.keep_alive,
            reasoning=reasoning,
            validate_model_on_init=True,
            client_kwargs={"timeout": config.timeout_seconds},
        )

    def _runtime_kwargs(self) -> dict[str, Any]:
        if self._runtime_options is None:
            return {}
        return {"options": dict(self._runtime_options)}

    def _acquire(self) -> None:
        acquired = self._slots.acquire(
            timeout=self.config.acquire_timeout_seconds
        )
        if not acquired:
            raise GenerationCapacityError(
                "Serviço de geração ocupado. Tente novamente."
            )

    def _translate_error(self, exc: Exception) -> None:
        if isinstance(exc, TimeoutError):
            raise exc
        raise GenerationServiceError(
            "Serviço de geração indisponível."
        ) from exc

    def invoke(self, messages: list) -> Any:
        self._acquire()
        try:
            return self.client.invoke(messages, **self._runtime_kwargs())
        except Exception as exc:
            self._translate_error(exc)
        finally:
            self._slots.release()

    def invoke_structured(
        self,
        messages: list,
        schema: Any,
        *,
        include_raw: bool = False,
    ) -> Any:
        self._acquire()
        try:
            key = (schema, bool(include_raw))
            structured = self._structured_clients.get(key)
            if structured is None:
                structured = self.client.with_structured_output(
                    schema=schema,
                    method="json_schema",
                    include_raw=include_raw,
                )
                self._structured_clients[key] = structured
            return structured.invoke(messages, **self._runtime_kwargs())
        except Exception as exc:
            self._translate_error(exc)
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
