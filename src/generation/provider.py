from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import re
from threading import BoundedSemaphore, Lock
from typing import Any

import boto3
import httpx
from botocore.config import Config as AWSConfig
from langchain_core.messages import AIMessage

logger = logging.getLogger("acartomante.generation")


class GenerationCapacityError(RuntimeError):
    pass


class GenerationServiceError(RuntimeError):
    pass


class GenerationQuotaError(GenerationCapacityError):
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
        método conforme o contrato de cada serviço.
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
        from langchain_ollama import ChatOllama
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
        response = httpx.get(f"{self.config.base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        names = {item.get("name") for item in response.json().get("models", [])}
        if self.config.model not in names and f"{self.config.model}:latest" not in names:
            raise GenerationServiceError("Modelo Ollama não instalado.")


class BedrockGenerationProvider(GenerationProvider):
    """Converse com IAM. JSON é validado aqui, sem presumir tool calling do modelo."""

    def __init__(self, config, *, client=None, control_client=None) -> None:
        self.config = config
        self._slots = BoundedSemaphore(config.max_parallel)
        self._usage_lock = Lock()
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        sdk_config = AWSConfig(
            connect_timeout=5, read_timeout=config.timeout_seconds,
            retries={"mode": "standard", "total_max_attempts": config.retry_attempts},
        )
        self.client = client or boto3.client("bedrock-runtime", region_name=config.region, config=sdk_config)
        self.control_client = control_client or boto3.client("bedrock", region_name=config.region, config=sdk_config)

    @staticmethod
    def _messages(messages: list) -> tuple[list, list]:
        system, conversation = [], []
        for message in messages:
            if isinstance(message, tuple):
                role, content = message
            else:
                role, content = message.type, message.content
            if not isinstance(content, str):
                raise ValueError("O provider Bedrock deste projeto recebe somente texto.")
            if role == "system":
                system.append({"text": content})
                continue
            role = "assistant" if role in {"ai", "assistant"} else "user"
            if conversation and conversation[-1]["role"] == role:
                conversation[-1]["content"].append({"text": content})
            else:
                conversation.append({"role": role, "content": [{"text": content}]})
        return system, conversation

    def invoke(self, messages: list) -> AIMessage:
        if not self._slots.acquire(timeout=self.config.acquire_timeout_seconds):
            raise GenerationCapacityError("Serviço de geração ocupado. Tente novamente.")
        try:
            system, conversation = self._messages(messages)
            if self.config.model == "qwen.qwen3-32b-v1:0" and not self.config.reasoning:
                # Soft switch documentado pelo Qwen; não presumir um parâmetro
                # de template privado na API Converse gerenciada.
                system.append({"text": "/no_think"})
            response = self.client.converse(
                modelId=self.config.model, system=system, messages=conversation,
                inferenceConfig={
                    "temperature": self.config.temperature, "topP": self.config.top_p,
                    "maxTokens": self.config.max_tokens,
                },
            )
            text = "\n".join(item["text"] for item in response["output"]["message"]["content"] if "text" in item)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            usage = response.get("usage", {})
            incoming, outgoing = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
            with self._usage_lock:
                self._usage["input_tokens"] += incoming
                self._usage["output_tokens"] += outgoing
                self._usage["calls"] += 1
            if not text or response.get("stopReason") == "max_tokens":
                raise GenerationServiceError("Modelo retornou resposta vazia ou truncada.")
            prices = self.config.model_prices.get(self.config.model, {})
            return AIMessage(
                content=text,
                usage_metadata={"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": incoming + outgoing},
                response_metadata={
                    "provider": "bedrock", "model": self.config.model,
                    "region": self.config.region, "stop_reason": response.get("stopReason"),
                    "retry_attempts": response.get("ResponseMetadata", {}).get("RetryAttempts", 0),
                    "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
                    "estimated_cost_usd": (
                        incoming * prices.get("input", self.config.input_usd_per_million) + outgoing * prices.get("output", self.config.output_usd_per_million)
                    ) / 1_000_000,
                },
            )
        except (GenerationCapacityError, GenerationServiceError, ValueError):
            raise
        except Exception as exc:
            details = getattr(exc, "response", {}) or {}
            logger.warning("bedrock_unavailable error_type=%s code=%s request_id=%s", type(exc).__name__, details.get("Error", {}).get("Code", "unknown"), details.get("ResponseMetadata", {}).get("RequestId", "unknown"))
            raise GenerationServiceError("Bedrock indisponível; consulte o tipo de erro nos logs operacionais.") from exc
        finally:
            self._slots.release()

    def invoke_structured(self, messages, schema, *, include_raw=False):
        instruction = "Retorne somente um objeto JSON válido, sem Markdown, conforme este JSON Schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        last_error = None
        raw = None
        for _ in range(2):
            raw = self.invoke([("system", instruction), *messages])
            try:
                content = raw.content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
                parsed = schema.model_validate_json(content)
                if not any(parsed.model_dump().values()):
                    raise ValueError("Saída estruturada vazia.")
                return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed
            except (ValueError, TypeError) as exc:
                last_error = exc
        if include_raw:
            return {"raw": raw, "parsed": None, "parsing_error": last_error}
        raise GenerationServiceError("Bedrock retornou JSON inválido após duas tentativas.") from last_error

    def usage_snapshot(self):
        with self._usage_lock:
            return dict(self._usage)

    def healthcheck(self) -> None:
        # APIs de catálogo: não geram tokens nem provam autorização de InvokeModel.
        if self.config.model.startswith(("us.", "eu.", "apac.")):
            self.control_client.get_inference_profile(inferenceProfileIdentifier=self.config.model)
        else:
            self.control_client.get_foundation_model(modelIdentifier=self.config.model)

    def warmup(self) -> None:
        self.healthcheck()


def create_generation_provider(config, **kwargs) -> GenerationProvider:
    if config.provider == "bedrock":
        return BedrockGenerationProvider(config)
    kwargs.setdefault("reasoning", config.reasoning)
    return OllamaGenerationProvider(config, **kwargs)
