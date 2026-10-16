from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import logging
import re
from threading import BoundedSemaphore, Lock
from time import sleep
from typing import Any
from urllib.parse import quote

import boto3
import httpx
from botocore.config import Config as AWSConfig
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import ClientError, NoCredentialsError
from langchain_core.messages import AIMessage
from pydantic import ValidationError

logger = logging.getLogger("acartomante.generation")


class GenerationCapacityError(RuntimeError):
    pass


class GenerationServiceError(RuntimeError):
    def __init__(self, message, *, diagnostics=None, batch_fatal=False):
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.batch_fatal = batch_fatal


class GenerationResponseError(GenerationServiceError):
    def __init__(self, message, *, diagnostics=None, retryable=True, batch_fatal=False, response_text=None):
        super().__init__(message, diagnostics=diagnostics, batch_fatal=batch_fatal)
        self.retryable = retryable
        # Só o serviço de enrichment persiste esta saída no S3; nunca no log.
        self.response_text = response_text


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


class _MantleClient:
    """Chat Completions com SigV4; adapta a resposta ao contrato interno Converse."""

    def __init__(self, config, *, session=None, http_client=None) -> None:
        self.config = config
        self.session = session or boto3.Session()
        self.url = f"https://bedrock-mantle.{config.region}.api.aws/v1"
        self.http = http_client or httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds, connect=5),
            follow_redirects=False,
        )

    def _request(self, method, path, payload=None):
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else b""
        for attempt in range(self.config.retry_attempts):
            credentials = self.session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()
            request = AWSRequest(
                method=method, url=self.url + path, data=body,
                headers={"Content-Type": "application/json"},
            )
            # Renova credenciais temporárias e assinatura a cada tentativa.
            SigV4Auth(credentials.get_frozen_credentials(), "bedrock-mantle", self.config.region).add_auth(request)
            try:
                response = self.http.request(
                    method, request.url, content=body, headers=dict(request.headers),
                )
            except httpx.TransportError:
                if attempt + 1 == self.config.retry_attempts:
                    raise GenerationServiceError("Conexão com Bedrock Mantle indisponível.", diagnostics={"reason": "network"}, batch_fatal=True) from None
                sleep(min(2 ** attempt, 8))
                continue
            if response.status_code == 200:
                try:
                    return response.json(), response.headers, attempt
                except ValueError:
                    raise GenerationResponseError("Bedrock Mantle retornou JSON HTTP inválido.", diagnostics={"reason": "malformed_response", "http_status": 200, "request_id": response.headers.get("x-request-id", "unknown")}) from None
            # Não incluir mensagem, payload, prompt ou headers de autenticação no erro.
            code = f"HTTP{response.status_code}"
            quota_scope = "unknown"
            try:
                error = response.json().get("error", {})
                candidate = error.get("code") or error.get("type") if isinstance(error, dict) else None
                if isinstance(candidate, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", candidate):
                    code = candidate
                message = str(error.get("message", "")).lower() if isinstance(error, dict) else ""
                if "per day" in message or "daily" in message:
                    quota_scope = "daily"
                elif "per minute" in message:
                    quota_scope = "minute"
            except (ValueError, AttributeError):
                pass
            if response.status_code in {429, 500, 502, 503, 504} and quota_scope != "daily" and attempt + 1 < self.config.retry_attempts:
                retry_after = response.headers.get("retry-after", "")
                delay = float(retry_after) if retry_after.isdigit() else 2 ** attempt
                if delay <= 60:
                    sleep(delay)
                    continue
            raise ClientError({
                "Error": {"Code": code, "Message": "Falha na API Mantle", "QuotaScope": quota_scope},
                "ResponseMetadata": {
                    "HTTPStatusCode": response.status_code,
                    "RequestId": response.headers.get("x-amzn-requestid") or response.headers.get("x-request-id", "unknown"),
                },
            }, "CreateInference" if method == "POST" else "GetModel")

    def get_model(self):
        self._request("GET", "/models/" + quote(self.config.model, safe=""))

    def converse(self, *, modelId, system, messages, inferenceConfig):
        chat = [{"role": "system", "content": "\n\n".join(item["text"] for item in system)}] if system else []
        chat.extend({"role": item["role"], "content": "\n".join(block["text"] for block in item["content"])} for item in messages)
        result, headers, retries = self._request("POST", "/chat/completions", {
            "model": modelId, "messages": chat, "stream": False,
            "max_tokens": inferenceConfig["maxTokens"],
            "temperature": inferenceConfig["temperature"], "top_p": inferenceConfig["topP"],
        })
        choice = result["choices"][0]
        content = choice["message"].get("content") or ""
        stop = choice.get("finish_reason")
        usage = result.get("usage", {})
        return {
            "output": {"message": {"content": [{"text": content}]}},
            "stopReason": {"length": "max_tokens", "stop": "end_turn"}.get(stop, stop or "unknown"),
            "usage": {"inputTokens": usage.get("prompt_tokens", 0), "outputTokens": usage.get("completion_tokens", 0)},
            "ResponseMetadata": {
                "RetryAttempts": retries, "RequestId": headers.get("x-amzn-requestid") or headers.get("x-request-id"),
                "FinishReason": str(stop or "unknown")[:80],
                "ReasoningTokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                "ContentChars": len(content),
            },
        }


class BedrockGenerationProvider(GenerationProvider):
    """Converse ou Mantle com IAM, mantendo validação JSON e métricas comuns."""

    def __init__(self, config, *, client=None, control_client=None) -> None:
        self.config = config
        self._slots = BoundedSemaphore(config.max_parallel)
        self._usage_lock = Lock()
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        sdk_config = AWSConfig(
            connect_timeout=5, read_timeout=config.timeout_seconds,
            retries={"mode": "standard", "total_max_attempts": config.retry_attempts},
        )
        mantle = config.provider == "bedrock-mantle"
        self.client = client or (_MantleClient(config) if mantle else boto3.client("bedrock-runtime", region_name=config.region, config=sdk_config))
        self.control_client = control_client if mantle else (control_client or boto3.client("bedrock", region_name=config.region, config=sdk_config))

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

    def invoke(self, messages: list, *, max_tokens=None) -> AIMessage:
        if not self._slots.acquire(timeout=self.config.acquire_timeout_seconds):
            raise GenerationCapacityError("Serviço de geração ocupado. Tente novamente.")
        try:
            system, conversation = self._messages(messages)
            if self.config.model in {"qwen.qwen3-32b-v1:0", "qwen.qwen3-32b"} and not self.config.reasoning:
                # Soft switch documentado pelo Qwen; não presumir um parâmetro
                # de template privado na API Converse gerenciada.
                system.append({"text": "/no_think"})
            response = self.client.converse(
                modelId=self.config.model, system=system, messages=conversation,
                inferenceConfig={
                    "temperature": self.config.temperature, "topP": self.config.top_p,
                    "maxTokens": self.config.max_tokens if max_tokens is None else max_tokens,
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
            metadata = response.get("ResponseMetadata", {})
            stop = response.get("stopReason")
            diagnostics = {
                "provider": self.config.provider, "model": self.config.model,
                "finish_reason": metadata.get("FinishReason", stop), "request_id": metadata.get("RequestId"),
                "input_tokens": incoming, "output_tokens": outgoing,
                "reasoning_tokens": metadata.get("ReasoningTokens", 0),
                "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            }
            reason = None
            if stop in {"max_tokens", "model_context_window_exceeded"}:
                reason = "truncated"
            elif stop in {"quota_exceeded", "rate_limit_exceeded", "throttled"}:
                reason = "quota"
            elif stop in {"content_filter", "guardrail_intervened"}:
                reason = "filtered"
            elif self.config.provider == "bedrock-mantle" and stop != "end_turn":
                reason = "unexpected_finish_reason"
            elif not text:
                reason = "empty"
            if reason:
                diagnostics["reason"] = reason
                diagnostics["output_limit_reached"] = outgoing >= diagnostics["max_tokens"]
                logger.warning("generation_response_error %s", json.dumps(diagnostics, ensure_ascii=False))
                raise GenerationResponseError(
                    f"Resposta do modelo rejeitada: {reason}; finish_reason={diagnostics['finish_reason']}; request_id={diagnostics['request_id']}.",
                    diagnostics=diagnostics, retryable=reason in {"empty", "truncated", "unexpected_finish_reason"}, batch_fatal=reason == "quota",
                )
            prices = self.config.model_prices.get(self.config.model, {})
            return AIMessage(
                content=text,
                usage_metadata={"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": incoming + outgoing},
                response_metadata={
                    "provider": self.config.provider, "model": self.config.model,
                    "region": self.config.region, "stop_reason": response.get("stopReason"),
                    "retry_attempts": response.get("ResponseMetadata", {}).get("RetryAttempts", 0),
                    "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
                    "diagnostics": diagnostics,
                    "estimated_cost_usd": (
                        incoming * prices.get("input", self.config.input_usd_per_million) + outgoing * prices.get("output", self.config.output_usd_per_million)
                    ) / 1_000_000,
                },
            )
        except (GenerationCapacityError, GenerationServiceError, ValueError):
            raise
        except Exception as exc:
            details = getattr(exc, "response", {}) or {}
            error = details.get("Error", {})
            metadata = details.get("ResponseMetadata", {})
            code, status = error.get("Code", "unknown"), metadata.get("HTTPStatusCode")
            quota = status == 429 or any(word in code.lower() for word in ("throttl", "quota", "rate_limit"))
            permission = status in {401, 403} or "credentials" in type(exc).__name__.lower()
            diagnostics = {"provider": self.config.provider, "reason": "quota" if quota else "permission" if permission else "provider_error", "error_code": code, "error_type": type(exc).__name__,
                           "http_status": status, "request_id": metadata.get("RequestId", "unknown"), "quota_scope": error.get("QuotaScope", "unknown")}
            logger.warning("bedrock_unavailable %s", json.dumps(diagnostics))
            raise GenerationServiceError("Bedrock indisponível; consulte o diagnóstico nos logs e no manifest.", diagnostics=diagnostics, batch_fatal=True) from None
        finally:
            self._slots.release()

    def _retry_response(self, error, attempt, *, next_max_tokens=None):
        error.diagnostics["response_attempts"] = attempt + 1
        if not error.retryable or attempt >= self.config.response_retries:
            return False
        delay = 10 * (2 ** attempt)
        logger.warning("generation_retry retry=%s/%s reason=%s delay_seconds=%s next_max_tokens=%s", attempt + 1, self.config.response_retries, error.diagnostics.get("reason"), delay, next_max_tokens if next_max_tokens is not None else error.diagnostics.get("max_tokens"))
        sleep(delay)
        return True

    def invoke_structured(self, messages, schema, *, include_raw=False):
        instruction = "Retorne somente um objeto JSON válido, sem Markdown, conforme este JSON Schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        last_error = None
        raw = None
        max_tokens = self.config.max_tokens
        for attempt in range(self.config.response_retries + 1):
            raw, parsed = None, None
            try:
                raw = self.invoke([("system", instruction), *messages], max_tokens=max_tokens)
                content = raw.content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
                parsed = schema.model_validate_json(content)
                if not any(parsed.model_dump().values()):
                    raise ValueError("Saída estruturada vazia.")
                return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed
            except GenerationResponseError as exc:
                last_error = exc
            except (ValueError, TypeError) as exc:
                if raw is None:
                    raise
                errors = exc.errors(include_url=False, include_context=False, include_input=False) if isinstance(exc, ValidationError) else []
                kind = "json_syntax" if any(item["type"] == "json_invalid" for item in errors) else "schema_validation" if errors else "empty_structure" if parsed is not None else "validation_error"
                diagnostics = {
                    **raw.response_metadata.get("diagnostics", {}), "reason": "invalid_json",
                    "validation_kind": kind, "schema_name": schema.__name__,
                    "validation_errors": [{"type": item["type"], "loc": list(item["loc"])} for item in errors[:8]],
                    "response_chars": len(raw.content),
                    "response_sha256": hashlib.sha256(raw.content.encode()).hexdigest(),
                }
                if parsed is not None:
                    diagnostics["parsed_fields"] = list(parsed.model_dump())
                last_error = GenerationResponseError("Modelo retornou JSON inválido ou fora do schema.", diagnostics=diagnostics, response_text=raw.content)
                logger.warning("generation_response_error %s", json.dumps(diagnostics))
            next_max_tokens = max_tokens
            if (
                last_error.diagnostics.get("reason") == "truncated"
                and last_error.diagnostics.get("output_limit_reached")
                and last_error.diagnostics.get("finish_reason") in {"length", "max_tokens"}
                and self.config.response_retry_max_tokens > max_tokens
            ):
                next_max_tokens = min(max_tokens * 2, self.config.response_retry_max_tokens)
            if not self._retry_response(last_error, attempt, next_max_tokens=next_max_tokens):
                break
            max_tokens = next_max_tokens
        if include_raw:
            return {"raw": raw, "parsed": None, "parsing_error": last_error}
        raise last_error from None

    def usage_snapshot(self):
        with self._usage_lock:
            return dict(self._usage)

    def healthcheck(self) -> None:
        # APIs de catálogo: não geram tokens nem provam autorização de InvokeModel.
        if self.config.provider == "bedrock-mantle":
            self.client.get_model()
        elif self.config.model.startswith(("us.", "eu.", "apac.")):
            self.control_client.get_inference_profile(inferenceProfileIdentifier=self.config.model)
        else:
            self.control_client.get_foundation_model(modelIdentifier=self.config.model)

    def warmup(self) -> None:
        self.healthcheck()

    def preflight(self) -> None:
        # Uma chamada curta comprova acesso à inferência antes do trabalho em CPU.
        for attempt in range(self.config.response_retries + 1):
            try:
                self.invoke([("human", "Responda somente OK.")], max_tokens=64)
                return
            except GenerationResponseError as exc:
                if not self._retry_response(exc, attempt):
                    raise


def create_generation_provider(config, **kwargs) -> GenerationProvider:
    if config.provider in {"bedrock", "bedrock-mantle"}:
        return BedrockGenerationProvider(config)
    kwargs.setdefault("reasoning", config.reasoning)
    return OllamaGenerationProvider(config, **kwargs)
