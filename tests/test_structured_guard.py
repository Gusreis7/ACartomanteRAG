"""Schemas no transporte e parsing real; sem serviços ou inferência externos."""

import json
from threading import BoundedSemaphore
from unittest.mock import Mock

from botocore.credentials import Credentials
import httpx
from pydantic import BaseModel, Field
import pytest

from src.config.config_manager import GenerationConfig
from src.generation.input_guard import InputGuardResult, LiteraryInputGuard
from src.generation.provider import (
    BedrockGenerationProvider,
    GenerationResponseError,
    GenerationServiceError,
    OllamaGenerationProvider,
    _MantleClient,
)


def ollama_provider(monkeypatch, responses, requests):
    # O cliente global do SDK lê proxies ao importar; o teste usa só MockTransport.
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    from langchain_ollama import ChatOllama

    config = GenerationConfig(model="qwen3.5:4b", max_tokens=1024)

    def serve(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/api/chat"
        result = {
            "model": config.model,
            "message": {"role": "assistant", "content": responses[len(requests) - 1]},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 5,
        }
        return httpx.Response(
            200,
            text=json.dumps(result) + "\n",
            headers={"content-type": "application/x-ndjson"},
        )

    provider = object.__new__(OllamaGenerationProvider)
    provider.config = config
    provider._slots = BoundedSemaphore(config.max_parallel)
    provider._structured_clients = {}
    provider._runtime_options = {
        "num_ctx": config.num_ctx,
        "num_predict": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_batch": 64,
    }
    provider.client = ChatOllama(
        model=config.model,
        base_url="http://ollama.test",
        num_predict=config.max_tokens,
        num_ctx=config.num_ctx,
        temperature=config.temperature,
        top_p=config.top_p,
        reasoning=False,
        client_kwargs={"transport": httpx.MockTransport(serve), "trust_env": False},
        async_client_kwargs={"trust_env": False},
    )
    return provider


def mantle_provider(responses, requests, **overrides):
    def serve(request):
        requests.append(request)
        status, content, stop, tokens = responses[len(requests) - 1]
        if status != 200:
            return httpx.Response(
                status,
                json={"error": {"code": "invalid_request", "message": "PRIVATE_BODY"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": stop}],
                "usage": {"prompt_tokens": 20, "completion_tokens": tokens},
            },
            headers={"x-request-id": "req-structured"},
        )

    config = GenerationConfig(
        provider="bedrock-mantle",
        model="qwen.qwen3-32b",
        response_retries=0,
        **overrides,
    )
    session = Mock()
    session.get_credentials.return_value = Credentials(
        "TEST", "test-secret", "test-token"
    )
    client = _MantleClient(
        config,
        session=session,
        http_client=httpx.Client(transport=httpx.MockTransport(serve)),
    )
    return BedrockGenerationProvider(config, client=client)


def test_ollama_uses_native_schema_parser_and_retains_options_and_usage(monkeypatch):
    requests = []
    provider = ollama_provider(
        monkeypatch, ['{"decision":"allow"}', "**Capitu**"], requests
    )
    result = LiteraryInputGuard(provider).review("Quem é Capitu?")
    assert result.decision == "allow" and result.usage["total_tokens"] == 25
    payload = requests[0]
    assert payload["format"] == InputGuardResult.model_json_schema()
    assert payload["format"]["additionalProperties"] is False
    assert (
        payload["options"]["num_predict"] == 128
        and payload["options"]["num_batch"] == 64
    )
    assert payload["options"]["num_ctx"] == provider.config.num_ctx
    assert "JSON" not in payload["messages"][0]["content"]
    assert provider.invoke([("human", "Explique Capitu")]).content == "**Capitu**"
    assert (
        requests[1].get("format") is None
        and requests[1]["options"]["num_predict"] == 1024
    )


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "{}",
        '{"decision":"ALLOW"}',
        '{"decision":"allow","extra":true}',
        '{"decision":null}',
        '{"decision":[]}',
    ],
)
def test_real_ollama_parser_rejects_invalid_classification(monkeypatch, content):
    provider = ollama_provider(monkeypatch, [content], [])
    with pytest.raises(GenerationServiceError, match="validar o assunto"):
        LiteraryInputGuard(provider).review("Quem é Capitu?")
    assert provider._slots.acquire(blocking=False)
    provider._slots.release()


def test_mantle_schema_is_a_signed_api_parameter_and_is_parsed_to_pydantic():
    requests = []
    provider = mantle_provider([(200, '{"decision":"off_topic"}', "stop", 5)], requests)
    result = LiteraryInputGuard(provider).review("Conserte meu carro")
    payload = json.loads(requests[0].content)
    fmt = payload["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "InputGuardResult"
    assert fmt["json_schema"]["schema"] == InputGuardResult.model_json_schema()
    assert payload["max_tokens"] == 128
    assert "/bedrock-mantle/aws4_request" in requests[0].headers["authorization"]
    assert (
        "JSON" not in payload["messages"][0]["content"]
        and "properties" not in payload["messages"][0]["content"]
    )
    assert result.decision == "off_topic" and result.usage["total_tokens"] == 25


def test_converse_sends_output_config_and_leaves_plain_chat_unconstrained():
    client = Mock()
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": '{"decision":"allow"}'}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 20, "outputTokens": 5},
    }
    provider = BedrockGenerationProvider(
        GenerationConfig(provider="bedrock", model="qwen.qwen3-32b-v1:0"),
        client=client,
        control_client=Mock(),
    )
    assert LiteraryInputGuard(provider).review("Quem é Capitu?").decision == "allow"
    request = client.converse.call_args.kwargs
    fmt = request["outputConfig"]["textFormat"]
    assert fmt["type"] == "json_schema"
    assert (
        json.loads(fmt["structure"]["jsonSchema"]["schema"])
        == InputGuardResult.model_json_schema()
    )
    assert request["inferenceConfig"]["maxTokens"] == 128
    assert all("JSON" not in block["text"] for block in request["system"])
    provider.invoke([("human", "Explique Capitu")])
    assert "outputConfig" not in client.converse.call_args.kwargs


def test_mantle_rejects_extra_fields_without_fallback_or_logging_content(caplog):
    requests = []
    provider = mantle_provider(
        [(200, '{"decision":"allow","extra":"PRIVATE_BODY"}', "stop", 12)], requests
    )
    with pytest.raises(GenerationServiceError, match="validar o assunto"):
        LiteraryInputGuard(provider).review("PRIVATE_QUESTION")
    assert len(requests) == 1 and "response_format" in json.loads(requests[0].content)
    assert "PRIVATE_BODY" not in caplog.text and "PRIVATE_QUESTION" not in caplog.text


def test_unsupported_structured_request_does_not_retry_as_free_text(caplog):
    requests = []
    provider = mantle_provider([(400, "", "", 0)], requests)
    with pytest.raises(GenerationServiceError, match="structured output") as error:
        LiteraryInputGuard(provider).review("Quem é Capitu?")
    assert len(requests) == 1
    assert error.value.diagnostics["reason"] == "structured_output_request_rejected"
    assert "PRIVATE_BODY" not in caplog.text


def test_guard_retry_preserves_128_token_cap(monkeypatch):
    requests = []
    responses = [(200, "parcial", "length", 128)] * 3 + [
        (200, '{"decision":"allow"}', "stop", 5)
    ]
    provider = mantle_provider(responses, requests)
    provider.config.response_retries = 3
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    assert LiteraryInputGuard(provider).review("Quem é Capitu?").decision == "allow"
    assert [json.loads(request.content)["max_tokens"] for request in requests] == [
        128
    ] * 4
    assert all("response_format" in json.loads(request.content) for request in requests)


def test_enrichment_wire_schema_is_strict_without_changing_checkpoint_schema():
    class Nested(BaseModel):
        text: str = ""

    class Summary(BaseModel):
        units: list[Nested] = Field(default_factory=list)
        default: str = "retained"

    before = Summary.model_json_schema()
    wire = json.loads(
        BedrockGenerationProvider._output_config(Summary)["textFormat"]["structure"][
            "jsonSchema"
        ]["schema"]
    )
    assert (
        wire["required"] == ["units", "default"]
        and wire["additionalProperties"] is False
    )
    assert wire["properties"]["default"] == {"title": "Default", "type": "string"}
    assert wire["$defs"]["Nested"]["required"] == ["text"]
    assert wire["$defs"]["Nested"]["additionalProperties"] is False
    assert "default" not in wire["$defs"]["Nested"]["properties"]["text"]
    assert Summary.model_json_schema() == before and Summary().model_dump() == {
        "units": [],
        "default": "retained",
    }


@pytest.mark.parametrize(
    "error",
    [
        ValueError("PRIVATE_PARSE"),
        TypeError("PRIVATE_PARSE"),
        NotImplementedError("PRIVATE_PARSE"),
    ],
)
def test_missing_or_failed_structured_parser_does_not_fall_back_to_invoke(error):
    provider = Mock()
    provider.invoke_structured.side_effect = error
    with pytest.raises(GenerationServiceError, match="validar o assunto") as result:
        LiteraryInputGuard(provider).review("Quem é Capitu?")
    assert "PRIVATE_PARSE" not in str(result.value)
    provider.invoke.assert_not_called()


def test_bedrock_fenced_text_is_rejected_instead_of_repaired():
    provider = mantle_provider(
        [(200, '```json\n{"decision":"allow"}\n```', "stop", 8)], []
    )
    with pytest.raises(GenerationResponseError):
        provider.invoke_structured([("human", "Quem é Capitu?")], InputGuardResult)
