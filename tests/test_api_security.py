"""Contrato HTTP real, com Redis/LLM substituídos por dependências locais."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import fakeredis
import pytest
import yaml
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from redis import ConnectionError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api.main import invalid_request
from src.api.recommendations import router as recommendation_router
from src.api.routes import router
from src.api.security import (
    APISecurityMiddleware,
    RedisRateLimiter,
    client_identity,
    trusted_proxy_ips,
)
from src.config.config_manager import ApiConfig
from src.services.chat import ChatService
from src.services.conversations import ConversationService
from src.storage.redis_chat_repository import RedisChatRepository
from src.storage.redis_message_repository import RedisMessageRepository

ROOT = Path(__file__).resolve().parents[1]
MESSAGE_PATH = f"/api/v1/conversations/{uuid4()}/messages"


def runtime(*, config=None, trusted=""):
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, invalid_request)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost"])
    app.add_middleware(
        APISecurityMiddleware, trusted_proxies=trusted_proxy_ips(trusted)
    )
    app.include_router(router)
    app.include_router(recommendation_router)
    redis = fakeredis.FakeRedis(decode_responses=True)
    repository = RedisChatRepository(redis)
    messages = RedisMessageRepository(redis)
    rag = Mock()
    rag.config.generation.daily_request_limit = 300
    rag.config.generation.session_requests_per_minute = 6
    app.state.secure_cookies = True
    app.state.api_security_config = config or ApiConfig()
    app.state.redis = redis
    app.state.api_rate_limiter = RedisRateLimiter(redis)
    app.state.http_chat_slots = BoundedSemaphore(2)
    app.state.conversation_service = ConversationService(repository)
    app.state.chat_service = ChatService(
        repository=repository, message_repository=messages, rag=rag
    )
    app.state.work_recommendation_service = Mock()
    app.state.work_recommendation_service.recommend.return_value = []
    return app


def browser(app, address="203.0.113.1"):
    return TestClient(app, base_url="https://localhost", client=(address, 54321))


@pytest.fixture
def app():
    return runtime()


@pytest.mark.parametrize(
    "message",
    [
        1,
        True,
        None,
        [],
        {},
        ["Capitu"],
        {"text": "Capitu"},
        [{"type": "image_url", "image_url": "data:image/png;base64,AA=="}],
        "",
        " \n\t ",
        "Capitu\u0000",
        "Capitu\u001b[31m",
        "\ud800",
        "x" * 4001,
    ],
)
def test_message_requires_bounded_plain_text(app, message):
    response = (
        browser(app).post(MESSAGE_PATH, json={"message": message})
        if not isinstance(message, str) or "\ud800" not in message
        else browser(app).post(
            MESSAGE_PATH,
            content=b'{"message":"\\ud800"}',
            headers={"Content-Type": "application/json"},
        )
    )
    assert response.status_code == 422
    assert "input" not in response.json()
    app.state.chat_service.rag.ask.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        "image",
        "images",
        "file",
        "files",
        "attachment",
        "url",
        "tools",
        "code",
        "system",
        "session_id",
    ],
)
def test_unknown_fields_cannot_enable_uploads_or_tools(app, extra):
    response = browser(app).post(
        MESSAGE_PATH, json={"message": "Capitu", extra: "private-marker"}
    )
    assert response.status_code == 422
    assert "private-marker" not in response.text
    app.state.chat_service.rag.ask.assert_not_called()


@pytest.mark.parametrize("title", [42, {}, [], "\x00", "  ", "x" * 121])
def test_conversation_title_is_strict_text(app, title):
    assert (
        browser(app).post("/api/v1/conversations", json={"title": title}).status_code
        == 422
    )
    assert not app.state.redis.keys("*:metadata")


@pytest.mark.parametrize(
    "media",
    [
        "multipart/form-data; boundary=test",
        "application/octet-stream",
        "application/pdf",
        "image/png",
        "text/plain",
        "text/html",
        "application/json; charset=utf-16",
        "application/ld+json",
    ],
)
def test_api_does_not_accept_other_content_types(app, media):
    response = browser(app).post(
        MESSAGE_PATH, content=b'{"message":"Capitu"}', headers={"Content-Type": media}
    )
    assert response.status_code == 415
    app.state.chat_service.rag.ask.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        b"%PDF-1.7",
        b"\x89PNG",
        b"\xff\xfe{\x00}",
        b'{"message":"\xff"}',
        b"{",
        b"[]",
        b"null",
        b'"Capitu"',
        b'{"message":"Capitu","message":"outro"}',
        b'{"message":NaN}',
        b'{"message":Infinity}',
        b"[" * 1200 + b"]" * 1200,
    ],
)
def test_invalid_or_ambiguous_json_never_reaches_chat(app, body):
    response = browser(app).post(
        MESSAGE_PATH, content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    app.state.chat_service.rag.ask.assert_not_called()


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": "-1"},
        {"Content-Length": "x"},
        {"Content-Length": b"\xef\xbc\x99"},
        {"Content-Length": "9" * 5000},
        {"Content-Length": "999999999"},
        {"Content-Encoding": "gzip"},
    ],
)
def test_framing_and_compression_rejected(app, headers):
    response = browser(app).post(
        MESSAGE_PATH,
        content=b"{}",
        headers={"Content-Type": "application/json", **headers},
    )
    assert response.status_code in {400, 413, 415}


async def asgi_request(app, chunks, *, headers=None, receive_delay=0):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": MESSAGE_PATH,
        "raw_path": MESSAGE_PATH.encode(),
        "root_path": "",
        "query_string": b"",
        "client": ("203.0.113.7", 5555),
        "server": ("localhost", 443),
        "headers": [
            (b"host", b"localhost"),
            (b"content-type", b"application/json"),
            *(headers or []),
        ],
    }
    incoming = iter(chunks)
    sent = []

    async def receive():
        if receive_delay:
            await asyncio.sleep(receive_delay)
        return next(incoming)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return next(
        message["status"]
        for message in sent
        if message["type"] == "http.response.start"
    )


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"2")]])
def test_streaming_body_limit_cannot_be_bypassed_by_content_length(app, headers):
    chunks = [
        {"type": "http.request", "body": b"x" * 16000, "more_body": True}
        for _ in range(3)
    ]
    assert asyncio.run(asgi_request(app, chunks, headers=headers)) == 413
    app.state.chat_service.rag.ask.assert_not_called()


def test_short_body_cannot_lie_about_content_length(app):
    assert (
        asyncio.run(
            asgi_request(
                app,
                [{"type": "http.request", "body": b"{}"}],
                headers=[(b"content-length", b"3")],
            )
        )
        == 400
    )


def test_body_has_total_receive_deadline(app):
    app.state.api_security_config.body_timeout_seconds = 0.01
    assert (
        asyncio.run(
            asgi_request(
                app, [{"type": "http.request", "body": b"{}"}], receive_delay=0.1
            )
        )
        == 408
    )


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
        {"Origin": "https://localhost.attacker.example"},
        {"Origin": "http://localhost"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
    ],
)
def test_cross_origin_and_sibling_sites_are_rejected(app, headers):
    response = browser(app).post("/api/v1/conversations", json={}, headers=headers)
    assert response.status_code == 403
    assert not app.state.redis.keys("*:metadata")


def test_same_origin_or_cli_still_work_and_cookie_is_private(app):
    client = browser(app)
    response = client.post(
        "/api/v1/conversations",
        json={},
        headers={"Origin": "https://localhost", "Sec-Fetch-Site": "same-origin"},
    )
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert all(value in cookie for value in ["httponly", "secure", "samesite=lax"])
    assert client.get("/api/v1/conversations").status_code == 200


def test_changing_cookie_does_not_reset_ip_message_limit(app):
    client = browser(app)
    for _ in range(6):
        client.cookies.set("acartomante_session", str(uuid4()))
        assert (
            client.post(MESSAGE_PATH, json={"message": "Quem é Capitu?"}).status_code
            == 404
        )
    client.cookies.clear()
    response = client.post(MESSAGE_PATH, json={"message": "Quem é Capitu?"})
    assert response.status_code == 429
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert (
        browser(app, "203.0.113.2")
        .post(MESSAGE_PATH, json={"message": "Capitu"})
        .status_code
        == 404
    )


def test_conversation_creation_and_reads_have_independent_limits(app):
    client = browser(app)
    for _ in range(5):
        assert client.post("/api/v1/conversations", json={}).status_code == 200
    assert client.post("/api/v1/conversations", json={}).status_code == 429
    assert client.get("/api/v1/conversations").status_code == 200


def test_read_requests_are_limited_and_health_remains_available(app):
    app.state.api_security_config.requests_per_minute = 2
    client = browser(app)
    assert client.get("/api/v1/conversations").status_code == 200
    assert client.get("/api/v1/recommendations/works").status_code == 200
    assert client.get("/api/v1/conversations").status_code == 429
    assert client.get("/health").status_code == 200


def test_global_creation_limit_is_shared_across_addresses(app):
    app.state.api_security_config.global_conversations_per_day = 2
    assert (
        browser(app, "203.0.113.1").post("/api/v1/conversations", json={}).status_code
        == 200
    )
    assert (
        browser(app, "203.0.113.2").post("/api/v1/conversations", json={}).status_code
        == 200
    )
    response = browser(app, "203.0.113.3").post("/api/v1/conversations", json={})
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 86000


def test_rate_limit_is_atomic_across_parallel_requests(app):
    limiter = app.state.api_rate_limiter
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(
            pool.map(
                lambda _: limiter.check("same-ip", "message", ApiConfig()), range(40)
            )
        )
    assert results.count(0) == 6
    assert all(1 <= result <= 60 for result in results if result)
    assert all(
        app.state.redis.ttl(key) > 0
        for key in app.state.redis.keys("acartomante:http-limits:*")
    )


def test_redis_outage_denies_work_instead_of_disabling_limits(app):
    app.state.api_rate_limiter.redis = Mock()
    app.state.api_rate_limiter.redis.eval.side_effect = ConnectionError(
        "private-secret"
    )
    response = browser(app).post(MESSAGE_PATH, json={"message": "Capitu"})
    assert response.status_code == 503
    assert "private-secret" not in response.text
    app.state.chat_service.rag.ask.assert_not_called()
    assert browser(app).get("/health").status_code == 200


def test_inflight_limit_covers_retrieval_and_generation(app):
    app.state.http_chat_slots.acquire()
    app.state.http_chat_slots.acquire()
    response = browser(app).post(MESSAGE_PATH, json={"message": "Capitu"})
    assert response.status_code == 503
    app.state.chat_service.rag.ask.assert_not_called()


def test_untrusted_forwarded_headers_cannot_change_identity(app):
    app.state.api_security_config.requests_per_minute = 1
    client = browser(app)
    assert (
        client.get(
            "/api/v1/conversations",
            headers={
                "X-Forwarded-For": "198.51.100.1",
                "X-Acartomante-Client-IP": "198.51.100.1",
            },
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/api/v1/conversations",
            headers={
                "X-Forwarded-For": "198.51.100.2",
                "X-Acartomante-Client-IP": "198.51.100.2",
            },
        ).status_code
        == 429
    )


def test_only_pinned_proxy_can_supply_independent_visitors():
    app = runtime(config=ApiConfig(requests_per_minute=1), trusted="172.30.42.3")
    client = browser(app, "172.30.42.3")
    assert (
        client.get(
            "/api/v1/conversations", headers={"X-Acartomante-Client-IP": "203.0.113.1"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/api/v1/conversations", headers={"X-Acartomante-Client-IP": "203.0.113.2"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/api/v1/conversations", headers={"X-Acartomante-Client-IP": "203.0.113.1"}
        ).status_code
        == 429
    )
    assert client.get("/api/v1/conversations").status_code == 400


@pytest.mark.parametrize("value", ["*", "0.0.0.0/0", "172.30.42.0/29", "not-an-ip"])
def test_proxy_trust_requires_exact_ip(value):
    with pytest.raises(ValueError):
        trusted_proxy_ips(value)


def test_ipv6_rotation_inside_same_network_keeps_quota(app):
    app.state.api_security_config.requests_per_minute = 1
    assert browser(app, "2001:db8::1").get("/api/v1/conversations").status_code == 200
    assert browser(app, "2001:db8::2").get("/api/v1/conversations").status_code == 429
    assert (
        browser(app, "2001:db8:0:1::1").get("/api/v1/conversations").status_code == 200
    )


def test_user_cannot_read_other_sessions_history(app):
    owner = browser(app)
    chat = owner.post("/api/v1/conversations", json={}).json()["chat_id"]
    other = browser(app, "203.0.113.2")
    assert owner.get(f"/api/v1/conversations/{chat}/messages").status_code == 200
    assert other.get(f"/api/v1/conversations/{chat}/messages").status_code == 404
    assert (
        other.post(
            f"/api/v1/conversations/{chat}/messages", json={"message": "Capitu"}
        ).status_code
        == 404
    )
    app.state.chat_service.rag.ask.assert_not_called()


def test_source_code_and_html_are_literal_text_never_commands(app, tmp_path):
    marker = tmp_path / "must-not-exist"
    text = f'<script>alert(1)</script> __import__("pathlib").Path("{marker}").touch(); $(id)'
    client = browser(app)
    chat = client.post("/api/v1/conversations", json={}).json()["chat_id"]
    app.state.chat_service.rag.ask.return_value = {
        "answer": text,
        "evidences": [],
        "retrieval_strategy": "bm25",
        "timings_ms": {},
    }
    response = client.post(
        f"/api/v1/conversations/{chat}/messages", json={"message": text}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == text
    assert app.state.chat_service.rag.ask.call_args.args[0] == text
    assert not marker.exists()
    assert (
        client.get(f"/api/v1/conversations/{chat}/messages").json()["messages"][0][
            "content"
        ]
        == text
    )


@pytest.mark.parametrize("path,status", [("/health", 200), (MESSAGE_PATH, 422)])
def test_security_headers_also_cover_errors(app, path, status):
    response = (
        browser(app).get(path)
        if path == "/health"
        else browser(app).post(path, json={"message": []})
    )
    assert response.status_code == status
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert "object-src 'none'" in response.headers["content-security-policy"]
    if path != "/health":
        assert response.headers["cache-control"] == "no-store"


def test_recommendation_parameters_and_query_string_are_bounded(app):
    client = browser(app)
    assert (
        client.get(
            "/api/v1/recommendations/works",
            params=[("exclude", str(i)) for i in range(21)],
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/recommendations/works", params={"exclude": "x" * 121}
        ).status_code
        == 422
    )
    assert (
        client.get("/api/v1/recommendations/works?x=" + "x" * 4096).status_code == 414
    )
    app.state.work_recommendation_service.recommend.assert_not_called()


def test_history_reads_are_bounded_without_deleting_old_records(app):
    repository = app.state.conversation_service.repository
    messages = app.state.chat_service.message_repository
    session, chat = uuid4(), uuid4()
    key = repository._session_chats_key(session)
    for i in range(105):
        repository.create(session_id=session, title=f"Chat {i}")
    assert len(repository.list_for_session(session)) == 100
    assert app.state.redis.zcard(key) == 105
    for i in range(205):
        messages.append_user_message(session_id=session, chat_id=chat, content=str(i))
    rows = messages.list_messages(session_id=session, chat_id=chat)
    assert len(rows) == 200
    assert rows[0]["content"] == "5" and rows[-1]["content"] == "204"
    assert (
        app.state.redis.xlen(messages._messages_key(session_id=session, chat_id=chat))
        == 205
    )


def test_aws_only_pinned_proxy_can_assert_client_identity():
    compose = yaml.safe_load((ROOT / "docker-compose.aws.yml").read_text())
    services = compose["services"]
    trusted = services["api"]["environment"]["API_TRUSTED_PROXY_IPS"]
    assert trusted == services["caddy"]["networks"]["proxy_guarded"]["ipv4_address"]
    assert compose["networks"]["proxy_guarded"]["internal"] is True
    assert "ports" not in services["api"]
    assert services["redis"]["ports"] == ["127.0.0.1:6379:6379"]
    assert (
        "header_up X-Acartomante-Client-IP {http.request.remote.host}"
        in (ROOT / "infra/aws/Caddyfile").read_text()
    )


@pytest.mark.parametrize("body", [None, {}])
def test_tutorial_still_accepts_no_body_or_empty_json(app, body):
    client = browser(app)
    before = client.get("/api/v1/session")
    assert before.json() == {"tutorial_completed": False}
    response = (
        client.post("/api/v1/session/tutorial")
        if body is None
        else client.post("/api/v1/session/tutorial", json=body)
    )
    assert response.status_code == 200
    assert response.json() == {"tutorial_completed": True}
    assert client.get("/api/v1/session").json() == {"tutorial_completed": True}


@pytest.mark.parametrize(
    "suffix,method,valid",
    [
        ("/guide/draw", "POST", {"card": 0}),
        ("/guide", "PATCH", {"persona": "neutral"}),
    ],
)
def test_guarded_http_keeps_draw_and_persona_without_llm_slots(
    app, suffix, method, valid
):
    client = browser(app)
    chat = client.post("/api/v1/conversations", json={}).json()["chat_id"]
    path = f"/api/v1/conversations/{chat}"
    assert client.get(path + "/guide").json()["draw_required"] is True
    app.state.http_chat_slots.acquire()
    app.state.http_chat_slots.acquire()
    response = client.request(method, path + suffix, json=valid)
    assert response.status_code == 200
    assert response.json()["draw_required"] is False
    if suffix == "/guide/draw":
        assert response.json()["drawn_guide"] in {"cosmo", "damiao"}
    else:
        assert response.json()["persona"] == "neutral"
    app.state.chat_service.rag.ask.assert_not_called()


@pytest.mark.parametrize(
    "path,method,body",
    [
        ("/api/v1/session/tutorial", "POST", {"file": "private-marker"}),
        ("/api/v1/session/tutorial", "POST", {"message": "private-marker"}),
        (
            "/api/v1/conversations/{chat}/guide/draw",
            "POST",
            {"card": 0, "image": "private-marker"},
        ),
        (
            "/api/v1/conversations/{chat}/guide",
            "PATCH",
            {"persona": "neutral", "tools": "private-marker"},
        ),
        (
            "/api/v1/conversations/{chat}/guide",
            "PATCH",
            {"persona": {"file": "private-marker"}},
        ),
    ],
)
def test_control_endpoints_reject_extra_fields_and_attachments(app, path, method, body):
    client = browser(app)
    chat = client.post("/api/v1/conversations", json={}).json()["chat_id"]
    response = client.request(method, path.format(chat=chat), json=body)
    assert response.status_code == 422
    assert "private-marker" not in response.text
    assert (
        client.get(f"/api/v1/conversations/{chat}/guide").json()["draw_required"]
        is True
    )
    assert client.get("/api/v1/session").json()["tutorial_completed"] is False


def test_patch_persona_uses_same_content_and_size_protection(app):
    client = browser(app)
    chat = client.post("/api/v1/conversations", json={}).json()["chat_id"]
    path = f"/api/v1/conversations/{chat}/guide"
    assert (
        client.patch(
            path, content=b"%PDF-1.7", headers={"Content-Type": "application/pdf"}
        ).status_code
        == 415
    )
    assert (
        client.patch(
            path, content=b"x" * 32769, headers={"Content-Type": "application/json"}
        ).status_code
        == 413
    )
    assert client.patch(path, content=b'{"persona":"cosmo"}').status_code == 415
    assert (
        client.patch(
            path, json={"persona": "cosmo"}, headers={"Origin": "https://other.example"}
        ).status_code
        == 403
    )
    assert client.get(path).json()["draw_required"] is True


@pytest.fixture
def current_app(monkeypatch):
    from src.api.main import app as actual_app
    from src.services.usage import UsageStats

    local = runtime()
    for name in (
        "api_security_config",
        "redis",
        "api_rate_limiter",
        "http_chat_slots",
        "conversation_service",
        "chat_service",
        "work_recommendation_service",
        "secure_cookies",
    ):
        monkeypatch.setattr(
            actual_app.state, name, getattr(local.state, name), raising=False
        )
    monkeypatch.setattr(
        actual_app.state, "usage", UsageStats(local.state.redis), raising=False
    )
    return actual_app


def test_main_middleware_records_rate_rejections_without_exposing_input(current_app):
    current_app.state.api_security_config.requests_per_minute = 1
    client = browser(current_app)
    assert client.get("/api/v1/session").status_code == 200
    response = client.get("/api/v1/session")
    assert response.status_code == 429
    assert response.headers["cache-control"] == "no-store"
    assert "script-src 'self'" in response.headers["content-security-policy"]
    summary = current_app.state.usage.summary(current_app.state.usage.day())
    assert summary["http_requests"] == 2
    assert summary["http_429"] == 1
    assert summary["sessions_estimated"] == 1
    assert client.get("/health").status_code == 200
    assert current_app.state.usage.summary(summary["day_utc"])["http_requests"] == 2


def test_real_guard_graph_and_markdown_remain_connected_through_security(current_app):
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from src.config.config_manager import ChatConfig, GenerationConfig
    from src.generation.graph import MachadoRAG
    from src.generation.input_guard import InputGuardResult, LiteraryInputGuard, REFUSAL
    from src.retrieval.types import RetrievalResponse

    provider = Mock()
    decision = InputGuardResult(decision="allow")
    provider.invoke_structured.return_value = {
        "parsed": decision,
        "parsing_error": None,
        "raw": AIMessage(content=decision.model_dump_json()),
    }
    provider.invoke.return_value = AIMessage(
        content='**Capitu** [E1]\n\n<img src=x onerror="alert(1)">'
    )
    rag = object.__new__(MachadoRAG)
    rag.config = SimpleNamespace(
        chat=ChatConfig(),
        generation=GenerationConfig(
            daily_request_limit=300, session_requests_per_minute=6
        ),
        retrieval_strategy="dense",
        enrichment_retrieval=SimpleNamespace(enabled=False),
    )
    rag.generation_provider = provider
    rag.retrieval = Mock()
    rag.retrieval.search.return_value = RetrievalResponse(
        method="dense",
        query="",
        hits=[],
        timings_ms={},
        silver_dataset_id="silver-test",
        index_id="idx-test",
        collection="text-test",
    )
    rag.enrichment_retrieval = None
    rag.input_guard = LiteraryInputGuard(provider)
    rag.checkpointer = InMemorySaver()
    rag.graph = rag._build_graph()
    current_app.state.chat_service.rag = rag
    client = browser(current_app)
    chat = client.post("/api/v1/conversations", json={}).json()["chat_id"]
    path = f"/api/v1/conversations/{chat}/messages"
    allowed = client.post(path, json={"message": "Quem é Capitu?"})
    assert allowed.status_code == 200
    assert "<strong>Capitu</strong> [E1]" in allowed.json()["answer_html"]
    assert "<img" not in allowed.json()["answer_html"]
    assert rag.retrieval.search.call_count == 1
    assert provider.invoke.call_count == 1
    provider.invoke_structured.return_value["parsed"] = InputGuardResult(
        decision="off_topic"
    )
    refused = client.post(path, json={"message": "Me dê uma receita de bolo"})
    assert refused.status_code == 200 and refused.json()["answer"] == REFUSAL
    assert rag.retrieval.search.call_count == 1 and provider.invoke.call_count == 1
    assert (
        client.get(path).json()["messages"][1]["content_html"]
        == allowed.json()["answer_html"]
    )
    summary = current_app.state.usage.summary(current_app.state.usage.day())
    assert summary["chat_answered"] == 1 and summary["chat_refused"] == 1
