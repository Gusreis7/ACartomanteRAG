"""Tiragem e persona por conversa, com Redis em memória e sem inferência."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import fakeredis
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from redis import RedisError, WatchError

from src.api.dependencies import SESSION_COOKIE
from src.api.routes import router
from src.services.conversations import ConversationService
from src.storage.redis_chat_repository import RedisChatRepository


@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr("src.storage.redis_chat_repository.choice", lambda _: "cosmo")
    redis = fakeredis.FakeRedis(decode_responses=True)
    repository = RedisChatRepository(redis)
    app = FastAPI()
    app.state.secure_cookies = True
    app.state.conversation_service = ConversationService(repository)
    app.state.chat_service = Mock()
    app.include_router(router)
    with TestClient(app, base_url="https://testserver") as client:
        conversation = client.post("/api/v1/conversations", json={}).json()
        session = UUID(client.cookies.get(SESSION_COOKIE))
        chat_id = UUID(conversation["chat_id"])
        yield SimpleNamespace(
            client=client, app=app, redis=redis, repository=repository,
            session=session, chat_id=chat_id, conversation=conversation,
            path=f"/api/v1/conversations/{chat_id}/guide",
            key=repository._chat_key(session_id=session, chat_id=chat_id),
        )


def test_new_chat_hides_deck_until_choice_and_does_not_call_generation(web):
    assert web.client.get(web.path).json() == {
        "draw_required": True, "persona": "neutral",
        "drawn_guide": None, "selected_card": None,
    }
    assert "guide_first" not in web.conversation
    assert "guide_first" not in web.client.get("/api/v1/conversations").text
    assert web.app.state.chat_service.mock_calls == []


@pytest.mark.parametrize("first,card,expected", [
    ("cosmo", 0, "cosmo"), ("cosmo", 1, "damiao"),
    ("damiao", 0, "damiao"), ("damiao", 1, "cosmo"),
])
def test_either_position_can_reveal_either_guide(web, first, card, expected):
    web.redis.hset(web.key, "guide_first", first)
    response = web.client.post(web.path + "/draw", json={"card": card})
    assert response.status_code == 200
    assert response.json() == {
        "draw_required": False, "persona": expected,
        "drawn_guide": expected, "selected_card": card,
    }
    assert web.client.get(web.path).json() == response.json()
    assert web.app.state.chat_service.mock_calls == []


def test_retry_and_other_card_cannot_replace_first_choice(web):
    first = web.client.post(web.path + "/draw", json={"card": 0}).json()
    for card in (0, 1, 1):
        assert web.client.post(web.path + "/draw", json={"card": card}).json() == first


def test_each_new_chat_receives_its_own_draw(web, monkeypatch):
    web.client.post(web.path + "/draw", json={"card": 0})
    monkeypatch.setattr("src.storage.redis_chat_repository.choice", lambda _: "damiao")
    other = web.client.post("/api/v1/conversations", json={}).json()
    path = f'/api/v1/conversations/{other["chat_id"]}/guide'
    assert web.client.get(path).json()["draw_required"] is True
    assert web.client.post(path + "/draw", json={"card": 0}).json()["persona"] == "damiao"
    assert web.client.get(web.path).json()["persona"] == "cosmo"


def test_persona_changes_persist_without_replacing_original_card(web):
    original = web.client.post(web.path + "/draw", json={"card": 0}).json()
    for persona in ("neutral", "damiao", "cosmo"):
        response = web.client.patch(web.path, json={"persona": persona})
        assert response.status_code == 200
        assert response.json() == {**original, "persona": persona}
        assert web.client.get(web.path).json() == response.json()
        assert web.client.post(web.path + "/draw", json={"card": 1}).json() == response.json()


def test_skipping_draw_keeps_neutral_on_reload(web):
    state = web.client.patch(web.path, json={"persona": "neutral"}).json()
    assert state == {
        "draw_required": False, "persona": "neutral",
        "drawn_guide": None, "selected_card": None,
    }
    assert web.client.get(web.path).json() == state
    assert web.client.post(web.path + "/draw", json={"card": 0}).json() == state


def test_other_sessions_cannot_read_draw_or_change_persona(web):
    before = web.redis.hgetall(web.key)
    keys = set(web.redis.keys())
    with TestClient(web.app, base_url="https://testserver") as outsider:
        assert outsider.get(web.path).status_code == 404
        assert outsider.post(web.path + "/draw", json={"card": 0}).status_code == 404
        assert outsider.patch(web.path, json={"persona": "damiao"}).status_code == 404
    assert web.redis.hgetall(web.key) == before
    assert set(web.redis.keys()) == keys


@pytest.mark.parametrize("card", [-1, 2, True, False, "0", 0.0, None])
def test_bad_card_is_rejected_without_saving(web, card):
    before = web.redis.hgetall(web.key)
    assert web.client.post(web.path + "/draw", json={"card": card}).status_code == 422
    assert web.redis.hgetall(web.key) == before


@pytest.mark.parametrize("persona", ["random", "<script>", "", None, 0])
def test_bad_persona_cannot_end_draw(web, persona):
    assert web.client.patch(web.path, json={"persona": persona}).status_code == 422
    assert web.client.get(web.path).json()["draw_required"] is True


def test_legacy_chat_opens_without_forcing_another_draw(web):
    web.redis.hdel(web.key, "guide_pending", "guide_first", "persona")
    state = web.client.get(web.path).json()
    assert state == {
        "draw_required": False, "persona": "neutral",
        "drawn_guide": None, "selected_card": None,
    }
    assert web.client.post(web.path + "/draw", json={"card": 0}).json() == state
    assert web.client.patch(web.path, json={"persona": "cosmo"}).json()["persona"] == "cosmo"


@pytest.mark.parametrize("method,suffix,body,operation", [
    ("GET", "", None, "guide"),
    ("POST", "/draw", {"card": 0}, "draw_guide"),
    ("PATCH", "", {"persona": "neutral"}, "set_persona"),
])
def test_storage_errors_are_retryable_and_do_not_expose_connection_details(web, monkeypatch, method, suffix, body, operation):
    monkeypatch.setattr(web.app.state.conversation_service, operation, Mock(side_effect=RedisError("private redis URL")))
    response = web.client.request(method, web.path + suffix, json=body)
    assert response.status_code == 503
    assert "private redis URL" not in response.text
    assert web.redis.hget(web.key, "guide_pending") == "1"


def test_concurrent_choices_return_the_same_saved_card(web, monkeypatch):
    real_pipeline = web.redis.pipeline
    barrier = Barrier(2)

    def synchronized_pipeline(*args, **kwargs):
        pipe = real_pipeline(*args, **kwargs)
        read = pipe.hgetall

        def snapshot(key):
            data = read(key)
            if data.get("guide_pending") == "1":
                barrier.wait(timeout=5)
            return data

        pipe.hgetall = snapshot
        return pipe

    monkeypatch.setattr(web.redis, "pipeline", synchronized_pipeline)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(
            lambda card: web.repository.draw_guide(session_id=web.session, chat_id=web.chat_id, card=card),
            [0, 1],
        ))
    assert results[0] == results[1]
    assert results[0] == web.repository.get_guide(session_id=web.session, chat_id=web.chat_id)


def test_repeated_transaction_conflicts_leave_draw_pending(web, monkeypatch):
    pipeline = web.redis.pipeline

    def conflicting_pipeline(*args, **kwargs):
        pipe = pipeline(*args, **kwargs)
        pipe.execute = Mock(side_effect=WatchError("simulated conflict"))
        return pipe

    monkeypatch.setattr(web.redis, "pipeline", conflicting_pipeline)
    response = web.client.post(web.path + "/draw", json={"card": 0})
    assert response.status_code == 503
    assert web.client.get(web.path).json()["draw_required"] is True


def test_missing_chat_is_not_created_by_draw_or_persona(web):
    missing = f"/api/v1/conversations/{uuid4()}/guide"
    keys = set(web.redis.keys())
    assert web.client.get(missing).status_code == 404
    assert web.client.post(missing + "/draw", json={"card": 0}).status_code == 404
    assert web.client.patch(missing, json={"persona": "cosmo"}).status_code == 404
    assert set(web.redis.keys()) == keys
