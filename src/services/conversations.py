from __future__ import annotations

from uuid import UUID

from src.storage.redis_chat_repository import RedisChatRepository


class ConversationService:
    def __init__(self, repository: RedisChatRepository) -> None:
        self.repository = repository

    def create(self, *, session_id: UUID, title: str | None = None) -> dict:
        return self.repository.create(
            session_id=session_id,
            title=title or "Nova conversa",
        )

    def list_for_session(self, session_id: UUID) -> list[dict]:
        return self.repository.list_for_session(session_id)

    def guide(self, *, session_id: UUID, chat_id: UUID) -> dict | None:
        return self.repository.get_guide(session_id=session_id, chat_id=chat_id)

    def draw_guide(self, *, session_id: UUID, chat_id: UUID, card: int) -> dict | None:
        return self.repository.draw_guide(
            session_id=session_id, chat_id=chat_id, card=card
        )

    def set_persona(
        self, *, session_id: UUID, chat_id: UUID, persona: str
    ) -> dict | None:
        return self.repository.set_persona(
            session_id=session_id, chat_id=chat_id, persona=persona
        )

    def tutorial_completed(self, session_id: UUID) -> bool:
        return bool(self.repository.redis.get(f"acartomante:tutorial:v1:{session_id}"))

    def complete_tutorial(self, session_id: UUID) -> None:
        self.repository.redis.set(
            f"acartomante:tutorial:v1:{session_id}", "1", ex=30 * 24 * 3600
        )
