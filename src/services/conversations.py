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
