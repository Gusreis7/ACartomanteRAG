from __future__ import annotations

import logging
from uuid import UUID

from src.generation.provider import (
    GenerationCapacityError,
    GenerationServiceError,
)
from src.storage.redis_chat_repository import (
    RedisChatRepository,
    conversation_thread_id,
)
from src.storage.redis_message_repository import (
    RedisMessageRepository,
)


logger = logging.getLogger("acartomante.chat")


class ChatNotFoundError(LookupError):
    pass


class ChatService:
    def __init__(
        self,
        *,
        repository: RedisChatRepository,
        message_repository: RedisMessageRepository,
        rag,
    ) -> None:
        self.repository = repository
        self.message_repository = message_repository
        self.rag = rag

    def send_message(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
        message: str,
        persona: str = "neutral",
    ) -> dict:
        conversation = self.repository.get(
            session_id=session_id,
            chat_id=chat_id,
        )

        if conversation is None:
            raise ChatNotFoundError(
                "Conversa não encontrada."
            )

        turn_id = self.message_repository.append_user_message(
            session_id=session_id,
            chat_id=chat_id,
            content=message,
        )

        result = self.rag.ask(
            message,
            thread_id=conversation_thread_id(
                session_id=session_id,
                chat_id=chat_id,
            ),
            persona=persona,
        )

        evidences = [
            {
                "evidence_id": item["evidence_id"],
                "work": item.get("obra"),
                "section": item.get("secao"),
                "excerpt": item.get("excerpt", ""),
                "chunk_id": item["chunk_id"],
            }
            for item in result["evidences"]
        ]

        self.message_repository.append_assistant_message(
            session_id=session_id,
            chat_id=chat_id,
            turn_id=turn_id,
            content=result["answer"],
            evidences=evidences,
        )

        self.repository.touch(
            session_id=session_id,
            chat_id=chat_id,
        )

        if conversation["title"] == "Nova conversa":
            self.repository.rename(
                session_id=session_id,
                chat_id=chat_id,
                title=message[:60],
            )

        logger.info(
            "chat_answered chat_id=%s persona=%s strategy=%s total_ms=%.2f",
            chat_id,
            persona,
            result["retrieval_strategy"],
            result["timings_ms"].get("total_ms", 0.0),
        )

        return {
            "chat_id": chat_id,
            "answer": result["answer"],
            "evidences": evidences,
        }

    def list_messages(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
    ) -> dict:
        conversation = self.repository.get(
            session_id=session_id,
            chat_id=chat_id,
        )

        if conversation is None:
            raise ChatNotFoundError(
                "Conversa não encontrada."
            )

        return {
            "chat_id": chat_id,
            "messages": self.message_repository.list_messages(
                session_id=session_id,
                chat_id=chat_id,
            ),
        }


__all__ = [
    "ChatService",
    "ChatNotFoundError",
    "GenerationCapacityError",
    "GenerationServiceError",
]
