from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import time
from uuid import UUID

from src.services.usage import UsageStats

from src.generation.provider import (
    GenerationCapacityError,
    GenerationServiceError,
    GenerationQuotaError,
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
        self.usage = UsageStats(repository.redis)

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
            raise ChatNotFoundError("Conversa não encontrada.")

        self._check_quota(session_id)

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
        guard = result.get("input_guard", {"decision": "disabled", "usage": {}})

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
            metadata={
                "persona": persona,
                "timings_ms": result["timings_ms"],
                "index_id": result.get("index_id"),
                **result.get("generation_usage", {}),
                "input_guard": guard,
            },
        )

        self.repository.touch(
            session_id=session_id,
            chat_id=chat_id,
        )
        decision = (
            "allow" if guard["decision"] in {"allow", "disabled"} else guard["decision"]
        )
        self.usage.record_chat(
            decision, guard["usage"], result.get("generation_usage", {})
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

    def _check_quota(self, session_id):
        config = self.rag.config.generation
        if not config.daily_request_limit and not config.session_requests_per_minute:
            return
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        allowed = self.repository.redis.eval(
            """
            local minute = tonumber(redis.call('GET', KEYS[1]) or '0')
            local day = tonumber(redis.call('GET', KEYS[2]) or '0')
            if (tonumber(ARGV[1]) > 0 and minute >= tonumber(ARGV[1])) or
               (tonumber(ARGV[2]) > 0 and day >= tonumber(ARGV[2])) then return 0 end
            redis.call('INCR', KEYS[1]); redis.call('EXPIRE', KEYS[1], 120)
            redis.call('INCR', KEYS[2]); redis.call('EXPIRE', KEYS[2], 172800)
            return 1
        """,
            2,
            f"acartomante:limits:session:{session_id}:{int(time() // 60)}",
            f"acartomante:limits:day:{day}",
            config.session_requests_per_minute,
            config.daily_request_limit,
        )
        if not allowed:
            raise GenerationQuotaError(
                "Limite de consultas atingido. Tente novamente mais tarde."
            )

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
            raise ChatNotFoundError("Conversa não encontrada.")

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
