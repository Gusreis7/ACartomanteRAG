from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from redis import Redis

from src.storage.redis_chat_repository import REDIS_NAMESPACE

MessageRole = Literal["user", "assistant"]


class RedisMessageRepository:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    def append_user_message(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
        content: str,
        turn_id: UUID | None = None,
    ) -> UUID:
        resolved_turn_id = turn_id or uuid4()

        self.redis.xadd(
            self._messages_key(
                session_id=session_id,
                chat_id=chat_id,
            ),
            self._serialize_message(
                turn_id=resolved_turn_id,
                role="user",
                content=content,
                evidences=[],
            ),
        )

        return resolved_turn_id

    def append_assistant_message(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
        turn_id: UUID,
        content: str,
        evidences: list[dict],
        metadata: dict | None = None,
    ) -> None:
        self.redis.xadd(
            self._messages_key(
                session_id=session_id,
                chat_id=chat_id,
            ),
            self._serialize_message(
                turn_id=turn_id,
                role="assistant",
                content=content,
                evidences=evidences,
                metadata=metadata,
            ),
        )

    def list_messages(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
    ) -> list[dict]:
        rows = self.redis.xrange(
            self._messages_key(
                session_id=session_id,
                chat_id=chat_id,
            ),
            min="-",
            max="+",
        )

        return [
            self._deserialize_message(
                stream_id=stream_id,
                data=data,
            )
            for stream_id, data in rows
        ]

    @staticmethod
    def _messages_key(
        *,
        session_id: UUID,
        chat_id: UUID,
    ) -> str:
        return f"{REDIS_NAMESPACE}:session:{session_id}:" f"chat:{chat_id}:messages"

    @staticmethod
    def _serialize_message(
        *,
        turn_id: UUID,
        role: MessageRole,
        content: str,
        evidences: list[dict],
        metadata: dict | None = None,
    ) -> dict[str, str]:
        return {
            "turn_id": str(turn_id),
            "role": role,
            "metadata": json.dumps(
                metadata or {}, ensure_ascii=False, separators=(",", ":")
            ),
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evidences": json.dumps(
                evidences,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }

    @staticmethod
    def _deserialize_message(
        *,
        stream_id: str,
        data: dict,
    ) -> dict:
        try:
            evidences = json.loads(data.get("evidences", "[]"))
        except json.JSONDecodeError:
            evidences = []

        return {
            "id": stream_id,
            "turn_id": UUID(data["turn_id"]),
            "role": data["role"],
            "content": data["content"],
            "created_at": datetime.fromisoformat(data["created_at"]),
            "evidences": evidences,
        }


__all__ = [
    "MessageRole",
    "RedisMessageRepository",
]
