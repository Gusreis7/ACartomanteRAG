from __future__ import annotations

from datetime import datetime, timezone
from secrets import choice
from uuid import UUID, uuid4

from redis import Redis, RedisError, WatchError


REDIS_NAMESPACE = "acartomante"


def conversation_thread_id(
    *,
    session_id: UUID,
    chat_id: UUID,
) -> str:
    """
    Identificador estável usado pelo LangGraph.

    O RedisSaver acrescenta seus próprios prefixes (`checkpoint`,
    `checkpoint_write`, `write_keys_zset`). Ao estruturar o thread_id,
    o RedisInsight passa a agrupar os checkpoints por sessão e chat.
    """
    return (
        f"{REDIS_NAMESPACE}:session:{session_id}:"
        f"chat:{chat_id}"
    )


class RedisChatRepository:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    def create(
        self,
        *,
        session_id: UUID,
        title: str,
    ) -> dict:
        chat_id = uuid4()
        now = datetime.now(timezone.utc)

        data = {
            "chat_id": str(chat_id),
            "session_id": str(session_id),
            "title": title,
            "status": "active",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "guide_pending": "1",
            "guide_first": choice(("cosmo", "damiao")),
            "persona": "neutral",
        }

        pipe = self.redis.pipeline()
        pipe.hset(
            self._chat_key(
                session_id=session_id,
                chat_id=chat_id,
            ),
            mapping=data,
        )
        pipe.zadd(
            self._session_chats_key(session_id),
            {str(chat_id): now.timestamp()},
        )
        pipe.execute()

        return self._deserialize(data)

    def get(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
    ) -> dict | None:
        data = self.redis.hgetall(
            self._chat_key(
                session_id=session_id,
                chat_id=chat_id,
            )
        )

        return self._deserialize(data) if data else None

    def list_for_session(
        self,
        session_id: UUID,
    ) -> list[dict]:
        ids = self.redis.zrevrange(
            self._session_chats_key(session_id),
            0,
            -1,
        )

        result: list[dict] = []

        for raw_id in ids:
            item = self.get(
                session_id=session_id,
                chat_id=UUID(raw_id),
            )

            if item is not None:
                result.append(item)

        return result

    @staticmethod
    def _guide_state(data: dict) -> dict:
        # Registros anteriores à tiragem continuam abertos em modo neutro.
        return {
            "draw_required": data.get("guide_pending") == "1",
            "persona": data.get("persona", "neutral"),
            "drawn_guide": data.get("drawn_guide") or None,
            "selected_card": int(data["guide_card"]) if data.get("guide_card") in {"0", "1"} else None,
        }

    def get_guide(self, *, session_id: UUID, chat_id: UUID) -> dict | None:
        data = self.redis.hgetall(self._chat_key(session_id=session_id, chat_id=chat_id))
        return self._guide_state(data) if data else None

    def draw_guide(self, *, session_id: UUID, chat_id: UUID, card: int) -> dict | None:
        if isinstance(card, bool) or card not in (0, 1):
            raise ValueError("Escolha uma das duas cartas.")
        return self._update_guide(session_id=session_id, chat_id=chat_id, card=card)

    def set_persona(self, *, session_id: UUID, chat_id: UUID, persona: str) -> dict | None:
        if persona not in {"neutral", "cosmo", "damiao"}:
            raise ValueError("Persona inválida.")
        return self._update_guide(session_id=session_id, chat_id=chat_id, persona=persona)

    def _update_guide(self, *, session_id: UUID, chat_id: UUID, card=None, persona=None) -> dict | None:
        key = self._chat_key(session_id=session_id, chat_id=chat_id)
        # A primeira escolha prevalece, inclusive em retries e abas concorrentes.
        for _ in range(5):
            with self.redis.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    data = pipe.hgetall(key)
                    if not data:
                        return None
                    if card is not None:
                        if data.get("guide_pending") != "1":
                            return self._guide_state(data)
                        first = data["guide_first"]
                        guide = first if card == 0 else {"cosmo": "damiao", "damiao": "cosmo"}[first]
                        changes = {"guide_pending": "0", "drawn_guide": guide, "guide_card": str(card), "persona": guide}
                    else:
                        changes = {"guide_pending": "0", "persona": persona}
                    pipe.multi()
                    pipe.hset(key, mapping=changes)
                    pipe.execute()
                    return self._guide_state({**data, **changes})
                except WatchError:
                    continue
        raise RedisError("Não foi possível salvar a escolha da conversa.")

    def touch(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
    ) -> None:
        conversation = self.get(
            session_id=session_id,
            chat_id=chat_id,
        )

        if conversation is None:
            return

        now = datetime.now(timezone.utc)

        pipe = self.redis.pipeline()
        pipe.hset(
            self._chat_key(
                session_id=session_id,
                chat_id=chat_id,
            ),
            "updated_at",
            now.isoformat(),
        )
        pipe.zadd(
            self._session_chats_key(session_id),
            {str(chat_id): now.timestamp()},
        )
        pipe.execute()

    def rename(
        self,
        *,
        session_id: UUID,
        chat_id: UUID,
        title: str,
    ) -> None:
        self.redis.hset(
            self._chat_key(
                session_id=session_id,
                chat_id=chat_id,
            ),
            "title",
            title.strip() or "Nova conversa",
        )

    @staticmethod
    def _chat_key(
        *,
        session_id: UUID,
        chat_id: UUID,
    ) -> str:
        return (
            f"{REDIS_NAMESPACE}:session:{session_id}:"
            f"chat:{chat_id}:metadata"
        )

    @staticmethod
    def _session_chats_key(
        session_id: UUID,
    ) -> str:
        return (
            f"{REDIS_NAMESPACE}:session:{session_id}:chats"
        )

    @staticmethod
    def _deserialize(data: dict) -> dict:
        return {
            "chat_id": UUID(data["chat_id"]),
            "session_id": data["session_id"],
            "title": data["title"],
            "status": data["status"],
            "created_at": datetime.fromisoformat(
                data["created_at"]
            ),
            "updated_at": datetime.fromisoformat(
                data["updated_at"]
            ),
        }


__all__ = [
    "REDIS_NAMESPACE",
    "RedisChatRepository",
    "conversation_thread_id",
]
