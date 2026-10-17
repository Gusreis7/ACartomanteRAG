"""Contadores agregados por dia UTC, privados no Redis, com retenção limitada."""

from datetime import datetime, timezone
import logging

from redis import RedisError

logger = logging.getLogger("acartomante.usage")
RETENTION = 14 * 24 * 3600


class UsageStats:
    def __init__(self, redis):
        self.redis = redis

    @staticmethod
    def day():
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _write(self, fields, session_id=None):
        key = f"acartomante:usage:{self.day()}"
        try:
            pipe = self.redis.pipeline(transaction=False)
            for field, value in fields.items():
                if field == "estimated_cost_usd":
                    pipe.hincrbyfloat(key, field, float(value))
                else:
                    pipe.hincrby(key, field, int(value))
            pipe.expire(key, RETENTION)
            if session_id is not None:
                pipe.pfadd(key + ":sessions", str(session_id))
                pipe.expire(key + ":sessions", RETENTION)
            pipe.execute()
        except RedisError:
            logger.warning("usage_storage_unavailable")

    def record_http(self, method, route, status, duration_ms, session_id=None):
        self._write(
            {
                "http_requests": 1,
                f"http_{status // 100}xx": 1,
                "http_429": int(status == 429),
                "http_503": int(status == 503),
                "page_views": int(
                    method == "GET" and route in {"/", "/reading", "/chat/{chat_id}"}
                ),
                "http_duration_ms": max(0, round(duration_ms)),
            },
            session_id,
        )
        logger.info(
            "http_request method=%s route=%s status=%s duration_ms=%.1f",
            method,
            route,
            status,
            duration_ms,
        )

    def record_chat(self, decision, *usage):
        fields = {
            "chat_answered" if decision == "allow" else "chat_refused": 1,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        for item in usage:
            fields["llm_input_tokens"] += int(item.get("input_tokens", 0))
            fields["llm_output_tokens"] += int(item.get("output_tokens", 0))
            fields["estimated_cost_usd"] += float(
                item.get("estimated_cost_usd", 0) or 0
            )
        self._write(fields)

    def summary(self, day):
        key = f"acartomante:usage:{day}"
        fields = self.redis.hgetall(key)
        return {
            "day_utc": day,
            "sessions_estimated": self.redis.pfcount(key + ":sessions"),
            **{
                name: float(value) if name == "estimated_cost_usd" else int(value)
                for name, value in fields.items()
            },
        }
