"""Limites HTTP anteriores ao parser JSON e ao processamento do chat."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from ipaddress import ip_address, ip_network

from redis import RedisError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse

from src.config.config_manager import ApiConfig

logger = logging.getLogger("acartomante.security")
CLIENT_IP_HEADER = "x-acartomante-client-ip"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'none'; font-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
}


def trusted_proxy_ips(value: str) -> frozenset[str]:
    """Somente endereços exatos; nunca aceitar qualquer proxy da Internet."""
    return frozenset(
        str(ip_address(item.strip())) for item in value.split(",") if item.strip()
    )


def client_identity(scope, headers: Headers, trusted: frozenset[str]) -> str:
    peer = str(ip_address(scope["client"][0]))
    address = ip_address(peer)
    if peer in trusted:
        values = headers.getlist(CLIENT_IP_HEADER)
        if len(values) != 1:
            raise ValueError("Proxy sem endereço de origem válido.")
        address = ip_address(values[0])
    if address.version == 6:
        # Impede contornar o limite apenas alternando endereços do mesmo /64.
        identity = str(ip_network(f"{address}/64", strict=False))
    else:
        identity = str(address)
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


class RedisRateLimiter:
    """Janelas compartilhadas e atômicas; o cookie não faz parte da identidade."""

    SCRIPT = """
        local retry = 0
        for i, key in ipairs(KEYS) do
            local count = tonumber(redis.call('GET', key) or '0')
            if count >= tonumber(ARGV[i * 2 - 1]) then
                retry = math.max(retry, redis.call('TTL', key), 1)
            end
        end
        if retry > 0 then return retry end
        for i, key in ipairs(KEYS) do
            local count = redis.call('INCR', key)
            if count == 1 then redis.call('EXPIRE', key, ARGV[i * 2]) end
        end
        return 0
    """

    def __init__(self, redis):
        self.redis = redis

    def check(self, identity: str, operation: str, config: ApiConfig) -> int:
        limits = [
            ("global:requests", config.global_requests_per_minute, 60),
            (f"ip:{identity}:requests", config.requests_per_minute, 60),
        ]
        if operation == "message":
            limits.append((f"ip:{identity}:messages", config.messages_per_minute, 60))
        elif operation == "create":
            limits.extend(
                [
                    (f"ip:{identity}:create", config.conversations_per_minute, 60),
                    (f"ip:{identity}:create-day", config.conversations_per_day, 86400),
                    ("global:create-day", config.global_conversations_per_day, 86400),
                ]
            )
        keys = [f"acartomante:http-limits:{name}" for name, _, _ in limits]
        arguments = [value for _, limit, ttl in limits for value in (limit, ttl)]
        return int(self.redis.eval(self.SCRIPT, len(keys), *keys, *arguments))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Campo JSON duplicado.")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("Constante JSON inválida.")


class APISecurityMiddleware:
    def __init__(self, app, *, trusted_proxies: frozenset[str] = frozenset()):
        self.app = app
        self.trusted_proxies = trusted_proxies
        self.defaults = ApiConfig()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        is_api = path == "/api" or path.startswith("/api/")

        async def secured_send(message):
            if message["type"] == "http.response.start":
                outgoing = MutableHeaders(scope=message)
                outgoing.update(SECURITY_HEADERS)
                if is_api:
                    outgoing["Cache-Control"] = "no-store"
            await send(message)

        async def reject(status, detail, *, retry=None):
            response_headers = {"Retry-After": str(retry)} if retry else None
            await JSONResponse(
                {"detail": detail}, status_code=status, headers=response_headers
            )(scope, receive, secured_send)

        if not is_api:
            await self.app(scope, receive, secured_send)
            return

        headers = Headers(scope=scope)
        state = scope["app"].state
        config = getattr(state, "api_security_config", self.defaults)
        if len(scope.get("query_string", b"")) > 4096:
            await reject(414, "Consulta muito longa.")
            return
        method = scope["method"]
        if method not in {"GET", "HEAD", "POST", "PATCH"}:
            await reject(405, "Método não permitido.")
            return

        # Sem CORS e sem confiança em X-Forwarded-* fornecido pelo visitante.
        # Origin ausente é permitido para CLI; navegadores recebem as duas checagens.
        origins = headers.getlist("origin")
        expected_scheme = (
            "https" if getattr(state, "secure_cookies", False) else scope["scheme"]
        )
        expected_origin = f"{expected_scheme}://{headers.get('host', '')}"
        if headers.get("sec-fetch-site") in {"cross-site", "same-site"} or (
            origins and (len(origins) != 1 or origins[0] != expected_origin)
        ):
            await reject(403, "Origem da requisição não permitida.")
            return

        lengths = headers.getlist("content-length")
        if lengths and (
            len(lengths) != 1
            or len(lengths[0]) > 20
            or not lengths[0].isascii()
            or not lengths[0].isdecimal()
        ):
            await reject(400, "Content-Length inválido.")
            return
        if lengths and int(lengths[0]) > config.max_body_bytes:
            await reject(413, "Requisição excede o limite de tamanho.")
            return
        if headers.get("content-encoding", "identity").lower() != "identity":
            await reject(415, "Envie JSON UTF-8 sem compressão ou anexos.")
            return
        writes_body = method in {"POST", "PATCH"}
        content_types = headers.getlist("content-type")
        if writes_body and content_types:
            parts = (
                content_types[0].lower().split(";") if len(content_types) == 1 else []
            )
            if (
                not parts
                or parts[0].strip() != "application/json"
                or any(
                    part.strip().replace('"', "")
                    not in {"charset=utf-8", "charset=utf8"}
                    for part in parts[1:]
                )
            ):
                await reject(415, "Somente application/json com texto UTF-8 é aceito.")
                return

        operation = "read"
        if method == "POST":
            normalized = path.rstrip("/")
            if normalized == "/api/v1/conversations":
                operation = "create"
            elif normalized.startswith(
                "/api/v1/conversations/"
            ) and normalized.endswith("/messages"):
                operation = "message"
        try:
            identity = client_identity(scope, headers, self.trusted_proxies)
        except (ValueError, TypeError):
            await reject(400, "Endereço de origem inválido.")
            return
        limiter = getattr(state, "api_rate_limiter", None)
        if limiter is None:
            await reject(
                503, "Controle de acesso temporariamente indisponível.", retry=5
            )
            return
        try:
            retry = await run_in_threadpool(limiter.check, identity, operation, config)
        except RedisError:
            logger.warning("http_rate_limit_unavailable")
            await reject(
                503, "Controle de acesso temporariamente indisponível.", retry=5
            )
            return
        if retry:
            await reject(
                429,
                "Limite de requisições atingido. Tente novamente mais tarde.",
                retry=retry,
            )
            return

        body = bytearray()
        try:
            async with asyncio.timeout(config.body_timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > config.max_body_bytes:
                        await reject(413, "Requisição excede o limite de tamanho.")
                        return
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await reject(408, "Tempo de envio da requisição excedido.")
            return
        if lengths and int(lengths[0]) != len(body):
            await reject(400, "Tamanho da requisição inconsistente.")
            return
        if writes_body and body:
            if not content_types:
                await reject(415, "Somente application/json com texto UTF-8 é aceito.")
                return
            try:
                payload = json.loads(
                    body.decode("utf-8"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_invalid_constant,
                )
                if not isinstance(payload, dict):
                    raise ValueError("Envie um objeto JSON.")
            except (ValueError, UnicodeError, RecursionError):
                await reject(
                    400,
                    "JSON UTF-8 inválido; envie somente os campos de texto esperados.",
                )
                return
        elif body:
            await reject(400, "Esta operação não aceita corpo de requisição.")
            return

        slots = (
            getattr(state, "http_chat_slots", None) if operation == "message" else None
        )
        if operation == "message" and (
            slots is None or not slots.acquire(blocking=False)
        ):
            await reject(
                503, "Consultas em andamento. Tente novamente em instantes.", retry=5
            )
            return

        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        try:
            await self.app(scope, replay, secured_send)
        finally:
            if slots is not None:
                slots.release()
