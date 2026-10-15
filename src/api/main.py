from __future__ import annotations

import logging
import os
from ipaddress import ip_address

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api.lifespan import lifespan
from src.api.reading import router as reading_router
from src.api.recommendations import router as recommendation_router
from src.api.routes import router as api_router
from src.api.views import WEB_ROOT, router as view_router
from src.config.config_manager import DEFAULT_ENV_PATH


load_dotenv(DEFAULT_ENV_PATH, override=False)
bind_ip = os.getenv("API_BIND_IP")
if bind_ip and ip_address(bind_ip).is_unspecified:
    raise ValueError("API_BIND_IP deve ser um IP específico, nunca 0.0.0.0 ou ::.")
allowed_hosts = [
    host.strip()
    for host in os.getenv("API_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",")
    if host.strip()
]
if not allowed_hosts or "*" in allowed_hosts:
    raise ValueError("API_ALLOWED_HOSTS deve listar os hosts permitidos, sem '*'.")

secure_cookie = os.getenv("SESSION_COOKIE_SECURE", "false").strip().lower()
if secure_cookie not in {"true", "false", "1", "0"}:
    raise ValueError("SESSION_COOKIE_SECURE deve ser true ou false.")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="ACartomante API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.state.secure_cookies = secure_cookie in {"true", "1"}
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts,
    www_redirect=False,
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


app.mount(
    "/assets",
    StaticFiles(directory=WEB_ROOT / "assets"),
    name="assets",
)
app.include_router(view_router)
app.include_router(reading_router)
app.include_router(recommendation_router)
app.include_router(api_router)
