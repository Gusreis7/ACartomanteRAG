from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.lifespan import lifespan
from src.api.routes import router as api_router
from src.api.views import WEB_ROOT, router as view_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="ACartomante API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/assets",
    StaticFiles(directory=WEB_ROOT / "assets"),
    name="assets",
)
app.include_router(view_router)
app.include_router(api_router)
