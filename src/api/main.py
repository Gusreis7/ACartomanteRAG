from __future__ import annotations

import logging

from fastapi import FastAPI

from src.api.lifespan import lifespan
from src.api.routes import router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="ACartomante API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
