from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter(include_in_schema=False)
WEB_ROOT = Path(__file__).resolve().parent / "web"


@router.get("/reading")
def reading_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "reading.html")
