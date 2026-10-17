from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import FileResponse

WEB_ROOT = Path(__file__).resolve().parent / "web"
router = APIRouter(include_in_schema=False)


@router.get("/")
def home_view() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "index.html",
        media_type="text/html",
    )


@router.get("/chat/{chat_id}")
def chat_view(chat_id: UUID) -> FileResponse:
    # O UUID é validado na borda HTTP. A propriedade da conversa continua
    # sendo responsabilidade dos endpoints /api/v1 e dos services.
    return FileResponse(
        WEB_ROOT / "chat.html",
        media_type="text/html",
    )
