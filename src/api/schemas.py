from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from src.generation.personas import PersonaName


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ConversationResponse(BaseModel):
    chat_id: UUID
    session_id: UUID
    title: str
    status: str
    created_at: datetime
    updated_at: datetime


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    persona: PersonaName = "neutral"


class GuideStateResponse(BaseModel):
    draw_required: bool
    persona: PersonaName
    drawn_guide: Literal["cosmo", "damiao"] | None = None
    selected_card: Literal[0, 1] | None = None


class GuideDrawRequest(BaseModel):
    card: int = Field(ge=0, le=1, strict=True)


class PersonaUpdateRequest(BaseModel):
    persona: PersonaName


class EvidenceResponse(BaseModel):
    evidence_id: str
    work: str | None = None
    section: str | None = None
    excerpt: str
    chunk_id: str


class ConversationMessageResponse(BaseModel):
    id: str
    turn_id: UUID
    role: Literal["user", "assistant"]
    content: str
    content_html: str | None = None
    created_at: datetime
    evidences: list[EvidenceResponse] = Field(
        default_factory=list
    )


class ConversationHistoryResponse(BaseModel):
    chat_id: UUID
    messages: list[ConversationMessageResponse]


class ChatResponse(BaseModel):
    chat_id: UUID
    answer: str
    answer_html: str | None = None
    evidences: list[EvidenceResponse]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    dependencies: dict[str, str]
