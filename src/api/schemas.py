from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


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
    persona: Literal[
        "neutral",
        "cosmo",
        "damiao",
    ] = "neutral"


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
    evidences: list[EvidenceResponse]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    dependencies: dict[str, str]
