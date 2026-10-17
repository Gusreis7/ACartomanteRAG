from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.generation.personas import PersonaName


def plain_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Texto vazio.")
    if any(
        unicodedata.category(char) in {"Cc", "Cs"} and char not in "\t\n\r"
        for char in value
    ):
        raise ValueError("Texto contém caracteres binários ou de controle.")
    return value


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str | None = Field(default=None, max_length=120)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value):
        return plain_text(value) if value is not None else None


class TutorialCompleteRequest(BaseModel):
    """Tutorial sem dados: aceita corpo ausente ou um objeto JSON vazio."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ConversationResponse(BaseModel):
    chat_id: UUID
    session_id: UUID
    title: str
    status: str
    created_at: datetime
    updated_at: datetime


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    message: str = Field(min_length=1, max_length=4000)
    persona: PersonaName = "neutral"

    @field_validator("message")
    @classmethod
    def validate_message(cls, value):
        return plain_text(value)


class GuideStateResponse(BaseModel):
    draw_required: bool
    persona: PersonaName
    drawn_guide: Literal["cosmo", "damiao"] | None = None
    selected_card: Literal[0, 1] | None = None


class GuideDrawRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    card: int = Field(ge=0, le=1, strict=True)


class PersonaUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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
    evidences: list[EvidenceResponse] = Field(default_factory=list)


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
