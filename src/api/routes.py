from __future__ import annotations

import logging

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from redis import RedisError

from src.api.dependencies import ChatServiceDep, ConversationServiceDep, SessionId
from src.api.markdown import render_answer
from src.api.schemas import (
    ChatResponse,
    ConversationCreateRequest,
    ConversationHistoryResponse,
    ConversationResponse,
    HealthResponse,
    MessageRequest,
    ReadyResponse,
)
from src.storage.qdrant_manager import check_qdrant_collection_ready
from src.generation.provider import GenerationQuotaError
from src.services.chat import (
    ChatNotFoundError,
    GenerationCapacityError,
    GenerationServiceError,
)


logger = logging.getLogger("acartomante.runtime")

router = APIRouter()


@router.get("/api/v1/session")
def session_status(session_id: SessionId, service: ConversationServiceDep):
    try:
        return {"tutorial_completed": service.tutorial_completed(session_id)}
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Não foi possível consultar o guia. Tente novamente.") from exc


@router.post("/api/v1/session/tutorial")
def complete_tutorial(session_id: SessionId, service: ConversationServiceDep):
    try:
        service.complete_tutorial(session_id)
        return {"tutorial_completed": True}
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Não foi possível salvar o guia. Tente novamente.") from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    checks: dict[str, str] = {}

    try:
        request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"

    try:
        rag = request.app.state.rag
        bucket = rag.config.storage.embedding_indexes_bucket
        if not rag.retrieval.rustfs.bucket_exists(bucket):
            raise RuntimeError("Bucket de índices indisponível.")
        checks["rustfs"] = "ok"
    except Exception:
        checks["rustfs"] = "error"

    try:
        request.app.state.generation_provider.healthcheck()
        checks["generation"] = "ok"
    except Exception:
        checks["generation"] = "error"

    strategy = request.app.state.config.retrieval_strategy
    if strategy in {"dense", "dense_cross_encoder"}:
        try:
            config = request.app.state.config
            retrieval = request.app.state.rag.retrieval

            check_qdrant_collection_ready(
                url=config.environment.qdrant.url,
                api_key=config.environment.qdrant.api_key,
                collection_name=retrieval.resolved.collection,
            )
            checks["qdrant"] = "ok"
        except Exception:
            logger.exception("readiness_qdrant_failed")
            checks["qdrant"] = "error"

    status = "ready" if all(v == "ok" for v in checks.values()) else "not_ready"
    if status != "ready":
        raise HTTPException(
            status_code=503,
            detail={"status": status, "dependencies": checks},
        )

    return ReadyResponse(status=status, dependencies=checks)


@router.post("/api/v1/conversations", response_model=ConversationResponse)
def create_conversation(
    body: ConversationCreateRequest,
    session_id: SessionId,
    service: ConversationServiceDep,
) -> ConversationResponse:
    return ConversationResponse(
        **service.create(session_id=session_id, title=body.title)
    )


@router.get("/api/v1/conversations", response_model=list[ConversationResponse])
def list_conversations(
    session_id: SessionId,
    service: ConversationServiceDep,
) -> list[ConversationResponse]:
    return [
        ConversationResponse(**item)
        for item in service.list_for_session(session_id)
    ]


@router.get(
    "/api/v1/conversations/{chat_id}/messages",
    response_model=ConversationHistoryResponse,
)
def list_messages(
    chat_id: UUID,
    session_id: SessionId,
    service: ChatServiceDep,
) -> ConversationHistoryResponse:
    try:
        result = service.list_messages(session_id=session_id, chat_id=chat_id)
        result["messages"] = [{**item, "content_html": render_answer(item["content"]) if item["role"] == "assistant" else None} for item in result["messages"]]
        return ConversationHistoryResponse(**result)
    except ChatNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except RedisError as exc:
        raise HTTPException(
            status_code=503,
            detail="Conversation storage temporarily unavailable.",
        ) from exc


@router.post(
    "/api/v1/conversations/{chat_id}/messages",
    response_model=ChatResponse,
)
def send_message(
    chat_id: UUID,
    body: MessageRequest,
    session_id: SessionId,
    service: ChatServiceDep,
) -> ChatResponse:
    try:
        result = service.send_message(session_id=session_id, chat_id=chat_id, message=body.message, persona=body.persona)
        return ChatResponse(**result, answer_html=render_answer(result["answer"]))
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GenerationQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (GenerationCapacityError, GenerationServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RedisError as exc:
        raise HTTPException(
            status_code=503,
            detail="Conversation storage temporarily unavailable.",
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
