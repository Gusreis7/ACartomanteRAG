from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, Request, Response

from src.services.chat import ChatService
from src.services.conversations import ConversationService


SESSION_COOKIE = "acartomante_session"


def get_session_id(request: Request, response: Response) -> UUID:
    raw = request.cookies.get(SESSION_COOKIE)

    if raw:
        try:
            session_id = UUID(raw)
            request.state.session_id = session_id
            return session_id
        except ValueError:
            pass

    session_id = uuid4()
    request.state.session_id = session_id
    response.set_cookie(
        key=SESSION_COOKIE,
        value=str(session_id),
        httponly=True,
        samesite="lax",
        secure=request.app.state.secure_cookies,
        max_age=60 * 60 * 24 * 30,
    )
    return session_id


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_conversation_service(request: Request) -> ConversationService:
    return request.app.state.conversation_service


SessionId = Annotated[UUID, Depends(get_session_id)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ConversationServiceDep = Annotated[
    ConversationService,
    Depends(get_conversation_service),
]
