from __future__ import annotations

from threading import Lock
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from src.services.recommendations import (
    RecommendationUnavailableError,
    WorkRecommendationService,
)


router = APIRouter(prefix="/api/v1", tags=["recommendations"])
_service_lock = Lock()


class WorkRecommendationResponse(BaseModel):
    title: str
    author: str | None = None
    category: str | None = None
    source_url: str


def _service(request: Request) -> WorkRecommendationService:
    service = getattr(request.app.state, "work_recommendation_service", None)
    if service is not None:
        return service

    with _service_lock:
        service = getattr(request.app.state, "work_recommendation_service", None)
        if service is None:
            service = WorkRecommendationService.from_runtime(
                config=request.app.state.config,
                rag=request.app.state.rag,
            )
            request.app.state.work_recommendation_service = service

    return service


@router.get(
    "/recommendations/works",
    response_model=list[WorkRecommendationResponse],
)
def recommend_works(
    request: Request,
    exclude: Annotated[list[str] | None, Query()] = None,
) -> list[WorkRecommendationResponse]:
    try:
        works = _service(request).recommend(
            count=3,
            exclude_titles=exclude,
        )
    except RecommendationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return [WorkRecommendationResponse(**item) for item in works]
