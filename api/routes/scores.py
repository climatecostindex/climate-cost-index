"""GET /v1/scores/{fips} and GET /v1/scores — CCI score endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/scores/{fips}")
async def get_score(fips: str, include_components: bool = False):
    raise NotImplementedError


@router.get("/scores")
async def list_scores(state: str | None = None, min_score: float | None = None,
                      max_score: float | None = None, limit: int = 100):
    raise NotImplementedError
