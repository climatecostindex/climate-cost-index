"""GET /v1/trends/{fips} — absolute/relative/national trend data."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/trends/{fips}")
async def get_trends(fips: str, type: str = "absolute", years: int = 5):
    raise NotImplementedError
