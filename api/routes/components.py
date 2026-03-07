"""GET /v1/components/{fips} — per-component detail."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/components/{fips}")
async def get_components(fips: str):
    raise NotImplementedError
