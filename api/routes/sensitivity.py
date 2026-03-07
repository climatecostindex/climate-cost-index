"""GET /v1/sensitivity/{fips} — sensitivity analysis results."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/sensitivity/{fips}")
async def get_sensitivity(fips: str):
    raise NotImplementedError
