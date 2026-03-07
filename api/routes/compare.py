"""GET /v1/compare — side-by-side county comparison."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/compare")
async def compare(fips: str):
    """Compare 2-4 counties. fips is comma-separated, e.g. '12086,04013,17031'."""
    raise NotImplementedError
