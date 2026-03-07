"""GET /v1/metadata — methodology version, data vintages, universe info."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/metadata")
async def get_metadata():
    raise NotImplementedError
