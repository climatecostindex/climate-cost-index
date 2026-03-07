"""FastAPI application for the Climate Cost Index API."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Climate Cost Index API",
    description="CCI v1 — composite household-level metric for climate-linked cost pressure",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# TODO: mount route modules
# from api.routes import scores, components, trends, compare, sensitivity, metadata
# app.include_router(scores.router, prefix="/v1")
# app.include_router(components.router, prefix="/v1")
# app.include_router(trends.router, prefix="/v1")
# app.include_router(compare.router, prefix="/v1")
# app.include_router(sensitivity.router, prefix="/v1")
# app.include_router(metadata.router, prefix="/v1")
