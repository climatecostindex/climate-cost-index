"""Pydantic response models for the CCI API."""

from __future__ import annotations

from pydantic import BaseModel


class ComponentScore(BaseModel):
    component_id: str
    name: str
    percentile: float
    weight: float
    penalty: float
    acceleration: float
    contribution: float
    confidence: str
    attribution: str


class CCIScoreResponse(BaseModel):
    fips: str
    county_name: str
    state: str
    cci_score: float
    cci_dollar: float | None = None
    cci_strain: float | None = None
    data_vintage: str
    methodology_version: str
    components: list[ComponentScore] | None = None


class CCINationalResponse(BaseModel):
    year: int
    cci_national: float
    methodology_version: str


class MetadataResponse(BaseModel):
    methodology_version: str
    data_vintage: str
    scoring_universe_size: int
    k_constant: float
    climate_normal_baseline: str
    components: list[dict]
