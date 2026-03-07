"""Confidence grade definitions per component."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConfidenceGrade(str, Enum):
    """Data confidence grade for a component measurement."""

    A = "A"  # Direct measurement, county-level, high spatial coverage
    B = "B"  # Proxy or modeled, or sparse spatial coverage
    C = "C"  # Indirect proxy, significant imputation, or limited vintage


@dataclass(frozen=True)
class GradeCriteria:
    grade: ConfidenceGrade
    spatial_coverage: str
    measurement_type: str
    description: str


CONFIDENCE_GRADES: dict[str, GradeCriteria] = {
    "A": GradeCriteria(
        grade=ConfidenceGrade.A,
        spatial_coverage=">=90% of counties with direct observation",
        measurement_type="Direct measurement or official classification",
        description="High-confidence direct data. Example: NOAA station degree days, "
        "FEMA flood zone classification, EIA state energy prices.",
    ),
    "B": GradeCriteria(
        grade=ConfidenceGrade.B,
        spatial_coverage=">=60% of counties, remainder modeled/interpolated",
        measurement_type="Modeled, interpolated, or proxy-based",
        description="Moderate-confidence data requiring spatial interpolation or "
        "indirect measurement. Example: wildfire hazard potential (raster model), "
        "storm severity (damage reports with significant missingness).",
    ),
    "C": GradeCriteria(
        grade=ConfidenceGrade.C,
        spatial_coverage="<60% direct coverage or heavy imputation",
        measurement_type="Indirect proxy or limited availability",
        description="Lower-confidence data with substantial gaps. Example: CDC heat "
        "ED visits (not all states report), county-level energy when only state available.",
    ),
}
