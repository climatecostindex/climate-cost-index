"""Compute drought severity scores from weekly USDM classifications.

Input: Weekly D0-D4 percentages from ingest/usdm_drought.py

Score formula:
  weekly_severity = (D0*1 + D1*2 + D2*3 + D3*4 + D4*5) / 100
  drought_score = sum(weeks in year) weekly_severity

Output columns: fips, year, drought_score, max_severity,
               weeks_in_drought, pct_area_avg
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import HARMONIZED_DIR, RAW_DIR
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEVERITY_WEIGHTS = {"d0": 1, "d1": 2, "d2": 3, "d3": 4, "d4": 5}

MIN_WEEKS_PER_YEAR = 48
PCT_SUM_TOLERANCE = 5.0
PCT_SUM_MAX_DEVIATION = 20.0

# File paths
DROUGHT_COMBINED_PATH = RAW_DIR / "usdm" / "usdm_all.parquet"
DROUGHT_DIR = RAW_DIR / "usdm"
DROUGHT_PER_YEAR_GLOB = "usdm_*.parquet"

OUTPUT_COLUMNS = ["fips", "year", "drought_score", "max_severity", "weeks_in_drought", "pct_area_avg"]

REQUIRED_COLUMNS = {"fips", "date", "d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "none_pct"}

METADATA_SOURCE = "USDM"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "proxy"

D_COLUMNS = ["d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_drought_scores(
    weekly_drought: pd.DataFrame,
    scoring_year: int,
    trailing_months: int = 12,
) -> pd.DataFrame:
    """Compute severity-area integral over trailing window.

    Args:
        weekly_drought: DataFrame with columns ``fips``, ``date``,
            ``d0_pct``, ``d1_pct``, ``d2_pct``, ``d3_pct``, ``d4_pct``,
            ``none_pct``.
        scoring_year: Calendar year to compute scores for.
        trailing_months: Not used in current implementation (calendar year
            window is used). Kept for API compatibility.

    Returns:
        DataFrame with columns: ``fips``, ``year``, ``drought_score``,
        ``max_severity``, ``weeks_in_drought``, ``pct_area_avg``.
    """
    _validate_columns(weekly_drought)

    if weekly_drought.empty:
        logger.warning("Empty weekly drought data — returning empty result.")
        return _empty_output()

    # Prepare data
    df = _prepare_weekly_data(weekly_drought)

    # Filter to scoring year
    year_df = df[df["year"] == scoring_year].copy()
    if year_df.empty:
        logger.info("No drought data for year %d — returning empty result.", scoring_year)
        return _empty_output()

    # Log week coverage
    n_weeks = year_df.groupby("fips")["date"].nunique()
    low_coverage = n_weeks[n_weeks < MIN_WEEKS_PER_YEAR]
    if len(low_coverage) > 0:
        logger.warning(
            "%d counties have fewer than %d weeks of data for year %d",
            len(low_coverage), MIN_WEEKS_PER_YEAR, scoring_year,
        )

    # Compute scores
    result = _compute_county_year_scores(year_df, scoring_year)

    return result


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    weekly_drought = _load_weekly_drought()

    if weekly_drought.empty:
        logger.warning("No drought data loaded — nothing to compute.")
        return _empty_output()

    # Prepare data once
    df = _prepare_weekly_data(weekly_drought)

    # Determine all complete years
    years = sorted(df["year"].unique())
    logger.info("Processing drought scores for %d years: %d–%d", len(years), years[0], years[-1])

    all_results = []
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

    for yr in years:
        year_result = compute_drought_scores(weekly_drought, scoring_year=yr)
        if year_result.empty:
            continue

        all_results.append(year_result)

        # Save parquet + metadata
        parquet_path = HARMONIZED_DIR / f"drought_scoring_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"drought_scoring_{yr}_metadata.json"

        year_result.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_result))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    if not all_results:
        logger.warning("No drought scoring results to write.")
        return _empty_output()

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        "Drought scoring transform complete: %d years, %d total county-year rows",
        len(all_results), len(combined),
    )

    return combined


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Weekly drought data missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading helpers (for run() only)
# ---------------------------------------------------------------------------
def _load_weekly_drought() -> pd.DataFrame:
    """Load weekly drought data from cached parquet files."""
    if DROUGHT_COMBINED_PATH.exists():
        logger.info("Loading combined drought data from %s", DROUGHT_COMBINED_PATH)
        return pd.read_parquet(DROUGHT_COMBINED_PATH)

    # Fall back to per-year files
    per_year = sorted(DROUGHT_DIR.glob(DROUGHT_PER_YEAR_GLOB))
    # Exclude the combined file and metadata files from glob results
    per_year = [
        p for p in per_year
        if "all" not in p.name and not p.name.endswith("_metadata.json")
    ]

    if not per_year:
        raise FileNotFoundError(
            f"No drought data found at {DROUGHT_COMBINED_PATH} "
            f"or matching {DROUGHT_DIR / DROUGHT_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year drought files (fallback path)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _prepare_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and prepare weekly drought data for scoring."""
    result = df.copy()

    # Normalize FIPS
    initial_count = len(result)
    result["fips"] = result["fips"].apply(lambda x: normalize_fips(x) if pd.notna(x) else None)
    invalid_fips = result["fips"].isna()
    invalid_count = invalid_fips.sum()
    if invalid_count > 0:
        logger.warning("Dropping %d rows with invalid/missing FIPS codes", invalid_count)
        result = result[~invalid_fips].copy()

    # Parse dates
    result["date"] = pd.to_datetime(result["date"])
    result["year"] = result["date"].dt.year

    # Clamp negative percentages to 0
    for col in D_COLUMNS + ["none_pct"]:
        neg_mask = result[col] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            logger.warning("Clamping %d negative values in %s to 0", neg_count, col)
            result.loc[neg_mask, col] = 0.0

    # Check percentage sums
    result["_pct_sum"] = (
        result["d0_pct"] + result["d1_pct"] + result["d2_pct"]
        + result["d3_pct"] + result["d4_pct"] + result["none_pct"]
    )
    deviation = (result["_pct_sum"] - 100.0).abs()

    minor_deviation = (deviation > PCT_SUM_TOLERANCE) & (deviation <= PCT_SUM_MAX_DEVIATION)
    major_deviation = deviation > PCT_SUM_MAX_DEVIATION

    if minor_deviation.sum() > 0:
        logger.warning(
            "%d rows have percentage sums deviating >%.0f%% from 100 (but ≤%.0f%%) — kept",
            minor_deviation.sum(), PCT_SUM_TOLERANCE, PCT_SUM_MAX_DEVIATION,
        )

    if major_deviation.sum() > 0:
        logger.warning(
            "Dropping %d rows with percentage sums deviating >%.0f%% from 100",
            major_deviation.sum(), PCT_SUM_MAX_DEVIATION,
        )
        result = result[~major_deviation].copy()

    result = result.drop(columns=["_pct_sum"])

    rows_dropped = initial_count - len(result)
    if rows_dropped > 0:
        logger.info(
            "Weekly data preparation: %d → %d rows (%d dropped)",
            initial_count, len(result), rows_dropped,
        )

    return result


def _compute_county_year_scores(year_df: pd.DataFrame, scoring_year: int) -> pd.DataFrame:
    """Compute drought scores for all counties in a single year."""
    # Compute weekly severity for each row
    year_df = year_df.copy()
    year_df["weekly_severity"] = sum(
        year_df[f"{level}_pct"] * weight
        for level, weight in SEVERITY_WEIGHTS.items()
    ) / 100.0

    # Drought area per week (100 - none_pct)
    year_df["drought_area_pct"] = 100.0 - year_df["none_pct"]

    # Whether any drought existed this week
    year_df["in_drought"] = year_df["drought_area_pct"] > 0

    # Determine max severity level per row
    # Check from D4 down to D0
    year_df["_max_d"] = -1
    for level, weight in sorted(SEVERITY_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
        col = f"{level}_pct"
        mask = (year_df[col] > 0) & (year_df["_max_d"] == -1)
        year_df.loc[mask, "_max_d"] = weight - 1  # D0=0, D1=1, ..., D4=4

    # Aggregate per county
    grouped = year_df.groupby("fips").agg(
        drought_score=("weekly_severity", "sum"),
        max_severity=("_max_d", "max"),
        weeks_in_drought=("in_drought", "sum"),
        pct_area_avg=("drought_area_pct", "mean"),
    ).reset_index()

    grouped["year"] = scoring_year

    # Cast types
    grouped["max_severity"] = grouped["max_severity"].astype(int)
    grouped["weeks_in_drought"] = grouped["weeks_in_drought"].astype(int)

    # Enforce output schema
    grouped = grouped[OUTPUT_COLUMNS].copy()

    return grouped


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "drought_score": float,
        "max_severity": int,
        "weeks_in_drought": int,
        "pct_area_avg": float,
    })


def _write_metadata(path: Path, year: int) -> None:
    """Write metadata JSON sidecar alongside the parquet output."""
    meta = {
        "source": METADATA_SOURCE,
        "confidence": METADATA_CONFIDENCE,
        "attribution": METADATA_ATTRIBUTION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "data_vintage": str(year),
        "description": (
            f"County-level annual drought severity scores for {year}. "
            f"Severity-area integral using USDM D0-D4 classifications. "
            f"Weights: D0=1, D1=2, D2=3, D3=4, D4=5. "
            f"Minimum weeks for completeness warning: {MIN_WEEKS_PER_YEAR}."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
