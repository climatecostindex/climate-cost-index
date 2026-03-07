"""Compute extreme heat day counts from daily station observations.

Input:
    - Daily TMAX from ingest/noaa_ncei.py:
      ``data/raw/noaa_ncei/noaa_ncei_observations_all.parquet`` (combined) or
      ``data/raw/noaa_ncei/noaa_ncei_{year}.parquet`` (per-year)
    - Station-to-county mapping from transform/station_to_county.py:
      ``data/harmonized/station_to_county.parquet``

Steps:
    1. Load daily observations, filter NaN TMAX and quality-flagged rows
    2. Load station-to-county mapping
    3. Compute daily threshold exceedances (95°F / 100°F) per station
    4. Aggregate to annual counts per station (≥335 valid days required)
    5. Aggregate station-level results to county-level means
    6. Save per-year parquet + metadata JSON sidecar

Output columns: fips, year, days_above_95f, days_above_100f

Confidence: A
Attribution: proxy
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import HARMONIZED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Thresholds in Fahrenheit (per SSRN paper, for documentation)
THRESHOLD_95F = 95.0
THRESHOLD_100F = 100.0

# Thresholds converted to Celsius (for computation)
THRESHOLD_95F_C = (THRESHOLD_95F - 32) * 5 / 9   # 35.0°C
THRESHOLD_100F_C = (THRESHOLD_100F - 32) * 5 / 9  # ≈ 37.778°C

MIN_DAYS_PER_YEAR = 335

# File paths
OBS_COMBINED_PATH = RAW_DIR / "noaa_ncei" / "noaa_ncei_observations_all.parquet"
OBS_DIR = RAW_DIR / "noaa_ncei"
OBS_PER_YEAR_GLOB = "noaa_ncei_*.parquet"
STATION_COUNTY_PATH = HARMONIZED_DIR / "station_to_county.parquet"

OUTPUT_COLUMNS = ["fips", "year", "days_above_95f", "days_above_100f"]

METADATA_SOURCE = "NOAA_GHCN_DAILY"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "proxy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_extreme_heat_days(
    daily_obs: pd.DataFrame,
    station_county_map: pd.DataFrame,
) -> pd.DataFrame:
    """Count days per county where TMAX exceeds 95°F and 100°F thresholds.

    Args:
        daily_obs: DataFrame with columns ``station_id``, ``date``, ``tmax``,
            ``q_flag_tmax``.  Temperatures in °C.
        station_county_map: DataFrame with columns ``station_id``, ``fips``.
            From ``station_to_county.py``.

    Returns:
        DataFrame with columns: ``fips``, ``year``, ``days_above_95f``,
        ``days_above_100f``.
    """
    # --- Validate inputs ---------------------------------------------------
    _validate_daily_obs_columns(daily_obs)
    _validate_station_county_columns(station_county_map)

    # --- Handle empty inputs -----------------------------------------------
    if daily_obs.empty:
        logger.warning("Empty daily observations — returning empty result.")
        return _empty_output()

    if station_county_map.empty:
        logger.warning("Empty station-to-county mapping — returning empty result.")
        return _empty_output()

    # --- Filter and prepare daily observations -----------------------------
    filtered_obs = _filter_daily_obs(daily_obs)
    if filtered_obs.empty:
        logger.warning("No valid observations after filtering — returning empty result.")
        return _empty_output()

    # --- Compute daily threshold exceedances -------------------------------
    daily_exc = _compute_daily_exceedances(filtered_obs)

    # --- Aggregate to annual counts per station ----------------------------
    station_annual = _aggregate_station_annual(daily_exc)
    if station_annual.empty:
        logger.warning("No station-years passed completeness threshold — returning empty result.")
        return _empty_output()

    # --- Filter station-to-county mapping ----------------------------------
    valid_map = _filter_station_county_map(station_county_map)

    # --- Aggregate to county level -----------------------------------------
    county_results = _aggregate_to_county(station_annual, valid_map)

    return county_results


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    daily_obs = _load_daily_observations()
    station_county_map = _load_station_county_map()

    result = compute_extreme_heat_days(daily_obs, station_county_map)

    if result.empty:
        logger.warning("No extreme-heat results to write.")
        return result

    # Save per-year outputs
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    years = sorted(result["year"].unique())
    for yr in years:
        year_df = result[result["year"] == yr].copy()
        parquet_path = HARMONIZED_DIR / f"extreme_heat_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"extreme_heat_{yr}_metadata.json"

        year_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_df))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    logger.info(
        "Extreme-heat transform complete: %d years, %d total county-year rows",
        len(years),
        len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_daily_obs_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from daily observations."""
    required = {"station_id", "date", "tmax", "q_flag_tmax"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Daily observations missing columns: {sorted(missing)}")


def _validate_station_county_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from station-county map."""
    required = {"station_id", "fips"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Station-county mapping missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading helpers (for run() only)
# ---------------------------------------------------------------------------
def _load_daily_observations() -> pd.DataFrame:
    """Load daily station observations from cached parquet files."""
    if OBS_COMBINED_PATH.exists():
        logger.info("Loading combined observations from %s", OBS_COMBINED_PATH)
        return pd.read_parquet(OBS_COMBINED_PATH)

    # Fall back to per-year files
    per_year = sorted(OBS_DIR.glob(OBS_PER_YEAR_GLOB))
    # Exclude non-observation files (normals, stations, combined)
    per_year = [
        p for p in per_year
        if "normals" not in p.name
        and "stations" not in p.name
        and "observations_all" not in p.name
    ]
    if not per_year:
        raise FileNotFoundError(
            f"No daily observation files found at {OBS_COMBINED_PATH} "
            f"or matching {OBS_DIR / OBS_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year observation files (fallback)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


def _load_station_county_map() -> pd.DataFrame:
    """Load station-to-county mapping from harmonized output."""
    if not STATION_COUNTY_PATH.exists():
        raise FileNotFoundError(
            f"Station-to-county mapping not found at {STATION_COUNTY_PATH}. "
            "Run station_to_county.py (Module 2.1) first."
        )
    logger.info("Loading station-to-county mapping from %s", STATION_COUNTY_PATH)
    return pd.read_parquet(STATION_COUNTY_PATH)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _filter_daily_obs(daily_obs: pd.DataFrame) -> pd.DataFrame:
    """Filter daily observations: drop NaN TMAX and quality-flagged rows."""
    df = daily_obs.copy()
    initial_count = len(df)

    # Drop rows with NaN tmax
    nan_mask = df["tmax"].isna()
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.info("Dropping %d rows with NaN tmax", nan_count)
    df = df[~nan_mask]

    # Drop quality-flagged observations
    # Fill NaN flags with empty string for consistent comparison
    q_tmax = df["q_flag_tmax"].fillna("")
    flag_mask = q_tmax != ""
    flag_count = flag_mask.sum()
    if flag_count > 0:
        logger.info("Dropping %d rows with non-empty quality flags", flag_count)
    df = df[~flag_mask]

    total_dropped = initial_count - len(df)
    logger.info(
        "Daily obs filtering: %d → %d rows (%d dropped)",
        initial_count, len(df), total_dropped,
    )

    # Extract year from date
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Keep only needed columns
    df = df[["station_id", "year", "tmax"]].copy()

    return df


def _compute_daily_exceedances(filtered_obs: pd.DataFrame) -> pd.DataFrame:
    """Compute daily threshold exceedance flags from filtered observations."""
    df = filtered_obs.copy()
    df["exceeds_95f"] = (df["tmax"] > THRESHOLD_95F_C).astype(int)
    df["exceeds_100f"] = (df["tmax"] > THRESHOLD_100F_C).astype(int)
    return df


def _aggregate_station_annual(daily_exc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily exceedances to annual counts per station.

    Applies the MIN_DAYS_PER_YEAR completeness threshold.
    """
    grouped = daily_exc.groupby(["station_id", "year"]).agg(
        days_above_95f=("exceeds_95f", "sum"),
        days_above_100f=("exceeds_100f", "sum"),
        n_days=("tmax", "count"),
    ).reset_index()

    # Apply completeness threshold
    before_count = len(grouped)
    incomplete_mask = grouped["n_days"] < MIN_DAYS_PER_YEAR
    incomplete_count = incomplete_mask.sum()

    if incomplete_count > 0:
        sample_ids = grouped.loc[incomplete_mask, "station_id"].head(5).tolist()
        logger.warning(
            "Dropping %d station-years below %d-day completeness threshold "
            "(sample station IDs: %s)",
            incomplete_count, MIN_DAYS_PER_YEAR, sample_ids,
        )

    result = grouped[~incomplete_mask].drop(columns=["n_days"]).copy()
    logger.info(
        "Station-year aggregation: %d total, %d passed completeness (%d dropped)",
        before_count, len(result), incomplete_count,
    )

    return result


def _filter_station_county_map(station_county_map: pd.DataFrame) -> pd.DataFrame:
    """Filter station-to-county mapping: drop placeholder rows with NaN station_id."""
    df = station_county_map[["station_id", "fips"]].copy()
    df = df[df["station_id"].notna()]
    return df


def _aggregate_to_county(
    station_data: pd.DataFrame,
    station_county_map: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate station-level results to county-level means.

    Joins station data with station-to-county mapping, then averages
    across all stations in each county per year.
    """
    # Join station results with county mapping
    joined = station_data.merge(station_county_map, on="station_id", how="inner")

    # Log stations that have no county mapping
    unmapped_stations = set(station_data["station_id"].unique()) - set(station_county_map["station_id"].unique())
    if unmapped_stations:
        logger.warning(
            "%d stations in observations have no county mapping (excluded from output)",
            len(unmapped_stations),
        )

    if joined.empty:
        logger.warning("No station data matched to any county — returning empty result.")
        return _empty_output()

    # Group by (fips, year) and compute means
    county_results = joined.groupby(["fips", "year"]).agg(
        days_above_95f=("days_above_95f", "mean"),
        days_above_100f=("days_above_100f", "mean"),
    ).reset_index()

    # Enforce output schema
    county_results = county_results[OUTPUT_COLUMNS].copy()

    logger.info(
        "County-level aggregation: %d county-year rows across %d counties",
        len(county_results), county_results["fips"].nunique(),
    )

    return county_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "days_above_95f": float,
        "days_above_100f": float,
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
            f"County-level annual extreme heat day counts for {year}. "
            f"Thresholds: {THRESHOLD_95F}°F ({THRESHOLD_95F_C:.1f}°C) and "
            f"{THRESHOLD_100F}°F ({THRESHOLD_100F_C:.3f}°C). "
            f"Completeness threshold: {MIN_DAYS_PER_YEAR} days/station-year."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
