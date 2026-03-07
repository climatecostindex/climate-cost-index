"""Compute county-level air quality metrics from monitor readings and HMS smoke data.

Input: Daily monitor readings from ingest/epa_airnow.py
       Monitor-to-county mapping from monitor_to_county.py
       HMS smoke plume polygons from ingest/noaa_hms.py
       County boundaries from Census TIGER

Smoke day identification (both methods use the same 1.5x multiplier):
- PRIMARY: county intersects HMS plume AND PM2.5 > 1.5x rolling 30-day median
- FALLBACK: PM2.5 spike detection (daily > 1.5x rolling 30-day median)
  used only when HMS data is unavailable for a given date.

Output columns: fips, year, pm25_annual_avg, aqi_unhealthy_days,
               aqi_very_unhealthy_days, smoke_days, smoke_day_method
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config.settings import HARMONIZED_DIR, RAW_DIR
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AQI_UNHEALTHY_THRESHOLD = 100
AQI_VERY_UNHEALTHY_THRESHOLD = 150
SMOKE_PM25_MULTIPLIER = 1.5
SMOKE_BASELINE_WINDOW_DAYS = 30

# EPA AQI breakpoints for PM2.5 24-hour average (ug/m3).
# Source: https://www.airnow.gov/aqi/aqi-basics/
# Each tuple: (pm25_lo, pm25_hi, aqi_lo, aqi_hi)
PM25_AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

# File paths
READINGS_COMBINED_PATH = RAW_DIR / "epa_airnow" / "epa_aqs_readings_all.parquet"
READINGS_DIR = RAW_DIR / "epa_airnow"
READINGS_PER_YEAR_GLOB = "epa_aqs_readings_*.parquet"

HMS_COMBINED_PATH = RAW_DIR / "noaa_hms" / "noaa_hms_all.parquet"
HMS_DIR = RAW_DIR / "noaa_hms"
HMS_PER_YEAR_GLOB = "noaa_hms_*.parquet"

COUNTY_BOUNDARY_DIR = RAW_DIR / "census_blocks"
COUNTY_BOUNDARY_GLOB = "cb_*_us_county_500k.zip"

MONITOR_COUNTY_PATH = HARMONIZED_DIR / "monitor_to_county.parquet"

CRS_NAD83 = "EPSG:4269"

OUTPUT_COLUMNS = [
    "fips", "year", "pm25_annual_avg", "aqi_unhealthy_days",
    "aqi_very_unhealthy_days", "smoke_days", "smoke_day_method",
]

METADATA_SOURCE = "EPA_AQS"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "proxy"

MIN_REPRESENTATIVE_DAYS = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_air_quality_scores(
    daily_readings: pd.DataFrame,
    monitor_county_map: pd.DataFrame,
    hms_plumes: gpd.GeoDataFrame | None = None,
    county_boundaries: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Compute county-level annual AQI metrics and smoke day counts.

    Args:
        daily_readings: DataFrame with columns ``monitor_id``, ``date``,
            ``pm25_value``, ``aqi_value``.  One row per monitor per day.
        monitor_county_map: DataFrame with columns ``monitor_id``, ``fips``.
            From ``monitor_to_county.py``.
        hms_plumes: GeoDataFrame with columns ``date``, ``geometry``,
            ``density``.  HMS smoke plume polygons.  None if unavailable.
        county_boundaries: GeoDataFrame with columns ``fips``, ``geometry``.
            County boundary polygons for HMS overlay.  None if unavailable.

    Returns:
        DataFrame with columns: ``fips``, ``year``, ``pm25_annual_avg``,
        ``aqi_unhealthy_days``, ``aqi_very_unhealthy_days``, ``smoke_days``,
        ``smoke_day_method``.
    """
    # --- Validate inputs ---------------------------------------------------
    _validate_readings_columns(daily_readings)
    _validate_monitor_map_columns(monitor_county_map)

    # --- Handle empty inputs -----------------------------------------------
    if daily_readings.empty:
        logger.warning("Empty daily readings — returning empty result.")
        return _empty_output()

    if monitor_county_map.empty:
        logger.warning("Empty monitor-to-county mapping — returning empty result.")
        return _empty_output()

    # --- Filter and clean readings -----------------------------------------
    readings = _clean_readings(daily_readings)
    if readings.empty:
        logger.warning("No valid readings after cleaning — returning empty result.")
        return _empty_output()

    # --- Join readings to counties -----------------------------------------
    county_readings = _join_readings_to_counties(readings, monitor_county_map)
    if county_readings.empty:
        logger.warning("No readings matched to any county — returning empty result.")
        return _empty_output()

    # --- Compute county-level daily aggregates -----------------------------
    county_daily = _compute_county_daily(county_readings)

    # --- Compute annual PM2.5 and AQI metrics ------------------------------
    annual_metrics = _compute_annual_metrics(county_daily)

    # --- Compute smoke days ------------------------------------------------
    smoke_days = _compute_smoke_days(
        county_daily, hms_plumes, county_boundaries,
    )

    # --- Merge annual metrics + smoke days ---------------------------------
    result = annual_metrics.merge(smoke_days, on=["fips", "year"], how="left")
    result["smoke_days"] = result["smoke_days"].fillna(0).astype(int)
    result["smoke_day_method"] = result["smoke_day_method"].fillna("spike_detection")

    # --- Enforce output schema ---------------------------------------------
    result = result[OUTPUT_COLUMNS].copy()

    # --- Log coverage stats ------------------------------------------------
    _log_coverage(result, county_daily)

    return result


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    daily_readings = _load_daily_readings()
    monitor_county_map = _load_monitor_county_map()
    hms_plumes = _load_hms_plumes()
    county_boundaries = _load_county_boundaries(hms_plumes is not None)

    result = compute_air_quality_scores(
        daily_readings, monitor_county_map, hms_plumes, county_boundaries,
    )

    if result.empty:
        logger.warning("No air quality results to write.")
        return result

    # Save per-year outputs
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    years = sorted(result["year"].unique())
    for yr in years:
        year_df = result[result["year"] == yr].copy()
        parquet_path = HARMONIZED_DIR / f"air_quality_scoring_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"air_quality_scoring_{yr}_metadata.json"

        year_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_df))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    logger.info(
        "Air quality scoring transform complete: %d years, %d total county-year rows",
        len(years),
        len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_readings_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from daily readings."""
    required = {"monitor_id", "date", "pm25_value", "aqi_value"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Daily readings missing columns: {sorted(missing)}")


def _validate_monitor_map_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from monitor-county map."""
    required = {"monitor_id", "fips"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Monitor-county mapping missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading helpers (for run() only)
# ---------------------------------------------------------------------------
def _load_daily_readings() -> pd.DataFrame:
    """Load daily monitor readings from cached parquet files."""
    if READINGS_COMBINED_PATH.exists():
        logger.info("Loading combined daily readings from %s", READINGS_COMBINED_PATH)
        return pd.read_parquet(READINGS_COMBINED_PATH)

    # Fall back to per-year files
    per_year = sorted(READINGS_DIR.glob(READINGS_PER_YEAR_GLOB))
    per_year = [p for p in per_year if "readings_all" not in p.name]
    if not per_year:
        raise FileNotFoundError(
            f"No daily reading files found at {READINGS_COMBINED_PATH} "
            f"or matching {READINGS_DIR / READINGS_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year daily reading files (fallback)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


def _load_monitor_county_map() -> pd.DataFrame:
    """Load monitor-to-county mapping from harmonized output."""
    if not MONITOR_COUNTY_PATH.exists():
        raise FileNotFoundError(
            f"Monitor-to-county mapping not found at {MONITOR_COUNTY_PATH}. "
            "Run monitor_to_county.py (Module 2.2) first."
        )
    logger.info("Loading monitor-to-county mapping from %s", MONITOR_COUNTY_PATH)
    return pd.read_parquet(MONITOR_COUNTY_PATH)


def _load_hms_plumes() -> gpd.GeoDataFrame | None:
    """Load HMS smoke plume data. Returns None if unavailable."""
    if HMS_COMBINED_PATH.exists():
        logger.info("Loading combined HMS smoke plumes from %s", HMS_COMBINED_PATH)
        return gpd.read_parquet(HMS_COMBINED_PATH)

    per_year = sorted(HMS_DIR.glob(HMS_PER_YEAR_GLOB))
    per_year = [p for p in per_year if "hms_all" not in p.name]
    if not per_year:
        logger.warning(
            "No HMS smoke plume files found — using spike detection for ALL smoke day identification."
        )
        return None

    logger.info("Loading %d per-year HMS smoke plume files (fallback)", len(per_year))
    gdfs = [gpd.read_parquet(p) for p in per_year]
    return pd.concat(gdfs, ignore_index=True)


def _load_county_boundaries(hms_available: bool) -> gpd.GeoDataFrame | None:
    """Load county boundaries for HMS overlay. Only needed if HMS data exists."""
    if not hms_available:
        return None

    matches = sorted(COUNTY_BOUNDARY_DIR.glob(COUNTY_BOUNDARY_GLOB))
    if not matches:
        logger.warning(
            "No county boundary file found for HMS overlay — "
            "falling back to spike detection for all smoke days."
        )
        return None

    county_path = matches[-1]  # most recent year
    logger.info("Loading county boundaries from %s", county_path)
    gdf = gpd.read_file(county_path)

    if gdf.empty:
        return None

    gdf["fips"] = gdf["GEOID"].apply(normalize_fips)

    if gdf.crs is None or gdf.crs.to_epsg() != 4269:
        logger.info("Reprojecting county boundaries to %s", CRS_NAD83)
        gdf = gdf.to_crs(CRS_NAD83)

    return gdf[["fips", "geometry"]]


# ---------------------------------------------------------------------------
# Data cleaning helpers
# ---------------------------------------------------------------------------
def _clean_readings(daily_readings: pd.DataFrame) -> pd.DataFrame:
    """Clean daily readings: drop invalid values, extract year."""
    df = daily_readings.copy()
    initial_count = len(df)

    # Drop rows where BOTH pm25 and aqi are NaN
    both_nan = df["pm25_value"].isna() & df["aqi_value"].isna()
    both_nan_count = both_nan.sum()
    if both_nan_count > 0:
        logger.info("Dropping %d rows with NaN for both pm25 and aqi", both_nan_count)
    df = df[~both_nan]

    # Set negative PM2.5 values to NaN (instrument error)
    neg_pm25 = df["pm25_value"] < 0
    neg_count = neg_pm25.sum()
    if neg_count > 0:
        logger.warning("Setting %d negative PM2.5 values to NaN (instrument error)", neg_count)
        df.loc[neg_pm25, "pm25_value"] = np.nan

    # Set AQI values outside 0-500 to NaN (invalid)
    invalid_aqi = (df["aqi_value"] < 0) | (df["aqi_value"] > 500)
    invalid_aqi_count = invalid_aqi.sum()
    if invalid_aqi_count > 0:
        logger.warning("Setting %d AQI values outside 0-500 range to NaN", invalid_aqi_count)
        df.loc[invalid_aqi, "aqi_value"] = np.nan

    # After cleaning, re-check: drop rows where BOTH are now NaN
    both_nan2 = df["pm25_value"].isna() & df["aqi_value"].isna()
    df = df[~both_nan2]

    # Extract year
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Keep needed columns
    df = df[["monitor_id", "date", "year", "pm25_value", "aqi_value"]].copy()

    logger.info(
        "Readings cleaning: %d → %d rows",
        initial_count, len(df),
    )

    return df


# ---------------------------------------------------------------------------
# County assignment
# ---------------------------------------------------------------------------
def _join_readings_to_counties(
    readings: pd.DataFrame,
    monitor_county_map: pd.DataFrame,
) -> pd.DataFrame:
    """Join readings to counties via monitor-to-county mapping."""
    # Filter out NaN monitor_ids in the mapping
    valid_map = monitor_county_map[monitor_county_map["monitor_id"].notna()].copy()
    valid_map = valid_map[["monitor_id", "fips"]].copy()

    # Normalize FIPS
    valid_map["fips"] = valid_map["fips"].apply(normalize_fips)

    # Ensure monitor_id types match
    readings = readings.copy()
    readings["monitor_id"] = readings["monitor_id"].astype(str)
    valid_map["monitor_id"] = valid_map["monitor_id"].astype(str)

    # Inner join: only keep readings for monitors with county assignments
    joined = readings.merge(valid_map, on="monitor_id", how="inner")

    unmapped_monitors = set(readings["monitor_id"].unique()) - set(valid_map["monitor_id"].unique())
    if unmapped_monitors:
        logger.warning(
            "%d monitors in readings have no county mapping (excluded)",
            len(unmapped_monitors),
        )

    return joined


# ---------------------------------------------------------------------------
# PM2.5 → AQI conversion (EPA breakpoint table)
# ---------------------------------------------------------------------------
def _pm25_to_aqi(pm25: float) -> float:
    """Convert a 24-hour PM2.5 concentration to AQI using EPA breakpoints.

    Returns NaN for negative or NaN inputs.
    """
    if np.isnan(pm25) or pm25 < 0:
        return np.nan
    for bp_lo, bp_hi, aqi_lo, aqi_hi in PM25_AQI_BREAKPOINTS:
        if pm25 <= bp_hi:
            return round(
                ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            )
    return 500.0


# ---------------------------------------------------------------------------
# County-level daily aggregation
# ---------------------------------------------------------------------------
def _compute_county_daily(county_readings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monitor readings to county-day level.

    PM2.5: simple mean across monitors.
    AQI: maximum across monitors (worst reading).  When no monitor in a
    county reports AQI for a given day, AQI is computed from the county
    daily mean PM2.5 using EPA breakpoint tables.
    """
    grouped = county_readings.groupby(["fips", "date", "year"]).agg(
        daily_pm25=("pm25_value", "mean"),
        daily_aqi=("aqi_value", "max"),
    ).reset_index()

    # Fill missing AQI from PM2.5 using EPA breakpoints
    missing_aqi = grouped["daily_aqi"].isna() & grouped["daily_pm25"].notna()
    n_filled = missing_aqi.sum()
    if n_filled > 0:
        grouped.loc[missing_aqi, "daily_aqi"] = (
            grouped.loc[missing_aqi, "daily_pm25"].apply(_pm25_to_aqi)
        )
        logger.info(
            "Filled %d county-days with AQI computed from PM2.5 (EPA breakpoints)",
            n_filled,
        )

    return grouped


# ---------------------------------------------------------------------------
# Annual metrics
# ---------------------------------------------------------------------------
def _compute_annual_metrics(county_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute annual PM2.5 average and AQI exceedance day counts."""

    def _agg_func(g: pd.DataFrame) -> pd.Series:
        pm25_avg = g["daily_pm25"].mean()
        unhealthy = (g["daily_aqi"] > AQI_UNHEALTHY_THRESHOLD).sum()
        very_unhealthy = (g["daily_aqi"] > AQI_VERY_UNHEALTHY_THRESHOLD).sum()
        return pd.Series({
            "pm25_annual_avg": pm25_avg,
            "aqi_unhealthy_days": int(unhealthy),
            "aqi_very_unhealthy_days": int(very_unhealthy),
        })

    result = county_daily.groupby(["fips", "year"]).apply(
        _agg_func, include_groups=False,
    ).reset_index()

    result["aqi_unhealthy_days"] = result["aqi_unhealthy_days"].astype(int)
    result["aqi_very_unhealthy_days"] = result["aqi_very_unhealthy_days"].astype(int)

    return result


# ---------------------------------------------------------------------------
# Smoke day computation
# ---------------------------------------------------------------------------
def _compute_smoke_days(
    county_daily: pd.DataFrame,
    hms_plumes: gpd.GeoDataFrame | None,
    county_boundaries: gpd.GeoDataFrame | None,
) -> pd.DataFrame:
    """Identify smoke days using HMS primary method + spike detection fallback.

    Returns DataFrame with columns: fips, year, smoke_days, smoke_day_method.
    """
    # Determine which dates have HMS coverage
    hms_available = (
        hms_plumes is not None
        and not hms_plumes.empty
        and county_boundaries is not None
        and not county_boundaries.empty
    )

    if not hms_available:
        if hms_plumes is not None and not hms_plumes.empty:
            logger.warning(
                "HMS data available but no county boundaries — "
                "falling back to spike detection for all smoke days."
            )
        else:
            logger.warning(
                "No HMS data — using spike detection for all smoke day identification."
            )
        return _compute_smoke_days_spike_only(county_daily)

    # HMS is available: determine which dates have HMS coverage
    hms_dates = set(pd.to_datetime(hms_plumes["date"]).dt.normalize().unique())
    all_dates = set(county_daily["date"].dt.normalize().unique())
    dates_without_hms = all_dates - hms_dates

    if dates_without_hms:
        n_without = len(dates_without_hms)
        logger.info(
            "%d dates without HMS coverage — using spike detection for those dates",
            n_without,
        )

    # Compute rolling baseline on the FULL time series (before splitting)
    county_daily = county_daily.copy()
    county_daily["_date_norm"] = county_daily["date"].dt.normalize()
    county_daily_sorted = county_daily.sort_values(["fips", "date"])
    county_daily_sorted["pm25_baseline"] = (
        county_daily_sorted.groupby("fips")["daily_pm25"]
        .transform(lambda x: x.rolling(
            window=SMOKE_BASELINE_WINDOW_DAYS, min_periods=1,
        ).median())
    )

    # Split into HMS-covered and non-HMS-covered dates
    hms_mask = county_daily_sorted["_date_norm"].isin(hms_dates)
    daily_hms = county_daily_sorted[hms_mask].copy()
    daily_no_hms = county_daily_sorted[~hms_mask].copy()

    # PRIMARY: HMS + PM2.5
    hms_smoke = _identify_hms_smoke_days(daily_hms, hms_plumes, county_boundaries)

    # FALLBACK: spike detection for non-HMS dates
    spike_smoke = _identify_spike_smoke_days(daily_no_hms)

    # Combine smoke day flags
    all_smoke = pd.concat([hms_smoke, spike_smoke], ignore_index=True)

    if all_smoke.empty:
        # Return zero smoke days for all county-years
        fips_years = county_daily[["fips", "year"]].drop_duplicates()
        fips_years["smoke_days"] = 0
        fips_years["smoke_day_method"] = "hms"
        return fips_years[["fips", "year", "smoke_days", "smoke_day_method"]]

    # Aggregate to county-year
    return _aggregate_smoke_to_county_year(all_smoke, county_daily)


def _compute_smoke_days_spike_only(county_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute smoke days using spike detection only (no HMS data)."""
    spike_smoke = _identify_spike_smoke_days(county_daily)

    if spike_smoke.empty:
        fips_years = county_daily[["fips", "year"]].drop_duplicates()
        fips_years["smoke_days"] = 0
        fips_years["smoke_day_method"] = "spike_detection"
        return fips_years[["fips", "year", "smoke_days", "smoke_day_method"]]

    # Aggregate
    annual = spike_smoke.groupby(["fips", "year"]).agg(
        smoke_days=("is_smoke_day", "sum"),
    ).reset_index()
    annual["smoke_days"] = annual["smoke_days"].astype(int)
    annual["smoke_day_method"] = "spike_detection"

    # Ensure all county-years present
    fips_years = county_daily[["fips", "year"]].drop_duplicates()
    result = fips_years.merge(annual, on=["fips", "year"], how="left")
    result["smoke_days"] = result["smoke_days"].fillna(0).astype(int)
    result["smoke_day_method"] = result["smoke_day_method"].fillna("spike_detection")

    return result[["fips", "year", "smoke_days", "smoke_day_method"]]


def _identify_hms_smoke_days(
    daily_hms: pd.DataFrame,
    hms_plumes: gpd.GeoDataFrame,
    county_boundaries: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Identify smoke days using HMS primary method.

    A day is a smoke day if:
    1. An HMS plume intersects the county on that date, AND
    2. The county's daily PM2.5 exceeds 1.5x its rolling 30-day median baseline.

    The 1.5x multiplier is the same threshold used in spike detection. HMS plume
    intersection provides spatial evidence of smoke presence; the PM2.5 threshold
    confirms meaningful surface-level air quality degradation.

    Returns DataFrame with columns: fips, date, year, is_smoke_day, method.
    """
    if daily_hms.empty:
        return pd.DataFrame(columns=["fips", "date", "year", "is_smoke_day", "method"])

    # Use precomputed pm25_baseline (rolling 30-day median from full time series)
    daily_sorted = daily_hms.sort_values(["fips", "date"]).copy()

    # Prepare HMS plumes: fix invalid geometries, normalize dates
    hms_plumes = hms_plumes.copy()
    hms_plumes["date"] = pd.to_datetime(hms_plumes["date"]).dt.normalize()

    # Fix invalid geometries
    from shapely.validation import make_valid
    invalid_count = (~hms_plumes.geometry.is_valid).sum()
    if invalid_count > 0:
        logger.info("Fixing %d invalid HMS plume geometries with make_valid()", invalid_count)
        hms_plumes["geometry"] = hms_plumes.geometry.apply(
            lambda g: make_valid(g) if g is not None and not g.is_valid else g
        )

    # Filter HMS plumes to only dates present in our data
    hms_dates_in_data = set(daily_sorted["_date_norm"].unique())
    hms_filtered = hms_plumes[hms_plumes["date"].isin(hms_dates_in_data)]

    if hms_filtered.empty:
        return pd.DataFrame(columns=["fips", "date", "year", "is_smoke_day", "method"])

    # Ensure CRS alignment for spatial join
    if county_boundaries.crs != hms_filtered.crs:
        logger.info("Reprojecting county boundaries to match HMS CRS")
        county_boundaries = county_boundaries.to_crs(hms_filtered.crs)

    # Spatial join: find (plume_date, county_fips) pairs where plumes intersect counties
    logger.info(
        "Running spatial join: %d HMS plumes × %d counties",
        len(hms_filtered), len(county_boundaries),
    )
    joined = gpd.sjoin(
        hms_filtered[["date", "geometry"]],
        county_boundaries[["fips", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    # Build set of (fips, date) tuples where plumes intersect
    plume_intersections = set(
        zip(joined["fips"], joined["date"])
    )
    logger.info(
        "Found %d unique (county, date) plume intersections",
        len(plume_intersections),
    )

    # Mark smoke days
    daily_sorted["_plume_intersects"] = [
        (fips, pd.Timestamp(d)) in plume_intersections
        for fips, d in zip(daily_sorted["fips"], daily_sorted["_date_norm"])
    ]

    daily_sorted["is_smoke_day"] = (
        daily_sorted["_plume_intersects"]
        & daily_sorted["daily_pm25"].notna()
        & (daily_sorted["daily_pm25"] > SMOKE_PM25_MULTIPLIER * daily_sorted["pm25_baseline"])
    )

    result = daily_sorted[["fips", "date", "year", "is_smoke_day"]].copy()
    result["method"] = "hms"

    # Only keep rows that are actually smoke days for efficient aggregation
    return result[result["is_smoke_day"]]


def _identify_spike_smoke_days(daily_no_hms: pd.DataFrame) -> pd.DataFrame:
    """Identify smoke days using PM2.5 spike detection (fallback).

    A day is a smoke day if daily_pm25 > 1.5x rolling 30-day trailing median.
    Same threshold as the HMS primary method.

    Returns DataFrame with columns: fips, date, year, is_smoke_day, method.
    """
    if daily_no_hms.empty:
        return pd.DataFrame(columns=["fips", "date", "year", "is_smoke_day", "method"])

    daily_sorted = daily_no_hms.sort_values(["fips", "date"]).copy()

    # Compute rolling baseline if not already present (spike-only path)
    if "pm25_baseline" not in daily_sorted.columns:
        daily_sorted["pm25_baseline"] = (
            daily_sorted.groupby("fips")["daily_pm25"]
            .transform(lambda x: x.rolling(
                window=SMOKE_BASELINE_WINDOW_DAYS, min_periods=1,
            ).median())
        )

    daily_sorted["is_smoke_day"] = (
        daily_sorted["daily_pm25"].notna()
        & daily_sorted["pm25_baseline"].notna()
        & (daily_sorted["daily_pm25"] > SMOKE_PM25_MULTIPLIER * daily_sorted["pm25_baseline"])
    )

    result = daily_sorted[["fips", "date", "year", "is_smoke_day"]].copy()
    result["method"] = "spike_detection"

    return result[result["is_smoke_day"]]


def _aggregate_smoke_to_county_year(
    smoke_flags: pd.DataFrame,
    county_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate smoke day flags to county-year with method attribution."""
    # Count smoke days per (fips, year)
    smoke_counts = smoke_flags.groupby(["fips", "year"]).agg(
        smoke_days=("is_smoke_day", "sum"),
    ).reset_index()
    smoke_counts["smoke_days"] = smoke_counts["smoke_days"].astype(int)

    # Determine predominant method per (fips, year)
    method_counts = smoke_flags.groupby(["fips", "year", "method"]).size().reset_index(name="count")
    method_pivot = method_counts.pivot_table(
        index=["fips", "year"], columns="method", values="count", fill_value=0,
    ).reset_index()

    def _get_method(row: pd.Series) -> str:
        hms_count = row.get("hms", 0)
        spike_count = row.get("spike_detection", 0)
        total = hms_count + spike_count
        if total == 0:
            return "hms"
        if hms_count > spike_count:
            return "hms"
        if spike_count > hms_count:
            return "spike_detection"
        return "mixed"

    method_pivot["smoke_day_method"] = method_pivot.apply(_get_method, axis=1)
    method_result = method_pivot[["fips", "year", "smoke_day_method"]]

    # Merge counts + method
    result = smoke_counts.merge(method_result, on=["fips", "year"], how="left")
    result["smoke_day_method"] = result["smoke_day_method"].fillna("hms")

    # Ensure all county-years present (zero smoke days for those without)
    fips_years = county_daily[["fips", "year"]].drop_duplicates()
    result = fips_years.merge(result, on=["fips", "year"], how="left")
    result["smoke_days"] = result["smoke_days"].fillna(0).astype(int)
    result["smoke_day_method"] = result["smoke_day_method"].fillna("hms")

    return result[["fips", "year", "smoke_days", "smoke_day_method"]]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "pm25_annual_avg": float,
        "aqi_unhealthy_days": int,
        "aqi_very_unhealthy_days": int,
        "smoke_days": int,
        "smoke_day_method": str,
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
            f"County-level annual air quality metrics for {year}. "
            f"Includes PM2.5 annual average, AQI exceedance days "
            f"(unhealthy >{AQI_UNHEALTHY_THRESHOLD}, very unhealthy "
            f">{AQI_VERY_UNHEALTHY_THRESHOLD}), and wildfire smoke days "
            f"(HMS primary + spike detection fallback)."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _log_coverage(result: pd.DataFrame, county_daily: pd.DataFrame) -> None:
    """Log coverage statistics."""
    if result.empty:
        return

    n_counties = result["fips"].nunique()
    n_years = result["year"].nunique()
    logger.info(
        "Air quality scoring: %d county-year rows across %d counties, %d years",
        len(result), n_counties, n_years,
    )

    # Log county-years with low reading counts
    reading_counts = county_daily.groupby(["fips", "year"]).size().reset_index(name="n_days")
    low_coverage = reading_counts[reading_counts["n_days"] < MIN_REPRESENTATIVE_DAYS]
    if not low_coverage.empty:
        logger.info(
            "%d county-years have fewer than %d valid reading days (potentially unrepresentative)",
            len(low_coverage), MIN_REPRESENTATIVE_DAYS,
        )
