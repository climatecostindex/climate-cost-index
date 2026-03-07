"""Compute heating and cooling degree days from daily station observations.

Input:
    - Daily TMAX/TMIN from ingest/noaa_ncei.py:
      ``data/raw/noaa_ncei/noaa_ncei_observations_all.parquet`` (combined) or
      ``data/raw/noaa_ncei/noaa_ncei_{year}.parquet`` (per-year)
    - 1991-2020 climate normals from ingest/noaa_ncei.py:
      ``data/raw/noaa_ncei/noaa_ncei_normals.parquet``
    - Station-to-county mapping from transform/station_to_county.py:
      ``data/harmonized/station_to_county.parquet``

Steps:
    1. Load daily observations, filter quality flags and NaN temps
    2. Load station-to-county mapping
    3. Load 1991-2020 climate normals
    4. Compute daily HDD/CDD per station (base 65°F = 18.333°C)
    5. Aggregate to annual totals per station (≥335 valid days required)
    6. Compute station-level normal annual HDD/CDD from monthly normals
    7. Compute station-level anomalies (observed - normal)
    8. Aggregate station-level results to county-level means
    9. Save per-year parquet + metadata JSON sidecar

Output columns: fips, year, hdd_annual, cdd_annual, hdd_anomaly, cdd_anomaly

Confidence: A
Attribution: proxy
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import HARMONIZED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_TEMP_F = 65.0
BASE_TEMP_C = (BASE_TEMP_F - 32) * 5 / 9  # ≈ 18.333°C

MIN_DAYS_PER_YEAR = 335
MIN_NORMAL_MONTHS = 12

# Standard month lengths (non-leap year for 30-year climatological average)
DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

# File paths
OBS_COMBINED_PATH = RAW_DIR / "noaa_ncei" / "noaa_ncei_observations_all.parquet"
OBS_DIR = RAW_DIR / "noaa_ncei"
OBS_PER_YEAR_GLOB = "noaa_ncei_*.parquet"
NORMALS_PATH = RAW_DIR / "noaa_ncei" / "noaa_ncei_normals.parquet"
STATION_COUNTY_PATH = HARMONIZED_DIR / "station_to_county.parquet"

OUTPUT_COLUMNS = ["fips", "year", "hdd_annual", "cdd_annual", "hdd_anomaly", "cdd_anomaly"]

METADATA_SOURCE = "NOAA_GHCN_DAILY"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "proxy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_degree_days(
    daily_obs: pd.DataFrame,
    station_county_map: pd.DataFrame,
    normals: pd.DataFrame,
) -> pd.DataFrame:
    """Compute county-level annual HDD/CDD and anomalies vs 1991-2020 normals.

    Args:
        daily_obs: DataFrame with columns ``station_id``, ``date``, ``tmax``,
            ``tmin``, ``q_flag_tmax``, ``q_flag_tmin``.  Temperatures in °C.
        station_county_map: DataFrame with columns ``station_id``, ``fips``.
            From ``station_to_county.py``.
        normals: DataFrame with columns ``station_id``, ``month``,
            ``normal_tmax``, ``normal_tmin``.  Temperatures in °C.

    Returns:
        DataFrame with columns: ``fips``, ``year``, ``hdd_annual``,
        ``cdd_annual``, ``hdd_anomaly``, ``cdd_anomaly``.
    """
    # --- Validate inputs ---------------------------------------------------
    _validate_daily_obs_columns(daily_obs)
    _validate_station_county_columns(station_county_map)
    _validate_normals_columns(normals)

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

    # --- Compute daily degree days -----------------------------------------
    daily_dd = _compute_daily_degree_days(filtered_obs)

    # --- Aggregate to annual totals per station ----------------------------
    station_annual = _aggregate_station_annual(daily_dd)
    if station_annual.empty:
        logger.warning("No station-years passed completeness threshold — returning empty result.")
        return _empty_output()

    # --- Compute station-level normal annual HDD/CDD ----------------------
    normal_annual = _compute_normal_annual(normals)

    # --- Compute station-level anomalies -----------------------------------
    station_with_anomalies = _compute_station_anomalies(station_annual, normal_annual)

    # --- Filter station-to-county mapping ----------------------------------
    valid_map = _filter_station_county_map(station_county_map)

    # --- Aggregate to county level -----------------------------------------
    county_results = _aggregate_to_county(station_with_anomalies, valid_map)

    return county_results


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    daily_obs = _load_daily_observations()
    station_county_map = _load_station_county_map()
    normals = _load_normals()

    result = compute_degree_days(daily_obs, station_county_map, normals)

    if result.empty:
        logger.warning("No degree-day results to write.")
        return result

    # Save per-year outputs
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    years = sorted(result["year"].unique())
    for yr in years:
        year_df = result[result["year"] == yr].copy()
        parquet_path = HARMONIZED_DIR / f"degree_days_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"degree_days_{yr}_metadata.json"

        year_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_df))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    logger.info(
        "Degree-day transform complete: %d years, %d total county-year rows",
        len(years),
        len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_daily_obs_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from daily observations."""
    required = {"station_id", "date", "tmax", "tmin", "q_flag_tmax", "q_flag_tmin"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Daily observations missing columns: {sorted(missing)}")


def _validate_station_county_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from station-county map."""
    required = {"station_id", "fips"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Station-county mapping missing columns: {sorted(missing)}")


def _validate_normals_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from normals."""
    required = {"station_id", "month", "normal_tmax", "normal_tmin"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Normals missing columns: {sorted(missing)}")


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


def _load_normals() -> pd.DataFrame:
    """Load 1991-2020 climate normals from cached parquet."""
    if not NORMALS_PATH.exists():
        raise FileNotFoundError(
            f"Climate normals not found at {NORMALS_PATH}"
        )
    logger.info("Loading climate normals from %s", NORMALS_PATH)
    return pd.read_parquet(NORMALS_PATH)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _filter_daily_obs(daily_obs: pd.DataFrame) -> pd.DataFrame:
    """Filter daily observations: drop NaN temps and quality-flagged rows."""
    df = daily_obs.copy()
    initial_count = len(df)

    # Drop rows with NaN tmax or tmin
    nan_mask = df["tmax"].isna() | df["tmin"].isna()
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.info("Dropping %d rows with NaN tmax or tmin", nan_count)
    df = df[~nan_mask]

    # Drop quality-flagged observations
    # Fill NaN flags with empty string for consistent comparison
    q_tmax = df["q_flag_tmax"].fillna("")
    q_tmin = df["q_flag_tmin"].fillna("")
    flag_mask = (q_tmax != "") | (q_tmin != "")
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
    df = df[["station_id", "year", "tmax", "tmin"]].copy()

    return df


def _compute_daily_degree_days(filtered_obs: pd.DataFrame) -> pd.DataFrame:
    """Compute daily HDD and CDD from filtered observations."""
    df = filtered_obs.copy()
    df["avg_temp"] = (df["tmax"] + df["tmin"]) / 2.0
    df["hdd_daily"] = np.maximum(0.0, BASE_TEMP_C - df["avg_temp"])
    df["cdd_daily"] = np.maximum(0.0, df["avg_temp"] - BASE_TEMP_C)
    return df


def _aggregate_station_annual(daily_dd: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily degree days to annual totals per station.

    Applies the MIN_DAYS_PER_YEAR completeness threshold.
    """
    grouped = daily_dd.groupby(["station_id", "year"]).agg(
        hdd_annual=("hdd_daily", "sum"),
        cdd_annual=("cdd_daily", "sum"),
        n_days=("hdd_daily", "count"),
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


def _compute_normal_annual(normals: pd.DataFrame) -> pd.DataFrame:
    """Compute station-level normal annual HDD/CDD from monthly normals.

    Returns DataFrame with columns: station_id, normal_hdd_annual, normal_cdd_annual.
    Only stations with all 12 months of normals are included.
    """
    if normals.empty:
        logger.info("No normals data provided — anomaly computation will be skipped.")
        return pd.DataFrame(columns=["station_id", "normal_hdd_annual", "normal_cdd_annual"])

    df = normals.copy()

    # Drop rows with NaN normal temps
    nan_mask = df["normal_tmax"].isna() | df["normal_tmin"].isna()
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.info("Dropping %d normals rows with NaN temperatures", nan_count)
    df = df[~nan_mask]

    if df.empty:
        return pd.DataFrame(columns=["station_id", "normal_hdd_annual", "normal_cdd_annual"])

    # Count months per station — require all 12
    month_counts = df.groupby("station_id")["month"].nunique()
    complete_stations = month_counts[month_counts >= MIN_NORMAL_MONTHS].index
    incomplete_count = len(month_counts) - len(complete_stations)

    if incomplete_count > 0:
        logger.info(
            "Excluding %d stations from anomaly computation: incomplete normals (<12 months)",
            incomplete_count,
        )

    df = df[df["station_id"].isin(complete_stations)].copy()

    if df.empty:
        return pd.DataFrame(columns=["station_id", "normal_hdd_annual", "normal_cdd_annual"])

    # Compute monthly degree days from normals
    df["normal_avg"] = (df["normal_tmax"] + df["normal_tmin"]) / 2.0
    df["days"] = df["month"].map(DAYS_IN_MONTH)
    df["normal_hdd_month"] = np.maximum(0.0, BASE_TEMP_C - df["normal_avg"]) * df["days"]
    df["normal_cdd_month"] = np.maximum(0.0, df["normal_avg"] - BASE_TEMP_C) * df["days"]

    # Sum to annual per station
    result = df.groupby("station_id").agg(
        normal_hdd_annual=("normal_hdd_month", "sum"),
        normal_cdd_annual=("normal_cdd_month", "sum"),
    ).reset_index()

    logger.info(
        "Computed normal annual HDD/CDD for %d stations (out of %d with normals data)",
        len(result), len(month_counts),
    )

    return result


def _compute_station_anomalies(
    station_annual: pd.DataFrame,
    normal_annual: pd.DataFrame,
) -> pd.DataFrame:
    """Compute station-level anomalies: observed - normal.

    Stations without normals get NaN anomalies but keep their observed values.
    """
    if normal_annual.empty:
        result = station_annual.copy()
        result["hdd_anomaly"] = np.nan
        result["cdd_anomaly"] = np.nan
        logger.info("No normals available — all anomalies set to NaN.")
        return result

    # Left join: all station-years keep their observed values
    result = station_annual.merge(normal_annual, on="station_id", how="left")

    # Compute anomalies where normals are available
    has_normals = result["normal_hdd_annual"].notna()
    result["hdd_anomaly"] = np.where(
        has_normals,
        result["hdd_annual"] - result["normal_hdd_annual"],
        np.nan,
    )
    result["cdd_anomaly"] = np.where(
        has_normals,
        result["cdd_annual"] - result["normal_cdd_annual"],
        np.nan,
    )

    no_normals_count = (~has_normals).sum()
    if no_normals_count > 0:
        logger.info(
            "%d station-years have no normals — anomalies set to NaN",
            no_normals_count,
        )

    # Drop normal columns — no longer needed
    result = result.drop(columns=["normal_hdd_annual", "normal_cdd_annual"])

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
        hdd_annual=("hdd_annual", "mean"),
        cdd_annual=("cdd_annual", "mean"),
        hdd_anomaly=("hdd_anomaly", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
        cdd_anomaly=("cdd_anomaly", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
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
        "hdd_annual": float,
        "cdd_annual": float,
        "hdd_anomaly": float,
        "cdd_anomaly": float,
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
            f"County-level annual heating/cooling degree days and anomalies "
            f"vs 1991-2020 normals for {year}. Base temperature: "
            f"{BASE_TEMP_F}°F ({BASE_TEMP_C:.3f}°C). "
            f"Completeness threshold: {MIN_DAYS_PER_YEAR} days/station-year."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
