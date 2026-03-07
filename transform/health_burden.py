"""Compute normalized health burden index from heat-related ED visit data.

Input:
    - Heat-related ED visits from ingest/cdc_epht.py:
      ``data/raw/cdc_epht/cdc_epht_all.parquet`` (combined) or
      ``data/raw/cdc_epht/cdc_epht_{year}.parquet`` (per-year)
    - Population from ingest/census_acs.py:
      ``data/raw/census_acs/census_acs_all.parquet`` (combined) or
      ``data/raw/census_acs/census_acs_{year}.parquet`` (per-year)

Steps:
    1. Load heat-related ED visit data, drop invalid records
    2. Load Census ACS population data
    3. Separate county-level and state-level ED data
    4. Compute per-capita rate for county-level data
    5. Disaggregate state-level data to counties (uniform state rate)
    6. Merge, with county-level data taking precedence over state-level
    7. Compute health burden index (= rate in v1)
    8. Save per-year parquet + metadata JSON sidecar

Output columns: fips, year, heat_ed_rate_per_100k, health_burden_index

Confidence: B
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
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RATE_PER = 100_000

# File paths
EPHT_COMBINED_PATH = RAW_DIR / "cdc_epht" / "cdc_epht_all.parquet"
EPHT_DIR = RAW_DIR / "cdc_epht"
EPHT_PER_YEAR_GLOB = "cdc_epht_*.parquet"

ACS_COMBINED_PATH = RAW_DIR / "census_acs" / "census_acs_all.parquet"
ACS_DIR = RAW_DIR / "census_acs"
ACS_PER_YEAR_GLOB = "census_acs_*.parquet"

OUTPUT_COLUMNS = ["fips", "year", "heat_ed_rate_per_100k", "health_burden_index"]

METADATA_SOURCE = "CDC_EPHT"
METADATA_CONFIDENCE = "B"
METADATA_ATTRIBUTION = "proxy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_health_burden(
    ed_visits: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-capita heat-related ED visit rates and burden index.

    Args:
        ed_visits: DataFrame with columns ``fips`` or ``state_fips``, ``year``,
            ``heat_ed_visits``, ``geo_resolution``.  Optionally ``population``.
        population: DataFrame with columns ``fips``, ``year``, ``population``.
            County-level Census ACS data.

    Returns:
        DataFrame with columns: ``fips``, ``year``,
        ``heat_ed_rate_per_100k``, ``health_burden_index``.
    """
    # --- Validate inputs ---------------------------------------------------
    _validate_ed_visits_columns(ed_visits)
    _validate_population_columns(population)

    # --- Handle empty inputs -----------------------------------------------
    if ed_visits.empty:
        logger.warning("Empty ED visits data — returning empty result.")
        return _empty_output()

    if population.empty:
        logger.warning("Empty population data — returning empty result.")
        return _empty_output()

    # --- Clean ED visit data -----------------------------------------------
    ed = _clean_ed_visits(ed_visits)
    if ed.empty:
        logger.warning("No valid ED visit records after cleaning — returning empty result.")
        return _empty_output()

    # --- Prepare population data -------------------------------------------
    pop = _prepare_population(population)

    # --- Separate county-level and state-level data ------------------------
    county_ed, state_ed = _split_by_resolution(ed)

    # --- Compute county-level rates ----------------------------------------
    county_rates = _compute_county_rates(county_ed, pop)

    # --- Disaggregate state-level data to counties -------------------------
    state_rates = _disaggregate_state_to_counties(state_ed, pop)

    # --- Merge: county-level takes precedence ------------------------------
    result = _merge_rates(county_rates, state_rates)

    if result.empty:
        logger.warning("No rates computed — returning empty result.")
        return _empty_output()

    # --- Compute health burden index (= rate in v1) ------------------------
    result["health_burden_index"] = result["heat_ed_rate_per_100k"]

    # --- Enforce output schema ---------------------------------------------
    result = result[OUTPUT_COLUMNS].copy()

    logger.info(
        "Health burden transform: %d county-year rows across %d counties",
        len(result), result["fips"].nunique(),
    )

    return result


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    ed_visits = _load_ed_visits()
    population = _load_population()

    result = compute_health_burden(ed_visits, population)

    if result.empty:
        logger.warning("No health burden results to write.")
        return result

    # Save per-year outputs
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    years = sorted(result["year"].unique())
    for yr in years:
        year_df = result[result["year"] == yr].copy()
        parquet_path = HARMONIZED_DIR / f"health_burden_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"health_burden_{yr}_metadata.json"

        year_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_df))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    logger.info(
        "Health burden transform complete: %d years, %d total county-year rows",
        len(years),
        len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_ed_visits_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from ED visits."""
    required = {"year", "heat_ed_visits", "geo_resolution"}
    # Need either fips or state_fips
    has_fips = "fips" in df.columns
    has_state_fips = "state_fips" in df.columns
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"ED visits missing columns: {sorted(missing)}")
    if not has_fips and not has_state_fips and not df.empty:
        raise ValueError("ED visits must have 'fips' or 'state_fips' column")


def _validate_population_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from population."""
    required = {"fips", "year", "population"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Population data missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading helpers (for run() only)
# ---------------------------------------------------------------------------
def _load_ed_visits() -> pd.DataFrame:
    """Load heat-related ED visit data from cached parquet files."""
    if EPHT_COMBINED_PATH.exists():
        logger.info("Loading combined EPHT data from %s", EPHT_COMBINED_PATH)
        return pd.read_parquet(EPHT_COMBINED_PATH)

    # Fall back to per-year files
    per_year = sorted(EPHT_DIR.glob(EPHT_PER_YEAR_GLOB))
    # Exclude combined file from glob matches
    per_year = [p for p in per_year if "all" not in p.name]
    if not per_year:
        raise FileNotFoundError(
            f"No EPHT data found at {EPHT_COMBINED_PATH} "
            f"or matching {EPHT_DIR / EPHT_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year EPHT files (fallback)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


def _load_population() -> pd.DataFrame:
    """Load Census ACS population data from cached parquet files."""
    if ACS_COMBINED_PATH.exists():
        logger.info("Loading combined Census ACS data from %s", ACS_COMBINED_PATH)
        return pd.read_parquet(ACS_COMBINED_PATH)

    # Fall back to per-year files
    per_year = sorted(ACS_DIR.glob(ACS_PER_YEAR_GLOB))
    per_year = [p for p in per_year if "all" not in p.name]
    if not per_year:
        raise FileNotFoundError(
            f"No Census ACS data found at {ACS_COMBINED_PATH} "
            f"or matching {ACS_DIR / ACS_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year Census ACS files (fallback)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _clean_ed_visits(ed_visits: pd.DataFrame) -> pd.DataFrame:
    """Clean ED visit data: drop NaN/invalid rows, normalize FIPS."""
    df = ed_visits.copy()
    initial_count = len(df)

    # Drop rows where heat_ed_visits is NaN
    nan_mask = df["heat_ed_visits"].isna()
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.info("Dropping %d rows with NaN heat_ed_visits", nan_count)
    df = df[~nan_mask]

    # Drop rows where heat_ed_visits <= 0 (set negative to NaN first, then drop)
    negative_mask = df["heat_ed_visits"] < 0
    negative_count = negative_mask.sum()
    if negative_count > 0:
        logger.warning("Found %d records with negative ED visit counts — dropping", negative_count)
    df = df[~negative_mask]

    zero_mask = df["heat_ed_visits"] == 0
    zero_count = zero_mask.sum()
    if zero_count > 0:
        logger.info("Dropping %d rows with zero heat_ed_visits", zero_count)
    df = df[~zero_mask]

    # Normalize FIPS codes where applicable
    if "fips" in df.columns:
        county_mask = df["geo_resolution"] == "county"
        if county_mask.any():
            df.loc[county_mask, "fips"] = df.loc[county_mask, "fips"].apply(normalize_fips)

    # Normalize state_fips to 2-digit zero-padded
    if "state_fips" in df.columns:
        df["state_fips"] = df["state_fips"].apply(
            lambda x: f"{int(x):02d}" if pd.notna(x) else x
        )

    total_dropped = initial_count - len(df)
    if total_dropped > 0:
        logger.info(
            "ED visit cleaning: %d → %d rows (%d dropped)",
            initial_count, len(df), total_dropped,
        )

    return df


def _prepare_population(population: pd.DataFrame) -> pd.DataFrame:
    """Prepare population data: normalize FIPS, keep relevant columns."""
    df = population[["fips", "year", "population"]].copy()
    df["fips"] = df["fips"].apply(normalize_fips)
    # Derive state FIPS (first 2 digits)
    df["state_fips"] = df["fips"].str[:2]
    return df


def _split_by_resolution(
    ed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ED visit data by geo_resolution tag."""
    county_mask = ed["geo_resolution"] == "county"
    county_ed = ed[county_mask].copy()
    state_ed = ed[~county_mask].copy()

    logger.info(
        "ED data split: %d county-level records, %d state-level records",
        len(county_ed), len(state_ed),
    )

    return county_ed, state_ed


def _compute_county_rates(
    county_ed: pd.DataFrame,
    pop: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-capita rates for county-level ED data.

    Prefers Census ACS population; falls back to EPHT-provided population.
    """
    if county_ed.empty:
        return pd.DataFrame(columns=["fips", "year", "heat_ed_rate_per_100k"])

    df = county_ed[["fips", "year", "heat_ed_visits"]].copy()

    # Check if EPHT provided its own population
    has_epht_pop = "population" in county_ed.columns

    # Join with Census ACS population
    df = df.merge(
        pop[["fips", "year", "population"]],
        on=["fips", "year"],
        how="left",
    )

    # Fall back to EPHT-provided population where Census is unavailable
    if has_epht_pop:
        epht_pop = county_ed[["fips", "year", "population"]].rename(
            columns={"population": "epht_population"},
        )
        df = df.merge(epht_pop, on=["fips", "year"], how="left")

        fallback_mask = df["population"].isna() & df["epht_population"].notna()
        fallback_count = fallback_mask.sum()
        if fallback_count > 0:
            logger.info(
                "Using EPHT-provided population for %d county-year records "
                "(Census ACS unavailable)", fallback_count,
            )
            df.loc[fallback_mask, "population"] = df.loc[fallback_mask, "epht_population"]

        df = df.drop(columns=["epht_population"])

    # Compute rate
    zero_pop_mask = (df["population"] == 0) | df["population"].isna()
    zero_pop_count = zero_pop_mask.sum()
    if zero_pop_count > 0:
        logger.warning(
            "%d county-year records with zero or NaN population — rate set to NaN",
            zero_pop_count,
        )

    df["heat_ed_rate_per_100k"] = np.where(
        zero_pop_mask,
        np.nan,
        (df["heat_ed_visits"] / df["population"]) * RATE_PER,
    )

    # Drop rows with NaN rate
    valid_mask = df["heat_ed_rate_per_100k"].notna()
    df = df[valid_mask].copy()

    return df[["fips", "year", "heat_ed_rate_per_100k"]]


def _disaggregate_state_to_counties(
    state_ed: pd.DataFrame,
    pop: pd.DataFrame,
) -> pd.DataFrame:
    """Disaggregate state-level ED rates to all counties in each state.

    Assigns the uniform state rate to every county in the state.
    """
    if state_ed.empty:
        return pd.DataFrame(columns=["fips", "year", "heat_ed_rate_per_100k"])

    # Determine the state FIPS column
    if "state_fips" in state_ed.columns:
        state_col = "state_fips"
    elif "fips" in state_ed.columns:
        # For state-level records, fips is the 2-digit state FIPS
        state_col = "fips"
    else:
        logger.warning("State-level records have no state identifier — skipping disaggregation")
        return pd.DataFrame(columns=["fips", "year", "heat_ed_rate_per_100k"])

    # Compute state-level populations from Census ACS
    state_pop = pop.groupby(["state_fips", "year"], as_index=False)["population"].sum()
    state_pop = state_pop.rename(columns={"population": "state_population"})

    # Prepare state ED data
    sdf = state_ed[[state_col, "year", "heat_ed_visits"]].copy()
    sdf = sdf.rename(columns={state_col: "state_fips"})
    # Ensure state_fips is 2-digit zero-padded
    sdf["state_fips"] = sdf["state_fips"].apply(lambda x: f"{int(x):02d}")

    # Join with state population
    sdf = sdf.merge(state_pop, on=["state_fips", "year"], how="left")

    # Compute state rate
    zero_mask = (sdf["state_population"] == 0) | sdf["state_population"].isna()
    zero_count = zero_mask.sum()
    if zero_count > 0:
        logger.warning(
            "%d state-year records with zero or NaN state population — rate set to NaN",
            zero_count,
        )

    sdf["state_rate"] = np.where(
        zero_mask,
        np.nan,
        (sdf["heat_ed_visits"] / sdf["state_population"]) * RATE_PER,
    )

    # Drop states with NaN rate
    sdf = sdf[sdf["state_rate"].notna()].copy()

    if sdf.empty:
        logger.info("No valid state-level rates computed.")
        return pd.DataFrame(columns=["fips", "year", "heat_ed_rate_per_100k"])

    # Get all counties per state from population data
    counties = pop[["fips", "state_fips", "year"]].drop_duplicates()

    # Join: assign state rate to each county
    disagg = counties.merge(
        sdf[["state_fips", "year", "state_rate"]],
        on=["state_fips", "year"],
        how="inner",
    )

    disagg = disagg.rename(columns={"state_rate": "heat_ed_rate_per_100k"})

    assigned_count = len(disagg)
    if assigned_count > 0:
        logger.info(
            "Assigned state-level rate to %d county-year records via disaggregation",
            assigned_count,
        )

    return disagg[["fips", "year", "heat_ed_rate_per_100k"]]


def _merge_rates(
    county_rates: pd.DataFrame,
    state_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Merge county-level and state-level rates. County-level takes precedence."""
    if county_rates.empty and state_rates.empty:
        return pd.DataFrame(columns=["fips", "year", "heat_ed_rate_per_100k"])

    if county_rates.empty:
        return state_rates.copy()

    if state_rates.empty:
        return county_rates.copy()

    # Mark which county-years have county-level data
    county_keys = set(zip(county_rates["fips"], county_rates["year"]))

    # Filter state rates: only keep rows where county-level data does NOT exist
    state_mask = [
        (f, y) not in county_keys
        for f, y in zip(state_rates["fips"], state_rates["year"])
    ]
    state_only = state_rates[state_mask].copy()

    override_count = len(state_rates) - len(state_only)
    if override_count > 0:
        logger.debug(
            "%d county-year records: county-level data preferred over state-level",
            override_count,
        )

    result = pd.concat([county_rates, state_only], ignore_index=True)

    # Ensure one row per (fips, year) — deduplicate just in case
    result = result.drop_duplicates(subset=["fips", "year"], keep="first")

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "heat_ed_rate_per_100k": float,
        "health_burden_index": float,
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
            f"County-level heat-related ED visit rates and health burden "
            f"index for {year}. Rate is per {RATE_PER:,} population. "
            f"State-level data disaggregated uniformly to counties."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
