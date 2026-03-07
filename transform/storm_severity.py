"""Compute severity-weighted storm scores per county.

Input:
    - Tiered event records from transform/event_severity_tiers.py:
      ``data/harmonized/event_severity_tiers.parquet``
    - FEMA IA declarations from ingest/fema_ia.py:
      ``data/raw/fema_ia/fema_ia_all.parquet`` or per-year files
    - FEMA Housing Assistance payouts from ingest/fema_ha.py:
      ``data/raw/fema_ha/fema_ha_all.parquet``
    - County boundary reference from ingest/census_blocks.py:
      ``data/raw/census_blocks/county_boundaries_2024.parquet``
    - Housing units from ingest/census_acs.py:
      ``data/raw/census_acs/census_acs_all.parquet`` or per-year files

Steps:
    1. Load tiered event records, extract year from date
    2. Aggregate tiered events to county-year (severity_raw, counts, damage)
    3. Compute pct_missing_damage per county-year
    4. Load FEMA IA declarations, enrich with FEMA HA payout amounts
    5. Compute FEMA-to-NCEI damage ratio
    6. Compute severity reliability flag
    7. Load housing units, join with temporal fallback
    8. Normalize by housing units → storm_severity_score
    9. Save per-year parquet + metadata JSON sidecar

Output columns: fips, year, storm_severity_score, event_count,
               total_damage, severity_reliability_flag,
               pct_missing_damage, fema_ncei_ratio

Confidence: B
Attribution: proxy
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.fips_codes import STATE_FIPS
from config.settings import HARMONIZED_DIR, RAW_DIR
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEMA_NCEI_RATIO_THRESHOLD = 3.0
MISSING_DAMAGE_THRESHOLD_PCT = 50.0
FEMA_NCEI_SENTINEL = 999.0  # sentinel for FEMA > 0 but NCEI = 0

# File paths — tiered events (from Module 2.7)
TIERED_EVENTS_PATH = HARMONIZED_DIR / "event_severity_tiers.parquet"

# File paths — FEMA IA declarations
FEMA_IA_COMBINED_PATH = RAW_DIR / "fema_ia" / "fema_ia_all.parquet"
FEMA_IA_DIR = RAW_DIR / "fema_ia"
FEMA_IA_PER_YEAR_GLOB = "fema_ia_*.parquet"

# File paths — FEMA Housing Assistance payouts
FEMA_HA_PATH = RAW_DIR / "fema_ha" / "fema_ha_all.parquet"

# File paths — County boundaries (for county name → FIPS crosswalk)
COUNTY_BOUNDARIES_PATH = RAW_DIR / "census_blocks" / "county_boundaries_2024.parquet"

# State FIPS → abbreviation (extend with territories for FEMA HA coverage)
_STATE_FIPS_TO_ABBR: dict[str, str] = {
    **STATE_FIPS,
    "60": "AS", "66": "GU", "69": "MP", "72": "PR", "78": "VI",
}
_ABBR_TO_STATE_FIPS: dict[str, str] = {v: k for k, v in _STATE_FIPS_TO_ABBR.items()}

# Regex to strip parenthetical suffixes from FEMA HA county names
_COUNTY_SUFFIX_RE = re.compile(r"\s*\(.*\)\s*$")

# File paths — Census ACS housing units
CENSUS_ACS_COMBINED_PATH = RAW_DIR / "census_acs" / "census_acs_all.parquet"
CENSUS_ACS_DIR = RAW_DIR / "census_acs"
CENSUS_ACS_PER_YEAR_GLOB = "census_acs_*.parquet"

# Output
OUTPUT_COLUMNS = [
    "fips", "year", "storm_severity_score", "event_count",
    "total_damage", "severity_reliability_flag",
    "pct_missing_damage", "fema_ncei_ratio",
]

METADATA_SOURCE = "NCEI_STORM_EVENTS"
METADATA_CONFIDENCE = "B"
METADATA_ATTRIBUTION = "proxy"

# Required input columns
TIERED_EVENT_REQUIRED = {"event_id", "fips", "date", "total_damage", "severity_tier", "tier_weight"}
FEMA_IA_REQUIRED = {"fips", "year", "ia_amount"}
HOUSING_REQUIRED = {"fips", "year", "total_housing_units"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_storm_severity(
    tiered_events: pd.DataFrame,
    fema_ia: pd.DataFrame | None,
    housing_units: pd.DataFrame,
) -> pd.DataFrame:
    """Compute county-level severity-weighted storm scores.

    Args:
        tiered_events: Event-level DataFrame from event_severity_tiers.py
            with columns event_id, fips, date, total_damage, severity_tier,
            tier_weight.
        fema_ia: FEMA IA payouts DataFrame with columns fips, year,
            ia_amount. May be None if FEMA data is unavailable.
        housing_units: Census ACS DataFrame with columns fips, year,
            total_housing_units.

    Returns:
        DataFrame with columns: fips, year, storm_severity_score,
        event_count, total_damage, severity_reliability_flag,
        pct_missing_damage, fema_ncei_ratio.
    """
    _validate_columns(tiered_events, TIERED_EVENT_REQUIRED, "Tiered events")
    _validate_columns(housing_units, HOUSING_REQUIRED, "Housing units")
    if fema_ia is not None and not fema_ia.empty:
        _validate_columns(fema_ia, FEMA_IA_REQUIRED, "FEMA IA")

    if tiered_events.empty:
        logger.warning("Empty tiered events input — returning empty result.")
        return _empty_output()

    if housing_units.empty:
        logger.warning("Empty housing units input — returning empty result.")
        return _empty_output()

    # --- Prepare tiered events -----------------------------------------------
    events = tiered_events.copy()

    # Filter NaN FIPS (should be handled by Module 2.7, but guard here)
    nan_fips = events["fips"].isna()
    if nan_fips.any():
        logger.warning(
            "Dropping %d tiered events with NaN FIPS codes", nan_fips.sum()
        )
        events = events[~nan_fips].copy()

    if events.empty:
        return _empty_output()

    # Extract year from date
    events["year"] = pd.to_datetime(events["date"]).dt.year

    # --- Aggregate to county-year --------------------------------------------
    county_year = _aggregate_county_year(events)

    # --- Compute pct_missing_damage ------------------------------------------
    county_year["pct_missing_damage"] = (
        county_year["unreported_event_count"] / county_year["event_count"] * 100
    )

    # --- FEMA IA cross-validation --------------------------------------------
    county_year = _join_fema_ia(county_year, fema_ia)

    # --- Compute reliability flag --------------------------------------------
    county_year = _compute_reliability_flag(county_year)

    # --- Replace sentinel ratio with NaN -------------------------------------
    # The 999.0 sentinel (FEMA > 0, NCEI = 0) has served its purpose: it
    # triggered the reliability flag above.  Replace with NaN so downstream
    # scoring never treats it as a real numeric ratio.
    sentinel_mask = county_year["fema_ncei_ratio"] == FEMA_NCEI_SENTINEL
    if sentinel_mask.any():
        logger.info(
            "Replacing %d sentinel fema_ncei_ratio values (%.1f) with NaN",
            sentinel_mask.sum(),
            FEMA_NCEI_SENTINEL,
        )
        county_year.loc[sentinel_mask, "fema_ncei_ratio"] = np.nan

    # --- Normalize by housing units ------------------------------------------
    county_year = _normalize_by_housing(county_year, housing_units)

    # --- Log summary ---------------------------------------------------------
    _log_summary(county_year)

    # --- Enforce output schema -----------------------------------------------
    return county_year[OUTPUT_COLUMNS].copy()


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    tiered_events = _load_tiered_events()
    fema_ia = _load_fema_ia()
    housing_units = _load_housing_units()

    result = compute_storm_severity(tiered_events, fema_ia, housing_units)

    if result.empty:
        logger.warning("No storm severity results to write.")
        return result

    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

    years = sorted(result["year"].unique())
    for yr in years:
        year_df = result[result["year"] == yr].copy()
        parquet_path = HARMONIZED_DIR / f"storm_severity_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"storm_severity_{yr}_metadata.json"

        year_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_df))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    logger.info(
        "Storm severity transform complete: %d years, %d county-year rows",
        len(years), len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_columns(
    df: pd.DataFrame, required: set[str], label: str
) -> None:
    """Raise ValueError if required columns are missing."""
    if df is None or df.empty:
        return
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading (for run() only)
# ---------------------------------------------------------------------------
def _load_tiered_events() -> pd.DataFrame:
    """Load tiered event records from harmonized output."""
    if not TIERED_EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Tiered event records not found at {TIERED_EVENTS_PATH}. "
            "Run event_severity_tiers.py (Module 2.7) first."
        )
    logger.info("Loading tiered events from %s", TIERED_EVENTS_PATH)
    return pd.read_parquet(TIERED_EVENTS_PATH)


def _load_fema_ia() -> pd.DataFrame | None:
    """Load FEMA IA declarations and enrich with HA payout amounts.

    Returns None if IA declaration data is unavailable. If HA payout data
    is available, merges dollar amounts into the IA records via a county
    name → FIPS crosswalk.
    """
    # Load IA declarations
    ia: pd.DataFrame | None = None
    if FEMA_IA_COMBINED_PATH.exists():
        logger.info("Loading combined FEMA IA from %s", FEMA_IA_COMBINED_PATH)
        ia = pd.read_parquet(FEMA_IA_COMBINED_PATH)
    else:
        per_year = sorted(FEMA_IA_DIR.glob(FEMA_IA_PER_YEAR_GLOB))
        per_year = [p for p in per_year if "all" not in p.name]
        if per_year:
            logger.info("Loading %d per-year FEMA IA files (fallback)", len(per_year))
            dfs = [pd.read_parquet(p) for p in per_year]
            ia = pd.concat(dfs, ignore_index=True)

    if ia is None:
        logger.warning(
            "No FEMA IA data found at %s or matching %s. "
            "Proceeding without FEMA cross-validation.",
            FEMA_IA_COMBINED_PATH, FEMA_IA_DIR / FEMA_IA_PER_YEAR_GLOB,
        )
        return None

    # Try to enrich with HA payout amounts
    ha = _load_fema_ha()
    if ha is None:
        logger.info("No FEMA HA data — ia_amount will remain NaN.")
        return ia

    ia = _enrich_ia_with_ha(ia, ha)
    return ia


def _load_fema_ha() -> pd.DataFrame | None:
    """Load FEMA Housing Assistance payout data. Returns None if unavailable."""
    if not FEMA_HA_PATH.exists():
        logger.warning("FEMA HA data not found at %s", FEMA_HA_PATH)
        return None
    logger.info("Loading FEMA HA from %s", FEMA_HA_PATH)
    return pd.read_parquet(FEMA_HA_PATH)


def _normalize_county_name(name: str) -> str:
    """Normalize a county name for matching.

    Strips parenthetical suffixes (County, Parish, Borough, etc.),
    lowercases, strips whitespace, normalizes "saint"/"st." variants,
    and removes diacritics (accents) for cross-source matching.
    """
    # Strip parenthetical suffix: "Barbour (County)" → "Barbour"
    name = _COUNTY_SUFFIX_RE.sub("", name)
    name = name.strip().lower()
    # Normalize St./Saint
    name = re.sub(r"^st\.\s*", "saint ", name)
    name = re.sub(r"\bst\.\s", "saint ", name)
    # Remove diacritics: "bayamón" → "bayamon", "añasco" → "anasco"
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name


def _build_county_fips_crosswalk() -> pd.DataFrame:
    """Build a (state_abbr, normalized_county_name) → fips crosswalk.

    Uses county_boundaries_2024.parquet as the reference source. Falls back
    to an empty crosswalk if the file is unavailable.

    Returns:
        DataFrame with columns: state_abbr, county_norm, fips
    """
    if not COUNTY_BOUNDARIES_PATH.exists():
        logger.warning(
            "County boundaries not found at %s — cannot build crosswalk",
            COUNTY_BOUNDARIES_PATH,
        )
        return pd.DataFrame(columns=["state_abbr", "county_norm", "fips"])

    cb = pd.read_parquet(COUNTY_BOUNDARIES_PATH)

    # Map state FIPS → abbreviation
    cb["state_abbr"] = cb["state_fips"].map(_STATE_FIPS_TO_ABBR)

    # Drop rows with unmappable states (shouldn't happen, but guard)
    unmapped = cb["state_abbr"].isna()
    if unmapped.any():
        logger.debug(
            "Dropping %d county boundary rows with unknown state FIPS",
            unmapped.sum(),
        )
        cb = cb[~unmapped].copy()

    # Normalize county name (apply same normalization as HA names)
    cb["county_norm"] = cb["county_name"].apply(
        lambda n: _normalize_county_name(n) if isinstance(n, str) else ""
    )

    # Normalize FIPS
    cb["fips"] = cb["county_fips"].astype(str).apply(normalize_fips)

    crosswalk = cb[["state_abbr", "county_norm", "fips"]].drop_duplicates(
        subset=["state_abbr", "county_norm"]
    )

    logger.info(
        "Built county FIPS crosswalk: %d entries from %d states/territories",
        len(crosswalk),
        crosswalk["state_abbr"].nunique(),
    )

    return crosswalk


def _enrich_ia_with_ha(
    ia: pd.DataFrame,
    ha: pd.DataFrame,
) -> pd.DataFrame:
    """Merge FEMA HA payout amounts into FEMA IA declaration records.

    Uses a county name → FIPS crosswalk to resolve HA county names to
    5-digit FIPS codes, then aggregates HA payouts to (fips, year) and
    updates the IA records' ia_amount and registrant_count columns.

    Args:
        ia: FEMA IA declarations with fips, year, disaster_number.
        ha: FEMA HA payouts with disaster_number, state, county, ia_amount.

    Returns:
        Updated IA DataFrame with enriched ia_amount and registrant_count.
    """
    # Build crosswalk
    crosswalk = _build_county_fips_crosswalk()
    if crosswalk.empty:
        logger.warning("Empty crosswalk — cannot merge HA data.")
        return ia

    # Normalize HA county names
    ha = ha.copy()
    ha["county_norm"] = ha["county"].apply(_normalize_county_name)

    # Join HA with crosswalk to get FIPS
    ha_with_fips = ha.merge(
        crosswalk,
        left_on=["state", "county_norm"],
        right_on=["state_abbr", "county_norm"],
        how="left",
    )

    # Log unmatched records
    unmatched = ha_with_fips["fips"].isna()
    if unmatched.any():
        n_unmatched = unmatched.sum()
        sample_pairs = (
            ha_with_fips.loc[unmatched, ["state", "county"]]
            .drop_duplicates()
            .head(10)
        )
        logger.warning(
            "%d of %d HA records could not be matched to FIPS. "
            "Sample unmatched (state, county): %s",
            n_unmatched,
            len(ha_with_fips),
            list(sample_pairs.itertuples(index=False, name=None)),
        )
        ha_with_fips = ha_with_fips[~unmatched].copy()

    if ha_with_fips.empty:
        logger.warning("No HA records matched FIPS — ia_amount remains NaN.")
        return ia

    # Get year from IA declarations via disaster_number
    disaster_years = (
        ia[["disaster_number", "year"]]
        .drop_duplicates(subset=["disaster_number"])
    )
    ha_with_fips = ha_with_fips.merge(
        disaster_years,
        on="disaster_number",
        how="inner",
    )

    if ha_with_fips.empty:
        logger.warning(
            "No HA disaster numbers matched IA declarations — "
            "ia_amount remains NaN."
        )
        return ia

    # Aggregate HA payouts to (fips, year)
    ha_agg = (
        ha_with_fips
        .groupby(["fips", "year"])
        .agg(
            ha_ia_amount=("ia_amount", "sum"),
            ha_registrant_count=("registrant_count", "sum"),
        )
        .reset_index()
    )

    logger.info(
        "FEMA HA enrichment: %d county-year rows with $%.1fM in IA payouts",
        len(ha_agg),
        ha_agg["ha_ia_amount"].sum() / 1e6,
    )

    # Update IA records: merge HA aggregates into IA on (fips, year)
    ia = ia.copy()

    # Normalize IA FIPS for join
    ia["fips"] = ia["fips"].astype(str).apply(
        lambda f: normalize_fips(f) if pd.notna(f) and f != "<NA>" else f
    )

    ia = ia.merge(ha_agg, on=["fips", "year"], how="left")

    # Fill ia_amount from HA where available
    has_ha = ia["ha_ia_amount"].notna()
    ia.loc[has_ha, "ia_amount"] = ia.loc[has_ha, "ha_ia_amount"]
    ia.loc[has_ha, "registrant_count"] = ia.loc[has_ha, "ha_registrant_count"]

    ia = ia.drop(columns=["ha_ia_amount", "ha_registrant_count"])

    enriched_count = has_ha.sum()
    total_count = len(ia)
    logger.info(
        "Enriched %d of %d IA declaration rows with HA payout amounts",
        enriched_count,
        total_count,
    )

    return ia


def _load_housing_units() -> pd.DataFrame:
    """Load Census ACS housing unit data."""
    if CENSUS_ACS_COMBINED_PATH.exists():
        logger.info("Loading combined Census ACS from %s", CENSUS_ACS_COMBINED_PATH)
        return pd.read_parquet(CENSUS_ACS_COMBINED_PATH)

    per_year = sorted(CENSUS_ACS_DIR.glob(CENSUS_ACS_PER_YEAR_GLOB))
    per_year = [p for p in per_year if "all" not in p.name]
    if per_year:
        logger.info("Loading %d per-year Census ACS files (fallback)", len(per_year))
        dfs = [pd.read_parquet(p) for p in per_year]
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(
        f"No Census ACS housing unit data found at {CENSUS_ACS_COMBINED_PATH} "
        f"or matching {CENSUS_ACS_DIR / CENSUS_ACS_PER_YEAR_GLOB}"
    )


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _aggregate_county_year(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tiered events to county-year grain.

    Computes county_severity_raw, event_count, total_damage,
    tiered_event_count, unreported_event_count.
    """
    grouped = events.groupby(["fips", "year"]).agg(
        county_severity_raw=("tier_weight", lambda x: x.dropna().sum()),
        event_count=("event_id", "count"),
        total_damage=("total_damage", lambda x: x.sum(min_count=1)),
        tiered_event_count=("severity_tier", lambda x: x.notna().sum()),
        unreported_event_count=("severity_tier", lambda x: x.isna().sum()),
    ).reset_index()

    # Ensure county_severity_raw is 0 (not NaN) when no events are tiered
    grouped["county_severity_raw"] = grouped["county_severity_raw"].fillna(0)

    logger.info(
        "Aggregated %d county-year rows from events",
        len(grouped),
    )

    return grouped


def _join_fema_ia(
    county_year: pd.DataFrame,
    fema_ia: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join FEMA IA payouts and compute FEMA/NCEI ratio."""
    if fema_ia is None or fema_ia.empty:
        logger.warning("No FEMA IA data — setting fema_ncei_ratio to NaN for all county-years.")
        county_year["fema_ia_total"] = np.nan
        county_year["fema_ncei_ratio"] = np.nan
        return county_year

    ia = fema_ia.copy()

    # Drop rows with NaN FIPS
    nan_fips = ia["fips"].isna()
    if nan_fips.any():
        logger.warning("Dropping %d FEMA IA records with NaN FIPS", nan_fips.sum())
        ia = ia[~nan_fips].copy()

    if ia.empty:
        logger.warning("No FEMA IA records with valid FIPS — skipping cross-validation.")
        county_year["fema_ia_total"] = np.nan
        county_year["fema_ncei_ratio"] = np.nan
        return county_year

    # Normalize FIPS
    ia["fips"] = ia["fips"].astype(str).apply(normalize_fips)

    # Clamp negative IA amounts
    neg_mask = ia["ia_amount"] < 0
    if neg_mask.any():
        logger.warning("Clamping %d negative ia_amount values to 0", neg_mask.sum())
        ia.loc[neg_mask, "ia_amount"] = 0

    # Aggregate to county-year
    ia_agg = ia.groupby(["fips", "year"]).agg(
        fema_ia_total=("ia_amount", "sum"),
    ).reset_index()

    # Left join — keep all county-years even without FEMA data
    county_year = county_year.merge(ia_agg, on=["fips", "year"], how="left")
    county_year["fema_ia_total"] = county_year["fema_ia_total"].fillna(0)

    # Compute ratio
    county_year["fema_ncei_ratio"] = np.nan

    # Case 1: both have positive values
    both_positive = (county_year["total_damage"] > 0) & (county_year["fema_ia_total"] > 0)
    county_year.loc[both_positive, "fema_ncei_ratio"] = (
        county_year.loc[both_positive, "fema_ia_total"]
        / county_year.loc[both_positive, "total_damage"]
    )

    # Case 2: NCEI = 0 or NaN but FEMA > 0 → sentinel (underreporting)
    ncei_zero_fema_pos = (
        (county_year["total_damage"].fillna(0) == 0)
        & (county_year["fema_ia_total"] > 0)
    )
    if ncei_zero_fema_pos.any():
        logger.info(
            "%d county-years have FEMA IA > $0 but NCEI damage = $0 (underreporting signal)",
            ncei_zero_fema_pos.sum(),
        )
        county_year.loc[ncei_zero_fema_pos, "fema_ncei_ratio"] = FEMA_NCEI_SENTINEL

    # Case 3: both zero or both NaN → NaN (already initialized)
    # No FEMA data for county → ratio stays NaN (from fillna(0) on fema_ia_total,
    # but if fema_ia_total is 0 and total_damage is 0 or NaN, ratio is NaN)

    return county_year


def _compute_reliability_flag(county_year: pd.DataFrame) -> pd.DataFrame:
    """Compute severity_reliability_flag based on FEMA ratio and pct_missing."""
    county_year["severity_reliability_flag"] = 0

    # Flag if FEMA/NCEI ratio exceeds threshold
    high_ratio = county_year["fema_ncei_ratio"].fillna(0) > FEMA_NCEI_RATIO_THRESHOLD
    county_year.loc[high_ratio, "severity_reliability_flag"] = 1

    # Flag if pct_missing_damage exceeds threshold
    high_missing = county_year["pct_missing_damage"] > MISSING_DAMAGE_THRESHOLD_PCT
    county_year.loc[high_missing, "severity_reliability_flag"] = 1

    flagged = county_year["severity_reliability_flag"].sum()
    if flagged > 0:
        logger.info(
            "%d county-years flagged for reliability concerns "
            "(FEMA ratio > %.1f or pct_missing > %.0f%%)",
            flagged, FEMA_NCEI_RATIO_THRESHOLD, MISSING_DAMAGE_THRESHOLD_PCT,
        )

    return county_year


def _normalize_by_housing(
    county_year: pd.DataFrame,
    housing_units: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize county_severity_raw by housing units.

    Uses temporal fallback: if the exact scoring year is unavailable for a
    county, falls back to the nearest available year.
    """
    hu = housing_units.copy()

    # Normalize FIPS
    hu["fips"] = hu["fips"].apply(normalize_fips)

    # Clamp negative housing units
    neg_mask = hu["total_housing_units"] < 0
    if neg_mask.any():
        logger.warning("Clamping %d negative total_housing_units to 0", neg_mask.sum())
        hu.loc[neg_mask, "total_housing_units"] = 0

    # Build a lookup: for each (fips, year) get total_housing_units
    hu_lookup = hu[["fips", "year", "total_housing_units"]].drop_duplicates(
        subset=["fips", "year"]
    )

    # Direct match first
    merged = county_year.merge(
        hu_lookup, on=["fips", "year"], how="left"
    )

    # Temporal fallback for missing housing units
    missing_hu = merged["total_housing_units"].isna()
    if missing_hu.any():
        fallback_count = 0
        for idx in merged.index[missing_hu]:
            fips = merged.loc[idx, "fips"]
            target_year = merged.loc[idx, "year"]
            county_hu = hu_lookup[hu_lookup["fips"] == fips]
            if county_hu.empty:
                continue
            # Find nearest year
            year_diffs = (county_hu["year"] - target_year).abs()
            nearest_idx = year_diffs.idxmin()
            fallback_year = county_hu.loc[nearest_idx, "year"]
            fallback_val = county_hu.loc[nearest_idx, "total_housing_units"]
            merged.loc[idx, "total_housing_units"] = fallback_val
            fallback_count += 1
            logger.debug(
                "Housing unit fallback: FIPS %s year %d → used year %d",
                fips, target_year, fallback_year,
            )
        if fallback_count > 0:
            logger.info(
                "Used temporal fallback for housing units on %d county-years",
                fallback_count,
            )

    # Check for counties still missing housing units
    still_missing = merged["total_housing_units"].isna()
    if still_missing.any():
        missing_fips = merged.loc[still_missing, "fips"].unique()
        logger.warning(
            "%d county-years have no housing unit data at all (FIPS: %s). "
            "storm_severity_score will be NaN.",
            still_missing.sum(),
            list(missing_fips[:10]),
        )

    # Normalize: score = raw / housing_units
    # Zero housing units → NaN
    zero_hu = merged["total_housing_units"] == 0
    if zero_hu.any():
        logger.warning(
            "%d county-years have 0 housing units — storm_severity_score set to NaN",
            zero_hu.sum(),
        )

    merged["storm_severity_score"] = np.where(
        (merged["total_housing_units"].isna()) | (merged["total_housing_units"] == 0),
        np.nan,
        merged["county_severity_raw"] / merged["total_housing_units"],
    )

    return merged


def _log_summary(county_year: pd.DataFrame) -> None:
    """Log summary statistics for the storm severity computation."""
    total_rows = len(county_year)
    years = county_year["year"].nunique()
    counties = county_year["fips"].nunique()
    flagged = county_year["severity_reliability_flag"].sum()
    nan_scores = county_year["storm_severity_score"].isna().sum()

    logger.info(
        "Storm severity summary: %d county-years (%d counties, %d years), "
        "%d flagged, %d NaN scores",
        total_rows, counties, years, flagged, nan_scores,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "storm_severity_score": float,
        "event_count": int,
        "total_damage": float,
        "severity_reliability_flag": int,
        "pct_missing_damage": float,
        "fema_ncei_ratio": float,
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
            f"County-level severity-weighted storm scores for {year}. "
            f"Normalized per housing unit. "
            f"FEMA/NCEI ratio threshold: {FEMA_NCEI_RATIO_THRESHOLD}. "
            f"Missing damage threshold: {MISSING_DAMAGE_THRESHOLD_PCT}%."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
