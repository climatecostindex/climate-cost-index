"""Classify individual storm events into damage severity tiers.

Input: Individual event records from ingest/ncei_storms.py

Tier classification:
  Tier 1: total_damage < $50,000      -> weight 1
  Tier 2: $50,000 - $500,000          -> weight 3
  Tier 3: $500,000 - $5,000,000       -> weight 7
  Tier 4: > $5,000,000                -> weight 15
  $0 damage -> flagged as "unreported", not tiered

Output: event records with added columns: total_damage, severity_tier, tier_weight

Notes:
- Tier cutoffs and weights approximate a log-linear loss-severity curve.
- Subject to annual recalibration. Sensitivity suite tests +/-25% perturbation.
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
TIER_CUTOFFS = [50_000, 500_000, 5_000_000]
TIER_WEIGHTS = {1: 1, 2: 3, 3: 7, 4: 15}

REQUIRED_COLUMNS = {"event_id", "fips", "date", "event_type", "property_damage", "crop_damage"}

OUTPUT_COLUMNS = [
    "event_id", "fips", "date", "event_type",
    "total_damage", "severity_tier", "tier_weight",
]

# File paths
STORMS_COMBINED_PATH = RAW_DIR / "ncei_storms" / "ncei_storms_all.parquet"
STORMS_DIR = RAW_DIR / "ncei_storms"
STORMS_PER_YEAR_GLOB = "ncei_storms_*.parquet"

OUTPUT_PATH = HARMONIZED_DIR / "event_severity_tiers.parquet"
METADATA_PATH = HARMONIZED_DIR / "event_severity_tiers_metadata.json"

METADATA_SOURCE = "NCEI_STORM_EVENTS"
METADATA_CONFIDENCE = "B"
METADATA_ATTRIBUTION = "proxy"

EXTREME_DAMAGE_THRESHOLD = 100_000_000_000  # $100B — log warning above this


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_event_severity(events: pd.DataFrame) -> pd.DataFrame:
    """Add severity_tier and tier_weight columns to event records.

    Events with $0 damage are flagged as unreported (tier=NaN, weight=NaN).
    Events with NaN FIPS are excluded (cannot be assigned to a county).

    Args:
        events: DataFrame with columns event_id, fips, date, event_type,
            property_damage, crop_damage.

    Returns:
        DataFrame with columns: event_id, fips, date, event_type,
        total_damage, severity_tier, tier_weight.
    """
    _validate_columns(events)

    if events.empty:
        logger.warning("Empty events input — returning empty result.")
        return _empty_output()

    df = events.copy()

    # --- Normalize FIPS codes -----------------------------------------------
    df = _normalize_and_filter_fips(df)
    if df.empty:
        logger.warning("No events with valid FIPS codes — returning empty result.")
        return _empty_output()

    # --- Filter invalid event_ids -------------------------------------------
    invalid_id_mask = df["event_id"].isna()
    invalid_id_count = invalid_id_mask.sum()
    if invalid_id_count > 0:
        logger.warning("Dropping %d events with missing event_id", invalid_id_count)
        df = df[~invalid_id_mask]

    if df.empty:
        return _empty_output()

    # --- Validate and clamp damage values -----------------------------------
    df = _clamp_negative_damage(df)

    # --- Compute total damage -----------------------------------------------
    df = _compute_total_damage(df)

    # --- Log extreme damage values ------------------------------------------
    _log_extreme_damage(df)

    # --- Classify tiers -----------------------------------------------------
    df = _assign_tiers(df)

    # --- Log summary stats --------------------------------------------------
    _log_classification_summary(df)

    # --- Enforce output schema ----------------------------------------------
    return df[OUTPUT_COLUMNS].copy()


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    events = _load_storm_events()
    result = classify_event_severity(events)

    if result.empty:
        logger.warning("No event severity results to write.")
        return result

    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

    result.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Wrote %s (%d events)", OUTPUT_PATH, len(result))

    _write_metadata(METADATA_PATH)
    logger.info("Wrote %s", METADATA_PATH)

    logger.info(
        "Event severity tier transform complete: %d events classified",
        len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    if df.empty:
        return
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Storm events missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading (for run() only)
# ---------------------------------------------------------------------------
def _load_storm_events() -> pd.DataFrame:
    """Load storm event records from cached parquet files."""
    if STORMS_COMBINED_PATH.exists():
        logger.info("Loading combined storm events from %s", STORMS_COMBINED_PATH)
        return pd.read_parquet(STORMS_COMBINED_PATH)

    # Fall back to per-year files
    per_year = sorted(STORMS_DIR.glob(STORMS_PER_YEAR_GLOB))
    # Exclude the combined file pattern if it somehow matches
    per_year = [p for p in per_year if "all" not in p.name]
    if not per_year:
        raise FileNotFoundError(
            f"No storm event files found at {STORMS_COMBINED_PATH} "
            f"or matching {STORMS_DIR / STORMS_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year storm event files (fallback)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _normalize_and_filter_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FIPS codes and drop events with NaN FIPS."""
    nan_fips_mask = df["fips"].isna()
    nan_fips_count = nan_fips_mask.sum()
    if nan_fips_count > 0:
        logger.warning("Dropping %d events with NaN FIPS codes", nan_fips_count)
        df = df[~nan_fips_mask].copy()

    if df.empty:
        return df

    df["fips"] = df["fips"].apply(normalize_fips)
    return df


def _clamp_negative_damage(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp negative damage values to 0 and log warnings."""
    for col in ("property_damage", "crop_damage"):
        if col not in df.columns:
            continue
        neg_mask = df[col] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            neg_ids = df.loc[neg_mask, "event_id"].head(5).tolist()
            logger.warning(
                "Clamping %d events with negative %s to 0 (sample event_ids: %s)",
                neg_count, col, neg_ids,
            )
            df.loc[neg_mask, col] = 0
    return df


def _compute_total_damage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total_damage = property_damage + crop_damage.

    NaN handling: if one field is NaN, treat as 0. If both are NaN,
    total_damage is NaN (truly missing).
    """
    prop = df["property_damage"]
    crop = df["crop_damage"]

    both_nan = prop.isna() & crop.isna()
    both_nan_count = both_nan.sum()
    if both_nan_count > 0:
        logger.warning(
            "%d events have both property_damage and crop_damage as NaN",
            both_nan_count,
        )

    df["total_damage"] = prop.fillna(0) + crop.fillna(0)
    df.loc[both_nan, "total_damage"] = np.nan

    # Log unreported ($0) count
    zero_mask = df["total_damage"] == 0
    zero_count = zero_mask.sum()
    if zero_count > 0:
        logger.info(
            "%d events have $0 total damage (unreported) — will not be assigned a tier",
            zero_count,
        )

    return df


def _log_extreme_damage(df: pd.DataFrame) -> None:
    """Log warnings for extremely large damage values."""
    extreme_mask = df["total_damage"] > EXTREME_DAMAGE_THRESHOLD
    extreme_count = extreme_mask.sum()
    if extreme_count > 0:
        extreme_ids = df.loc[extreme_mask, "event_id"].head(5).tolist()
        logger.warning(
            "%d events have total_damage > $%.0fB (sample event_ids: %s). "
            "May indicate parsing errors — review manually.",
            extreme_count,
            EXTREME_DAMAGE_THRESHOLD / 1e9,
            extreme_ids,
        )


def _assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Assign severity_tier and tier_weight based on total_damage."""
    df["severity_tier"] = np.nan
    df["tier_weight"] = np.nan

    # Only classify events with total_damage > 0 (excludes NaN and $0)
    has_damage = df["total_damage"].notna() & (df["total_damage"] > 0)

    # Use pd.cut for tier assignment on events with positive damage
    if has_damage.any():
        bins = [0, TIER_CUTOFFS[0], TIER_CUTOFFS[1], TIER_CUTOFFS[2], np.inf]
        labels = [1, 2, 3, 4]
        tiers = pd.cut(
            df.loc[has_damage, "total_damage"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=False,
        )
        df.loc[has_damage, "severity_tier"] = tiers.astype(float)
        df.loc[has_damage, "tier_weight"] = (
            tiers.map(TIER_WEIGHTS).astype(float)
        )

    return df


def _log_classification_summary(df: pd.DataFrame) -> None:
    """Log summary of tier classification results."""
    total = len(df)
    tiered = df["severity_tier"].notna().sum()
    unreported = ((df["total_damage"] == 0)).sum()
    nan_damage = df["total_damage"].isna().sum()

    tier_counts = df["severity_tier"].dropna().value_counts().sort_index()
    tier_str = ", ".join(f"T{int(k)}={int(v)}" for k, v in tier_counts.items())

    logger.info(
        "Classification summary: %d total events, %d tiered [%s], "
        "%d unreported ($0), %d missing damage",
        total, tiered, tier_str, unreported, nan_damage,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "event_id": object,
        "fips": str,
        "date": object,
        "event_type": str,
        "total_damage": float,
        "severity_tier": float,
        "tier_weight": float,
    })


def _write_metadata(path: Path) -> None:
    """Write metadata JSON sidecar alongside the parquet output."""
    meta = {
        "source": METADATA_SOURCE,
        "confidence": METADATA_CONFIDENCE,
        "attribution": METADATA_ATTRIBUTION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "data_vintage": "all_years",
        "description": (
            "Event-level storm severity tier classification. "
            f"Tier cutoffs: {TIER_CUTOFFS}. "
            f"Tier weights: {TIER_WEIGHTS}. "
            "Events with $0 damage flagged as unreported (tier=NaN)."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
