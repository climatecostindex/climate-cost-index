"""Compute FEMA Individual Assistance burden per housing unit by county.

This is a standalone scoring component measuring disaster recovery financial
burden on households, distinct from storm_severity (which measures event
frequency/intensity). The overlap penalty system handles the correlation
between the two signals.

Input:
    - FEMA IA declarations from ingest/fema_ia.py:
      ``data/raw/fema_ia/fema_ia_all.parquet`` or per-year files
    - FEMA Housing Assistance payouts from ingest/fema_ha.py:
      ``data/raw/fema_ha/fema_ha_all.parquet``
    - County boundary reference from ingest/census_blocks.py:
      ``data/raw/census_blocks/county_boundaries_2024.parquet``
    - Housing units from ingest/census_acs.py:
      ``data/raw/census_acs/census_acs_all.parquet`` or per-year files

Steps:
    1. Load FEMA IA declarations, enrich with FEMA HA payout amounts
       (reuses storm_severity enrichment logic)
    2. Aggregate enriched IA to (fips, year): sum(ia_amount), count(registrants)
    3. Load housing units with temporal fallback
    4. Normalize: fema_ia_burden = total_ia_amount / total_housing_units
    5. Save per-year parquet + metadata JSON sidecar

Output columns: fips, year, fema_ia_burden, total_ia_amount,
               registrant_count, ia_events

Confidence: A (FEMA payout data is comprehensive and audited)
Attribution: proxy
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import HARMONIZED_DIR
from ingest.utils import normalize_fips
from transform.storm_severity import (
    _load_fema_ia,
    _load_housing_units,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "fips", "year", "fema_ia_burden", "total_ia_amount",
    "registrant_count", "ia_events",
]

METADATA_SOURCE = "FEMA_INDIVIDUAL_ASSISTANCE"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "proxy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_fema_ia_burden(
    fema_ia: pd.DataFrame,
    housing_units: pd.DataFrame,
) -> pd.DataFrame:
    """Compute FEMA IA payout burden per housing unit by county-year.

    Args:
        fema_ia: Enriched FEMA IA DataFrame with columns fips, year,
            ia_amount, registrant_count (after HA enrichment).
        housing_units: Census ACS DataFrame with columns fips, year,
            total_housing_units.

    Returns:
        DataFrame with columns: fips, year, fema_ia_burden,
        total_ia_amount, registrant_count, ia_events.
    """
    if fema_ia is None or fema_ia.empty:
        logger.warning("No FEMA IA data — returning empty result.")
        return _empty_output()

    if housing_units is None or housing_units.empty:
        logger.warning("No housing unit data — returning empty result.")
        return _empty_output()

    ia = fema_ia.copy()

    # Drop rows with NaN FIPS or ia_amount
    nan_fips = ia["fips"].isna()
    if nan_fips.any():
        logger.warning("Dropping %d FEMA IA records with NaN FIPS", nan_fips.sum())
        ia = ia[~nan_fips].copy()

    if ia.empty:
        return _empty_output()

    # Normalize FIPS
    ia["fips"] = ia["fips"].astype(str).apply(
        lambda f: normalize_fips(f) if pd.notna(f) and f != "<NA>" else f
    )

    # Clamp negative IA amounts to 0
    neg_mask = ia["ia_amount"] < 0
    if neg_mask.any():
        logger.warning("Clamping %d negative ia_amount values to 0", neg_mask.sum())
        ia.loc[neg_mask, "ia_amount"] = 0

    # --- Aggregate to county-year ----------------------------------------
    county_year = (
        ia.groupby(["fips", "year"])
        .agg(
            total_ia_amount=("ia_amount", "sum"),
            registrant_count=("registrant_count", lambda x: x.sum(min_count=1)),
            ia_events=("fips", "count"),
        )
        .reset_index()
    )

    # NaN total_ia_amount means all records for that county-year had NaN
    # ia_amount (no HA enrichment matched). These are declaration-only
    # records with no dollar signal — drop them.
    no_dollars = county_year["total_ia_amount"].isna()
    if no_dollars.any():
        logger.info(
            "Dropping %d county-years with no IA payout data (declaration-only)",
            no_dollars.sum(),
        )
        county_year = county_year[~no_dollars].copy()

    if county_year.empty:
        logger.warning("No county-years with IA payout data after filtering.")
        return _empty_output()

    logger.info(
        "Aggregated %d county-years with $%.1fM in IA payouts",
        len(county_year),
        county_year["total_ia_amount"].sum() / 1e6,
    )

    # --- Normalize by housing units --------------------------------------
    county_year = _normalize_by_housing(county_year, housing_units)

    # --- Enforce output schema -------------------------------------------
    return county_year[OUTPUT_COLUMNS].copy()


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    fema_ia = _load_fema_ia()
    housing_units = _load_housing_units()

    if fema_ia is None:
        logger.warning("No FEMA IA data available — cannot compute burden.")
        return _empty_output()

    result = compute_fema_ia_burden(fema_ia, housing_units)

    if result.empty:
        logger.warning("No FEMA IA burden results to write.")
        return result

    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

    years = sorted(result["year"].unique())
    for yr in years:
        year_df = result[result["year"] == yr].copy()
        parquet_path = HARMONIZED_DIR / f"fema_ia_burden_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"fema_ia_burden_{yr}_metadata.json"

        year_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_df))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    logger.info(
        "FEMA IA burden transform complete: %d years, %d county-year rows",
        len(years), len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Housing unit normalization
# ---------------------------------------------------------------------------
def _normalize_by_housing(
    county_year: pd.DataFrame,
    housing_units: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize IA payouts by housing units with temporal fallback."""
    hu = housing_units.copy()
    hu["fips"] = hu["fips"].apply(normalize_fips)

    # Clamp negative housing units
    neg_mask = hu["total_housing_units"] < 0
    if neg_mask.any():
        logger.warning("Clamping %d negative total_housing_units to 0", neg_mask.sum())
        hu.loc[neg_mask, "total_housing_units"] = 0

    hu_lookup = hu[["fips", "year", "total_housing_units"]].drop_duplicates(
        subset=["fips", "year"]
    )

    # Direct match
    merged = county_year.merge(hu_lookup, on=["fips", "year"], how="left")

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
            year_diffs = (county_hu["year"] - target_year).abs()
            nearest_idx = year_diffs.idxmin()
            merged.loc[idx, "total_housing_units"] = county_hu.loc[
                nearest_idx, "total_housing_units"
            ]
            fallback_count += 1
        if fallback_count > 0:
            logger.info(
                "Used temporal fallback for housing units on %d county-years",
                fallback_count,
            )

    # Counties still missing housing units → NaN burden
    still_missing = merged["total_housing_units"].isna()
    if still_missing.any():
        logger.warning(
            "%d county-years have no housing unit data — fema_ia_burden will be NaN",
            still_missing.sum(),
        )

    # Normalize: burden = total_ia_amount / housing_units
    merged["fema_ia_burden"] = np.where(
        (merged["total_housing_units"].isna()) | (merged["total_housing_units"] == 0),
        np.nan,
        merged["total_ia_amount"] / merged["total_housing_units"],
    )

    return merged


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "fema_ia_burden": float,
        "total_ia_amount": float,
        "registrant_count": float,
        "ia_events": int,
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
            f"FEMA Individual Assistance payout burden per housing unit "
            f"for {year}. Computed from FEMA IA declarations enriched "
            f"with Housing Assistance payout amounts, normalized by "
            f"Census ACS total housing units."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
