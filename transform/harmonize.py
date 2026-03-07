"""Merge all transformed components into a single county-year DataFrame.

This is the single input file for the scoring engine. All years are stacked
(not just the scoring year) so the acceleration engine has historical data
for Theil-Sen slope computation.

Input: All transform module outputs from data/harmonized/
Output: data/harmonized/cci_input_{scoring_year}.parquet

Column naming:
  - Primary scoring columns use component IDs from config/components.py
    (e.g., "extreme_heat_days", "energy_cost_attributed")
  - Auxiliary columns are namespaced: {component_id}__{aux_column}
    (e.g., "storm_severity__reliability_flag")

Confidence: n/a (meta-module)
Attribution: n/a (meta-module)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from config.components import COMPONENTS, ComponentDef
from config.settings import HARMONIZED_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component → file/column mapping
#
# Each entry maps a component ID (from config/components.py) to:
#   - file_prefix: glob pattern for harmonized parquet files
#   - source_column: the column in the parquet to rename to the component ID
#   - aux_columns: additional columns to carry (namespaced as {id}__{col})
# ---------------------------------------------------------------------------
class ComponentMapping(NamedTuple):
    file_prefix: str
    source_column: str
    aux_columns: list[str]


COMPONENT_MAP: dict[str, ComponentMapping] = {
    "hdd_anomaly": ComponentMapping(
        file_prefix="degree_days",
        source_column="hdd_anomaly",
        aux_columns=["hdd_annual", "cdd_annual"],
    ),
    "cdd_anomaly": ComponentMapping(
        file_prefix="degree_days",
        source_column="cdd_anomaly",
        aux_columns=[],  # shared file with hdd_anomaly; aux already captured
    ),
    "extreme_heat_days": ComponentMapping(
        file_prefix="extreme_heat",
        source_column="days_above_95f",
        aux_columns=["days_above_100f"],
    ),
    "storm_severity": ComponentMapping(
        file_prefix="storm_severity",
        source_column="storm_severity_score",
        aux_columns=[
            "event_count", "total_damage", "severity_reliability_flag",
            "pct_missing_damage", "fema_ncei_ratio",
        ],
    ),
    "pm25_annual": ComponentMapping(
        file_prefix="air_quality_scoring",
        source_column="pm25_annual_avg",
        aux_columns=["aqi_very_unhealthy_days", "smoke_days", "smoke_day_method"],
    ),
    "aqi_unhealthy_days": ComponentMapping(
        file_prefix="air_quality_scoring",
        source_column="aqi_unhealthy_days",
        aux_columns=[],  # shared file with pm25_annual; aux already captured
    ),
    "flood_exposure": ComponentMapping(
        file_prefix="flood_zone_scoring",
        source_column="flood_exposure_score",
        aux_columns=[
            "pct_area_high_risk", "pct_area_moderate_risk",
            "pct_hu_high_risk", "nfhl_effective_date", "map_currency_flag",
        ],
    ),
    "wildfire_score": ComponentMapping(
        file_prefix="wildfire_scoring",
        source_column="wildfire_score",
        aux_columns=[
            "whp_mean", "pct_high_hazard", "whp_max",
            "fire_event_count", "fire_damage", "wildfire_activity_score",
        ],
    ),
    "drought_score": ComponentMapping(
        file_prefix="drought_scoring",
        source_column="drought_score",
        aux_columns=["max_severity", "weeks_in_drought", "pct_area_avg"],
    ),
    "energy_cost_attributed": ComponentMapping(
        file_prefix="energy_attribution",
        source_column="climate_attributed_energy_cost",
        aux_columns=[
            "total_energy_cost", "attribution_fraction",
            "regression_r_squared", "structural_breaks_detected",
        ],
    ),
    "health_burden": ComponentMapping(
        file_prefix="health_burden",
        source_column="health_burden_index",
        aux_columns=["heat_ed_rate_per_100k"],
    ),
    "fema_ia_burden": ComponentMapping(
        file_prefix="fema_ia_burden",
        source_column="fema_ia_burden",
        aux_columns=["total_ia_amount", "registrant_count", "ia_events"],
    ),
}

# Intermediate/spatial files to skip (not scoring components)
_SKIP_PREFIXES = {"event_severity_tiers", "station_to_county", "monitor_to_county"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def harmonize(scoring_year: int | None = None) -> pd.DataFrame:
    """Load all component outputs, merge on (fips, year), attach metadata.

    Args:
        scoring_year: If provided, used for output filename and metadata.
            All available years are included regardless (acceleration needs
            historical data).

    Returns:
        Wide DataFrame with one row per (fips, year), component ID columns
        for primary scores, namespaced auxiliary columns, per-component
        confidence/attribution metadata, and a component_count column.
    """
    # --- Load and merge each component source file -----------------------
    # Group mappings by file_prefix to avoid loading the same file twice
    prefix_groups: dict[str, list[str]] = {}
    for comp_id, mapping in COMPONENT_MAP.items():
        prefix_groups.setdefault(mapping.file_prefix, []).append(comp_id)

    merged: pd.DataFrame | None = None

    for prefix, comp_ids in prefix_groups.items():
        source_df = _load_component_files(prefix)
        if source_df is None:
            logger.warning("No files found for prefix '%s' — skipping components: %s", prefix, comp_ids)
            continue

        # Verify county-year grain
        if "fips" not in source_df.columns or "year" not in source_df.columns:
            logger.error("Component '%s' missing fips/year columns — skipping", prefix)
            continue

        # Build the subset: fips, year, renamed primary columns, namespaced aux columns
        subset_cols = {"fips": source_df["fips"], "year": source_df["year"]}

        for comp_id in comp_ids:
            mapping = COMPONENT_MAP[comp_id]

            # Primary scoring column → rename to component ID
            if mapping.source_column in source_df.columns:
                subset_cols[comp_id] = source_df[mapping.source_column]
            else:
                logger.warning(
                    "Column '%s' not found in '%s' files — component '%s' will be NaN",
                    mapping.source_column, prefix, comp_id,
                )

            # Auxiliary columns → namespace as {comp_id}__{aux_col}
            for aux_col in mapping.aux_columns:
                if aux_col in source_df.columns:
                    namespaced = f"{comp_id}__{aux_col}"
                    subset_cols[namespaced] = source_df[aux_col]

        subset = pd.DataFrame(subset_cols)

        # Deduplicate on (fips, year) — shouldn't happen but guard
        dupes = subset.duplicated(subset=["fips", "year"], keep="first")
        if dupes.any():
            logger.warning(
                "Dropping %d duplicate (fips, year) rows from '%s'",
                dupes.sum(), prefix,
            )
            subset = subset[~dupes].copy()

        # Merge
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on=["fips", "year"], how="outer")

    if merged is None or merged.empty:
        logger.error("No component data loaded — returning empty DataFrame.")
        return pd.DataFrame()

    # --- Attach per-component metadata columns ---------------------------
    component_ids = [cid for cid in COMPONENT_MAP if cid in merged.columns]

    for comp_id in component_ids:
        comp_def: ComponentDef = COMPONENTS[comp_id]
        merged[f"{comp_id}__confidence"] = comp_def.confidence
        merged[f"{comp_id}__attribution"] = comp_def.attribution.value

    # --- Component coverage count ----------------------------------------
    merged["component_count"] = merged[component_ids].notna().sum(axis=1)

    # --- Gap fills --------------------------------------------------------
    merged = _apply_gap_fills(merged, component_ids, scoring_year)

    # Recompute component count after gap fills
    merged["component_count"] = merged[component_ids].notna().sum(axis=1)

    # --- Drop rows with zero scoring components --------------------------
    # These arise from outer joins bringing in (fips, year) combos where
    # all primary scoring columns are NaN (e.g., FEMA declarations with no
    # dollar data, or FIPS codes that don't match any other data source).
    zero_mask = merged["component_count"] == 0
    if zero_mask.any():
        logger.info(
            "Dropping %d rows with zero scoring components",
            zero_mask.sum(),
        )
        merged = merged[~zero_mask].copy()

    # --- Sort for deterministic output -----------------------------------
    merged = merged.sort_values(["fips", "year"]).reset_index(drop=True)

    # --- Log summary -----------------------------------------------------
    _log_summary(merged, component_ids)

    # --- Save output -----------------------------------------------------
    if scoring_year is not None:
        _save_output(merged, scoring_year)

    return merged


def run(scoring_year: int = 2024) -> pd.DataFrame:
    """Entry point for pipeline/run_transform.py."""
    return harmonize(scoring_year=scoring_year)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
def _load_component_files(prefix: str) -> pd.DataFrame | None:
    """Load all year files for a component prefix and concatenate.

    Looks for files matching data/harmonized/{prefix}_*.parquet,
    excluding metadata JSON sidecars.
    """
    pattern = f"{prefix}_*.parquet"
    files = sorted(HARMONIZED_DIR.glob(pattern))

    # Filter out metadata files and the "all" consolidated files
    files = [
        f for f in files
        if "_metadata" not in f.name and "_all" not in f.name
    ]

    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception:
            logger.exception("Failed to read %s", f)

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Loaded '%s': %d files, %d rows, years %s–%s",
        prefix, len(dfs), len(result),
        result["year"].min() if "year" in result.columns else "?",
        result["year"].max() if "year" in result.columns else "?",
    )
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _save_output(df: pd.DataFrame, scoring_year: int) -> None:
    """Save harmonized output and metadata sidecar."""
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = HARMONIZED_DIR / f"cci_input_{scoring_year}.parquet"
    metadata_path = HARMONIZED_DIR / f"cci_input_{scoring_year}_metadata.json"

    df.to_parquet(parquet_path, index=False)
    logger.info("Wrote %s (%d rows)", parquet_path, len(df))

    component_ids = [cid for cid in COMPONENT_MAP if cid in df.columns]
    meta = {
        "source": "harmonize",
        "scoring_year": scoring_year,
        "methodology_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(df),
        "total_counties": int(df["fips"].nunique()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "components_included": component_ids,
        "components_missing": [
            cid for cid in COMPONENT_MAP if cid not in df.columns or df[cid].isna().all()
        ],
        "description": (
            f"Harmonized CCI input for scoring year {scoring_year}. "
            f"All available years stacked for acceleration engine. "
            f"Component columns use config/components.py IDs. "
            f"Auxiliary columns namespaced as {{component_id}}__{{column}}."
        ),
    }
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", metadata_path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _log_summary(df: pd.DataFrame, component_ids: list[str]) -> None:
    """Log coverage and completeness summary."""
    total_rows = len(df)
    total_counties = df["fips"].nunique()
    year_range = (df["year"].min(), df["year"].max())

    logger.info(
        "Harmonized output: %d rows, %d counties, years %s–%s",
        total_rows, total_counties, year_range[0], year_range[1],
    )

    # Per-component coverage
    for comp_id in component_ids:
        non_null = df[comp_id].notna().sum()
        pct = non_null / total_rows * 100 if total_rows > 0 else 0
        logger.info(
            "  %-30s %6d / %d rows (%.1f%%)",
            comp_id, non_null, total_rows, pct,
        )

    # Counties with all components vs partial
    full_coverage = (df["component_count"] == len(component_ids)).sum()
    logger.info(
        "  Full coverage (all %d components): %d / %d rows (%.1f%%)",
        len(component_ids), full_coverage, total_rows,
        full_coverage / total_rows * 100 if total_rows > 0 else 0,
    )


# ---------------------------------------------------------------------------
# Gap fills
# ---------------------------------------------------------------------------
def _apply_gap_fills(
    df: pd.DataFrame,
    component_ids: list[str],
    scoring_year: int | None,
) -> pd.DataFrame:
    """Apply data gap-fill strategies after initial merge.

    1. Spatial IDW interpolation for degree-days, extreme heat, air quality
    2. Forward-fill health_burden from most recent year
    3. Set fema_ia_burden = 0 for counties with no FEMA events

    All gap-filled values are flagged and confidence is downgraded.
    """
    result = df.copy()
    target_year = scoring_year or int(result["year"].max())

    # --- 1. Spatial IDW interpolation (per-year) --------------------------
    try:
        from transform.spatial_gap_fill import (
            IDW_COMPONENTS,
            downgrade_interpolated_confidence,
            spatial_idw_fill,
        )

        # Apply IDW to each year that has enough source data to interpolate.
        # Skip years where a component has <10% valid values (nothing to
        # interpolate from) or 0 missing values (nothing to fill).
        years_to_fill = sorted(result["year"].unique())
        years_filled = 0
        for year in years_to_fill:
            year_mask = result["year"] == year
            year_df = result.loc[year_mask].copy().set_index("fips")
            n_counties = len(year_df)

            # Check each IDW component: skip if <10% coverage or 0% missing
            idw_cols = []
            for c in IDW_COMPONENTS:
                if c not in year_df.columns:
                    continue
                n_valid = year_df[c].notna().sum()
                n_missing = year_df[c].isna().sum()
                if n_valid >= n_counties * 0.10 and n_missing > 0:
                    idw_cols.append(c)

            if not idw_cols:
                continue

            filled = spatial_idw_fill(year_df, components=idw_cols)
            filled = downgrade_interpolated_confidence(filled, components=idw_cols)

            # Write back into result
            filled_reset = filled.reset_index()
            for col in filled_reset.columns:
                if col in ["fips", "year"]:
                    continue
                result.loc[year_mask, col] = filled_reset[col].values

            years_filled += 1

        logger.info("Spatial IDW gap fill applied across %d years", years_filled)

    except Exception:
        logger.exception("Spatial gap fill failed — continuing without interpolation")

    # --- 2. Forward-fill health_burden from most recent year ---------------
    if "health_burden" in result.columns:
        result = _forward_fill_health_burden(result, target_year)

    # --- 3. FEMA IA burden: missing = zero burden -------------------------
    if "fema_ia_burden" in result.columns:
        result = _fill_fema_ia_zeros(result)

    return result


def _forward_fill_health_burden(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Forward-fill health_burden from the most recent available year.

    If a county has health_burden data in year Y but not in target_year,
    copy the most recent value forward. Downgrade confidence B→C.
    """
    result = df.copy()

    # Find counties missing health_burden in the target year
    target_mask = result["year"] == target_year
    target_missing = target_mask & result["health_burden"].isna()
    n_missing_before = target_missing.sum()

    if n_missing_before == 0:
        return result

    # Build lookup: for each fips, find the most recent year with health_burden
    has_health = result[result["health_burden"].notna()][["fips", "year", "health_burden"]].copy()
    if has_health.empty:
        logger.info("No health_burden data in any year — cannot forward-fill")
        return result

    # Get most recent year per county
    latest = has_health.sort_values("year").groupby("fips").last().reset_index()
    latest = latest.rename(columns={"health_burden": "health_burden_fill", "year": "fill_year"})

    # Also grab auxiliary columns if available
    aux_cols = [c for c in result.columns if c.startswith("health_burden__") and c != "health_burden__confidence" and c != "health_burden__attribution"]
    if aux_cols:
        has_health_aux = result[result["health_burden"].notna()][["fips", "year"] + aux_cols].copy()
        latest_aux = has_health_aux.sort_values("year").groupby("fips").last().reset_index()
        latest = latest.merge(latest_aux[["fips"] + aux_cols], on="fips", how="left", suffixes=("", "_aux"))

    # Apply forward-fill to target year rows
    missing_fips = result.loc[target_missing, "fips"].values
    fill_lookup = latest.set_index("fips")

    filled_count = 0
    for idx in result.index[target_missing]:
        fips = result.loc[idx, "fips"]
        if fips in fill_lookup.index:
            result.loc[idx, "health_burden"] = fill_lookup.loc[fips, "health_burden_fill"]
            # Downgrade confidence
            result.loc[idx, "health_burden__confidence"] = "C"
            # Copy auxiliary columns
            for aux_col in aux_cols:
                if aux_col in fill_lookup.columns:
                    result.loc[idx, aux_col] = fill_lookup.loc[fips, aux_col]
            filled_count += 1

    if filled_count > 0:
        logger.info(
            "Forward-filled health_burden for %d counties in year %d (confidence downgraded to C)",
            filled_count, target_year,
        )

    return result


def _fill_fema_ia_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Set fema_ia_burden = 0 for counties that have other data but no FEMA events.

    Counties with no FEMA disaster declarations genuinely have zero burden —
    this is correct data, not missing data.
    """
    result = df.copy()

    # Only fill for county-years that have at least some component data
    # (i.e., the county exists in other data sources for that year)
    has_other_data = result[[c for c in COMPONENTS if c in result.columns and c != "fema_ia_burden"]].notna().any(axis=1)
    missing_fema = result["fema_ia_burden"].isna()
    fill_mask = has_other_data & missing_fema

    n_fill = fill_mask.sum()
    if n_fill > 0:
        result.loc[fill_mask, "fema_ia_burden"] = 0.0
        result.loc[fill_mask, "fema_ia_burden__confidence"] = "A"
        # Set auxiliary columns to zero/sensible defaults
        if "fema_ia_burden__total_ia_amount" in result.columns:
            result.loc[fill_mask, "fema_ia_burden__total_ia_amount"] = 0.0
        if "fema_ia_burden__registrant_count" in result.columns:
            result.loc[fill_mask, "fema_ia_burden__registrant_count"] = 0.0
        if "fema_ia_burden__ia_events" in result.columns:
            result.loc[fill_mask, "fema_ia_burden__ia_events"] = 0

        logger.info(
            "Set fema_ia_burden = 0 for %d county-years with no FEMA events",
            n_fill,
        )

    return result
