"""Convert NFHL geospatial flood zones into county-level numeric scores.

Input:
    - Flood zone polygons from ingest/fema_nfhl.py:
      ``data/raw/fema_nfhl/fema_nfhl_{county_fips}.parquet`` (per-county GeoParquet)
    - Panel effective dates from ingest/fema_nfhl.py:
      ``data/raw/fema_nfhl/fema_nfhl_panels_{county_fips}.parquet``
    - County boundary polygons from ingest/census_blocks.py:
      ``data/raw/census_blocks/cb_{year}_us_county_500k.zip``
    - Block-group housing unit locations from ingest/census_blocks.py:
      ``data/raw/census_blocks/census_blocks_{year}.parquet``

Steps:
    1. Load county boundary polygons, compute land areas in equal-area CRS
    2. Load flood zone polygons per county, classify zones
    3. Compute area-based flood metrics per county
    4. Load block-group housing unit data (optional fallback)
    5. Compute housing-unit-based flood metric
    6. Compute flood exposure score: (pct_high × 3) + (pct_moderate × 1)
    7. Compute map currency flag from panel effective dates
    8. Save per-year parquet + metadata JSON sidecar

Output columns: fips, year, flood_exposure, pct_area_high_risk,
    pct_area_moderate_risk, pct_hu_high_risk, nfhl_effective_date,
    map_currency_flag

Confidence: A
Attribution: proxy
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from config.settings import HARMONIZED_DIR, RAW_DIR, get_settings
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRS_NAD83 = "EPSG:4269"
CRS_ALBERS = "EPSG:5070"  # NAD83/Conus Albers Equal Area — for area calculations

# Zone classification
HIGH_RISK_ZONES = {"A", "AE", "AH", "AO", "V", "VE"}
MODERATE_RISK_ZONE_B = {"B"}
MODERATE_RISK_SUBTYPES = {"0.2 PCT ANNUAL CHANCE FLOOD HAZARD"}

# Scoring weights
HIGH_RISK_WEIGHT = 3
MODERATE_RISK_WEIGHT = 1

# Map currency
MAP_CURRENCY_THRESHOLD_YEARS = 10

# File paths
NFHL_DIR = RAW_DIR / "fema_nfhl"
NFHL_FLOOD_GLOB = "fema_nfhl_[0-9]*.parquet"
NFHL_PANELS_GLOB = "fema_nfhl_panels_*.parquet"
COUNTY_BOUNDARY_DIR = RAW_DIR / "census_blocks"
COUNTY_BOUNDARY_GLOB = "cb_*_us_county_500k.zip"
BLOCK_GROUP_DIR = RAW_DIR / "census_blocks"
BLOCK_GROUP_GLOB = "census_blocks_*.parquet"

OUTPUT_COLUMNS = [
    "fips", "year", "flood_exposure", "pct_area_high_risk",
    "pct_area_moderate_risk", "pct_hu_high_risk", "nfhl_effective_date",
    "map_currency_flag",
]

METADATA_SOURCE = "FEMA_NFHL"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "proxy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_flood_scores(
    flood_zones: gpd.GeoDataFrame,
    county_boundaries: gpd.GeoDataFrame,
    block_group_housing: pd.DataFrame | None = None,
    panel_dates: pd.DataFrame | None = None,
    scoring_year: int | None = None,
) -> pd.DataFrame:
    """Overlay flood zones with county boundaries and compute exposure scores.

    If block_group_housing is provided, computes pct_hu_high_risk.
    Otherwise falls back to area-only calculations.

    Args:
        flood_zones: GeoDataFrame with columns ``flood_zone``, ``zone_subtype``,
            ``county_fips``, ``geometry``. CRS should be EPSG:4269.
        county_boundaries: GeoDataFrame with columns ``fips``, ``geometry``.
            CRS should be EPSG:4269.
        block_group_housing: Optional DataFrame with columns ``county_fips``,
            ``housing_units``, ``lat``, ``lon``.
        panel_dates: Optional DataFrame with columns ``county_fips``,
            ``effective_date``.
        scoring_year: Year for output. Defaults to settings.scoring_year.

    Returns:
        DataFrame with columns: ``fips``, ``year``, ``flood_exposure``,
        ``pct_area_high_risk``, ``pct_area_moderate_risk``,
        ``pct_hu_high_risk``, ``nfhl_effective_date``, ``map_currency_flag``.
    """
    if scoring_year is None:
        scoring_year = get_settings().scoring_year

    # --- Validate inputs ---------------------------------------------------
    _validate_flood_zones_columns(flood_zones)
    _validate_county_boundaries_columns(county_boundaries)

    # --- Handle empty inputs -----------------------------------------------
    if county_boundaries.empty:
        logger.warning("Empty county boundaries — returning empty result.")
        return _empty_output()

    if flood_zones.empty:
        logger.warning("Empty flood zones — returning empty result.")
        return _empty_output()

    # --- Ensure CRS --------------------------------------------------------
    if flood_zones.crs is None or flood_zones.crs.to_epsg() != 4269:
        flood_zones = flood_zones.to_crs(CRS_NAD83)
    if county_boundaries.crs is None or county_boundaries.crs.to_epsg() != 4269:
        county_boundaries = county_boundaries.to_crs(CRS_NAD83)

    # --- Repair invalid geometries -----------------------------------------
    flood_zones = _repair_geometries(flood_zones)

    # --- Classify zones ----------------------------------------------------
    flood_zones = _classify_zones(flood_zones)

    # --- Compute county land areas in equal-area CRS -----------------------
    counties_albers = county_boundaries[["fips", "geometry"]].to_crs(CRS_ALBERS)
    county_areas = counties_albers.copy()
    county_areas["county_area_m2"] = county_areas.geometry.area

    # --- Compute area-based metrics per county -----------------------------
    area_results = _compute_area_metrics(flood_zones, county_boundaries, county_areas)

    # --- Compute housing-unit-based metric ---------------------------------
    if block_group_housing is not None and not block_group_housing.empty:
        hu_results = _compute_housing_unit_metrics(
            flood_zones, block_group_housing, county_boundaries,
        )
        logger.info("Housing unit overlay computed for %d counties", len(hu_results))
    else:
        logger.warning(
            "Block-group housing data unavailable — falling back to area-only "
            "calculations. pct_hu_high_risk will be NaN for all counties."
        )
        hu_results = None

    # --- Compute panel date / map currency ---------------------------------
    if panel_dates is not None and not panel_dates.empty:
        currency_results = _compute_map_currency(panel_dates, scoring_year)
    else:
        currency_results = None

    # --- Merge all results -------------------------------------------------
    result = _merge_results(area_results, hu_results, currency_results, scoring_year)

    return result


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    scoring_year = get_settings().scoring_year

    county_boundaries = _load_county_boundaries()
    flood_zones = _load_all_flood_zones()
    block_group_housing = _load_block_group_housing()
    panel_dates = _load_all_panel_dates()

    result = compute_flood_scores(
        flood_zones, county_boundaries, block_group_housing, panel_dates,
        scoring_year=scoring_year,
    )

    if result.empty:
        logger.warning("No flood zone scoring results to write.")
        return result

    # Save output
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = HARMONIZED_DIR / f"flood_zone_scoring_{scoring_year}.parquet"
    metadata_path = HARMONIZED_DIR / f"flood_zone_scoring_{scoring_year}_metadata.json"

    result.to_parquet(parquet_path, index=False)
    logger.info("Wrote %s (%d counties)", parquet_path, len(result))

    _write_metadata(metadata_path, scoring_year)
    logger.info("Wrote %s", metadata_path)

    logger.info(
        "Flood zone scoring transform complete: %d counties for year %d",
        len(result), scoring_year,
    )

    return result


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_flood_zones_columns(df: gpd.GeoDataFrame) -> None:
    """Raise ValueError if required columns are missing from flood zones."""
    required = {"flood_zone", "zone_subtype", "county_fips", "geometry"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Flood zones missing columns: {sorted(missing)}")


def _validate_county_boundaries_columns(df: gpd.GeoDataFrame) -> None:
    """Raise ValueError if required columns are missing from county boundaries."""
    required = {"fips", "geometry"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"County boundaries missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading helpers (for run() only)
# ---------------------------------------------------------------------------
def _load_county_boundaries() -> gpd.GeoDataFrame:
    """Load county boundary polygons from Census CB shapefile."""
    matches = sorted(COUNTY_BOUNDARY_DIR.glob(COUNTY_BOUNDARY_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No county boundary file matching {COUNTY_BOUNDARY_GLOB} "
            f"in {COUNTY_BOUNDARY_DIR}"
        )
    path = matches[-1]
    logger.info("Loading county boundaries from %s", path)

    gdf = gpd.read_file(path)

    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["fips", "geometry"], geometry="geometry", crs=CRS_NAD83,
        )

    gdf["fips"] = gdf["GEOID"].apply(normalize_fips)

    if gdf.crs is None or gdf.crs.to_epsg() != 4269:
        gdf = gdf.to_crs(CRS_NAD83)

    return gdf[["fips", "geometry"]].copy()


def _load_all_flood_zones() -> gpd.GeoDataFrame:
    """Load all per-county NFHL flood zone GeoParquet files and concatenate."""
    files = sorted(NFHL_DIR.glob(NFHL_FLOOD_GLOB))
    # Exclude panel files
    files = [f for f in files if "_panels_" not in f.name and "_metadata" not in f.name]

    if not files:
        logger.warning("No NFHL flood zone files found in %s", NFHL_DIR)
        return gpd.GeoDataFrame(
            columns=["flood_zone", "zone_subtype", "county_fips", "geometry"],
            geometry="geometry",
            crs=CRS_NAD83,
        )

    logger.info("Loading %d NFHL flood zone files", len(files))
    gdfs = []
    for f in files:
        try:
            gdf = gpd.read_parquet(f)
            gdfs.append(gdf)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e)

    if not gdfs:
        return gpd.GeoDataFrame(
            columns=["flood_zone", "zone_subtype", "county_fips", "geometry"],
            geometry="geometry",
            crs=CRS_NAD83,
        )

    combined = pd.concat(gdfs, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=CRS_NAD83)

    logger.info(
        "Loaded %d total flood zone polygons across %d counties",
        len(combined), combined["county_fips"].nunique(),
    )

    return combined


def _load_all_panel_dates() -> pd.DataFrame | None:
    """Load all per-county NFHL panel date files and concatenate."""
    files = sorted(NFHL_DIR.glob(NFHL_PANELS_GLOB))
    files = [f for f in files if "_metadata" not in f.name]

    if not files:
        logger.warning("No NFHL panel date files found in %s", NFHL_DIR)
        return None

    logger.info("Loading %d NFHL panel date files", len(files))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d panel records", len(combined))
    return combined


def _load_block_group_housing() -> pd.DataFrame | None:
    """Load block-group housing unit data."""
    files = sorted(BLOCK_GROUP_DIR.glob(BLOCK_GROUP_GLOB))
    if not files:
        logger.warning(
            "No block-group housing data found in %s — "
            "will fall back to area-only flood scoring.",
            BLOCK_GROUP_DIR,
        )
        return None

    path = files[-1]
    logger.info("Loading block-group housing data from %s", path)
    df = pd.read_parquet(path)

    if df.empty:
        logger.warning("Block-group housing data is empty.")
        return None

    return df


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid geometries using buffer(0). Drop if repair fails."""
    if gdf.empty:
        return gdf

    invalid_mask = ~gdf.geometry.is_valid
    invalid_count = invalid_mask.sum()

    if invalid_count == 0:
        return gdf

    gdf = gdf.copy()
    gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

    # Check if repair succeeded
    still_invalid = ~gdf.geometry.is_valid
    still_invalid_count = still_invalid.sum()

    repaired_count = invalid_count - still_invalid_count
    if repaired_count > 0:
        logger.info("Repaired %d invalid geometries via buffer(0)", repaired_count)

    if still_invalid_count > 0:
        logger.warning(
            "Dropping %d geometries that could not be repaired", still_invalid_count,
        )
        gdf = gdf[~still_invalid].copy()

    return gdf


def _classify_zones(flood_zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify flood zone polygons into high/moderate/minimal risk."""
    fz = flood_zones.copy()

    # Fill NaN subtypes with empty string
    fz["zone_subtype"] = fz["zone_subtype"].fillna("")

    # High-risk: A, AE, AH, AO, V, VE
    high_mask = fz["flood_zone"].isin(HIGH_RISK_ZONES)

    # Moderate-risk: B zone OR X zone with 0.2% subtype
    moderate_mask = (
        fz["flood_zone"].isin(MODERATE_RISK_ZONE_B)
        | (
            (fz["flood_zone"] == "X")
            & fz["zone_subtype"].str.contains("0.2 PCT ANNUAL CHANCE", case=False, na=False)
        )
    )

    # Minimal-risk: everything else
    minimal_mask = ~high_mask & ~moderate_mask

    fz["risk_class"] = "minimal"
    fz.loc[high_mask, "risk_class"] = "high"
    fz.loc[moderate_mask, "risk_class"] = "moderate"

    # Log unrecognized zones (not in known sets and not X)
    known_zones = HIGH_RISK_ZONES | MODERATE_RISK_ZONE_B | {"X", "C", "D"}
    unrecognized_mask = ~fz["flood_zone"].isin(known_zones)
    unrecognized_count = unrecognized_mask.sum()
    if unrecognized_count > 0:
        unrecognized_values = fz.loc[unrecognized_mask, "flood_zone"].unique().tolist()
        logger.warning(
            "%d polygons with unrecognized flood zone classifications "
            "(treated as minimal-risk): %s",
            unrecognized_count, unrecognized_values,
        )

    logger.info(
        "Zone classification: %d high-risk, %d moderate-risk, %d minimal-risk",
        high_mask.sum(), moderate_mask.sum(), minimal_mask.sum(),
    )

    return fz


def _compute_area_metrics(
    flood_zones: gpd.GeoDataFrame,
    county_boundaries: gpd.GeoDataFrame,
    county_areas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute area-based flood metrics per county.

    Computes individual polygon areas in equal-area CRS and sums per
    (county_fips, risk_class). NFHL map polygons are official FIRM panels
    that tile each county with minimal overlap, so summing individual areas
    is accurate and orders of magnitude faster than dissolving 5M+ polygons.

    Returns DataFrame with: fips, pct_area_high_risk, pct_area_moderate_risk,
    flood_exposure.
    """
    import shapely

    # Filter to only high/moderate risk (minimal-risk doesn't contribute to score)
    scored_fz = flood_zones[flood_zones["risk_class"].isin({"high", "moderate"})].copy()
    if scored_fz.empty:
        return pd.DataFrame(
            columns=["fips", "pct_area_high_risk", "pct_area_moderate_risk",
                      "flood_exposure"],
        )

    # Project to equal-area CRS for area calculation
    scored_fz = scored_fz[["county_fips", "risk_class", "geometry"]].to_crs(CRS_ALBERS)

    # Fix invalid geometries after reprojection
    scored_fz["geometry"] = shapely.make_valid(scored_fz.geometry.values)

    # Compute individual polygon areas (vectorized — fast)
    scored_fz["area_m2"] = scored_fz.geometry.area

    # Sum areas by (county_fips, risk_class)
    area_by_group = scored_fz.groupby(
        ["county_fips", "risk_class"]
    )["area_m2"].sum().reset_index()

    # Pivot to get high_risk and moderate_risk areas per county
    pivoted = area_by_group.pivot_table(
        index="county_fips",
        columns="risk_class",
        values="area_m2",
        fill_value=0.0,
    ).reset_index()

    # Ensure both columns exist
    if "high" not in pivoted.columns:
        pivoted["high"] = 0.0
    if "moderate" not in pivoted.columns:
        pivoted["moderate"] = 0.0

    pivoted = pivoted.rename(columns={
        "county_fips": "fips",
        "high": "high_area_m2",
        "moderate": "moderate_area_m2",
    })

    # Build county area lookup and merge
    county_area_df = county_areas[["fips", "county_area_m2"]].copy()

    # Filter out counties with zero/negative area
    bad_area = county_area_df["county_area_m2"] <= 0
    if bad_area.any():
        bad_fips = county_area_df.loc[bad_area, "fips"].tolist()
        logger.warning(
            "%d counties with zero/negative area — skipping: %s",
            len(bad_fips), bad_fips[:5],
        )
        county_area_df = county_area_df[~bad_area]

    result = pivoted.merge(county_area_df, on="fips", how="inner")

    if result.empty:
        return pd.DataFrame(
            columns=["fips", "pct_area_high_risk", "pct_area_moderate_risk",
                      "flood_exposure"],
        )

    # Compute percentages
    result["pct_area_high_risk"] = (
        result["high_area_m2"] / result["county_area_m2"] * 100
    ).clip(upper=100.0)
    result["pct_area_moderate_risk"] = (
        result["moderate_area_m2"] / result["county_area_m2"] * 100
    ).clip(upper=100.0)

    # Compute flood exposure score
    result["flood_exposure"] = (
        result["pct_area_high_risk"] * HIGH_RISK_WEIGHT
        + result["pct_area_moderate_risk"] * MODERATE_RISK_WEIGHT
    )

    result = result[["fips", "pct_area_high_risk", "pct_area_moderate_risk",
                      "flood_exposure"]].copy()

    logger.info("Computed area-based flood metrics for %d counties", len(result))
    return result


def _compute_housing_unit_metrics(
    flood_zones: gpd.GeoDataFrame,
    block_group_housing: pd.DataFrame,
    county_boundaries: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute housing-unit-based flood metric per county.

    Returns DataFrame with: fips, pct_hu_high_risk.
    """
    # Build point geometries from block-group centroids
    bg = block_group_housing.copy()
    bg = bg.dropna(subset=["lat", "lon"])

    if bg.empty:
        return pd.DataFrame(columns=["fips", "pct_hu_high_risk"])

    bg_gdf = gpd.GeoDataFrame(
        bg,
        geometry=[Point(xy) for xy in zip(bg["lon"], bg["lat"])],
        crs=CRS_NAD83,
    )

    # Get high-risk zones only
    high_risk_fz = flood_zones[flood_zones["risk_class"] == "high"].copy()

    if high_risk_fz.empty:
        # No high-risk zones at all — all counties get 0%
        county_totals = bg.groupby("county_fips")["housing_units"].sum().reset_index()
        county_totals.columns = ["fips", "total_hu"]
        county_totals["pct_hu_high_risk"] = 0.0
        return county_totals[["fips", "pct_hu_high_risk"]]

    # Spatial join: which block-group centroids fall within high-risk zones
    joined = gpd.sjoin(
        bg_gdf,
        high_risk_fz[["geometry"]],
        how="left",
        predicate="within",
    )

    # Block groups that matched at least one high-risk zone
    # Deduplicate to avoid double-counting block groups in overlapping zones
    joined_deduped = joined.drop_duplicates(subset=["block_group_fips"])
    in_high_risk = joined_deduped[joined_deduped["index_right"].notna()]

    # Sum housing units in high-risk zones per county
    hu_in_hr = in_high_risk.groupby("county_fips")["housing_units"].sum().reset_index()
    hu_in_hr.columns = ["fips", "hu_in_high_risk"]

    # Total housing units per county
    hu_total = bg.groupby("county_fips")["housing_units"].sum().reset_index()
    hu_total.columns = ["fips", "total_hu"]

    # Merge and compute percentage
    merged = hu_total.merge(hu_in_hr, on="fips", how="left")
    merged["hu_in_high_risk"] = merged["hu_in_high_risk"].fillna(0)

    # Handle zero housing units (avoid division by zero)
    merged["pct_hu_high_risk"] = np.where(
        merged["total_hu"] > 0,
        (merged["hu_in_high_risk"] / merged["total_hu"]) * 100,
        0.0,
    )

    return merged[["fips", "pct_hu_high_risk"]]


def _compute_map_currency(
    panel_dates: pd.DataFrame,
    scoring_year: int,
) -> pd.DataFrame:
    """Compute map currency flag from panel effective dates.

    Returns DataFrame with: fips, nfhl_effective_date, map_currency_flag.
    """
    df = panel_dates.copy()

    # Ensure county_fips is available
    if "county_fips" not in df.columns:
        return pd.DataFrame(
            columns=["fips", "nfhl_effective_date", "map_currency_flag"],
        )

    # Convert effective_date to pandas Timestamp for groupby compatibility
    df["effective_date"] = pd.to_datetime(df["effective_date"])

    # Treat FEMA sentinel dates (9999-xx-xx) as missing — apply pessimistic flag
    sentinel_mask = df["effective_date"].dt.year >= 9000
    if sentinel_mask.any():
        logger.warning(
            "%d panel records have sentinel effective_date (year >= 9000) — "
            "treated as unknown (pessimistic flag=1)",
            sentinel_mask.sum(),
        )
        df.loc[sentinel_mask, "effective_date"] = pd.NaT

    # Most recent effective date per county
    most_recent = df.groupby("county_fips")["effective_date"].max().reset_index()
    most_recent.columns = ["fips", "effective_date"]

    # Counties with all-NaT dates (sentinel or genuinely missing): pessimistic flag
    missing_date_mask = most_recent["effective_date"].isna()

    # Compute currency flag
    cutoff_date = pd.Timestamp(scoring_year - MAP_CURRENCY_THRESHOLD_YEARS, 1, 1)
    most_recent["map_currency_flag"] = (
        most_recent["effective_date"] < cutoff_date
    ).astype("Int64")
    most_recent.loc[missing_date_mask, "map_currency_flag"] = 1

    # Format effective date as ISO string (date only)
    most_recent["nfhl_effective_date"] = most_recent["effective_date"].dt.strftime(
        "%Y-%m-%d"
    )

    return most_recent[["fips", "nfhl_effective_date", "map_currency_flag"]]


def _merge_results(
    area_results: pd.DataFrame,
    hu_results: pd.DataFrame | None,
    currency_results: pd.DataFrame | None,
    scoring_year: int,
) -> pd.DataFrame:
    """Merge area, housing unit, and currency results into final output."""
    if area_results.empty:
        return _empty_output()

    result = area_results.copy()
    result["year"] = scoring_year

    # Merge housing unit results
    if hu_results is not None and not hu_results.empty:
        result = result.merge(hu_results, on="fips", how="left")
    else:
        result["pct_hu_high_risk"] = np.nan

    # Merge currency results
    if currency_results is not None and not currency_results.empty:
        result = result.merge(currency_results, on="fips", how="left")
        # Counties without panel dates: pessimistic default
        missing_dates = result["nfhl_effective_date"].isna()
        missing_count = missing_dates.sum()
        if missing_count > 0:
            logger.warning(
                "%d counties missing panel effective dates — "
                "setting map_currency_flag=1 (pessimistic)",
                missing_count,
            )
            result.loc[missing_dates, "map_currency_flag"] = 1
    else:
        logger.warning(
            "No panel date data available — setting map_currency_flag=1 "
            "for all %d counties (pessimistic default)",
            len(result),
        )
        result["nfhl_effective_date"] = None
        result["map_currency_flag"] = 1

    # Ensure map_currency_flag is int
    result["map_currency_flag"] = result["map_currency_flag"].fillna(1).astype(int)

    # Enforce output schema
    result = result[OUTPUT_COLUMNS].copy()

    logger.info(
        "Flood zone scoring: %d counties, mean exposure score %.2f",
        len(result), result["flood_exposure"].mean(),
    )

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "flood_exposure": float,
        "pct_area_high_risk": float,
        "pct_area_moderate_risk": float,
        "pct_hu_high_risk": float,
        "nfhl_effective_date": str,
        "map_currency_flag": int,
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
            f"County-level flood exposure scores for scoring year {year}. "
            f"Score formula: (pct_area_high_risk × {HIGH_RISK_WEIGHT}) + "
            f"(pct_area_moderate_risk × {MODERATE_RISK_WEIGHT}). "
            f"Map currency threshold: {MAP_CURRENCY_THRESHOLD_YEARS} years."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
