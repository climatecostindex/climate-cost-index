"""Compute county-level wildfire scores from WHP raster zonal statistics
blended with annual wildfire activity from NCEI storm events.

Input:
    - WHP raster from ingest/usfs_wildfire.py:
      ``data/raw/usfs_wildfire/whp_national.tif`` (or similar GeoTIFF)
    - County boundaries from Census Cartographic Boundary:
      ``data/raw/census_blocks/cb_{year}_us_county_500k.zip``
    - NCEI storm events (optional) from ingest/ncei_storms.py:
      ``data/raw/ncei_storms/`` (Wildfire event_type records)

Steps:
    1. Load county boundary polygons, extract GEOID as 5-digit FIPS
    2. Locate WHP raster, reproject county polygons to match raster CRS
    3. Compute zonal statistics per county (mean, max, pct_high_hazard)
    4. Compute static WHP composite score (whp_score)
    5. Load NCEI wildfire events, compute trailing 5-year activity score
    6. Blend: wildfire_score = WHP_WEIGHT * whp_score + ACTIVITY_WEIGHT * activity
    7. Handle counties with no valid pixels (score=0, mean/max=NaN)
    8. Save per-year parquet + metadata JSON sidecar

Output columns: fips, year, wildfire_score, whp_mean, pct_high_hazard, whp_max,
                fire_event_count, fire_damage, wildfire_activity_score

Confidence: B
Attribution: proxy

Notes:
    - WHP is a static landscape assessment (structural risk).
    - Wildfire activity from NCEI provides an annual signal for acceleration.
    - The blended score enables the acceleration engine to detect trends.
    - If no NCEI data is available, falls back to WHP-only (activity=0).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats

from config.settings import HARMONIZED_DIR, RAW_DIR
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIGH_HAZARD_CLASSES = {4, 5}
VALID_WHP_RANGE = (1, 5)
# WHP classified raster also contains non-hazard land cover classes:
# 6 = non-burnable lands, 7 = open water. These are silently excluded.
NON_HAZARD_CLASSES = {6, 7}
MEAN_WEIGHT = 0.5
HIGH_HAZARD_WEIGHT = 0.5

# Blending weights: structural risk (WHP) vs recent activity (NCEI)
WHP_WEIGHT = 0.7
ACTIVITY_WEIGHT = 0.3
ACTIVITY_WINDOW_YEARS = 5

RASTER_DIR = RAW_DIR / "usfs_wildfire"
RASTER_GLOB = "*.tif"
COUNTY_BOUNDARY_DIR = RAW_DIR / "census_blocks"
COUNTY_BOUNDARY_GLOB = "cb_*_us_county_500k.zip"
STORMS_DIR = RAW_DIR / "ncei_storms"
STORMS_COMBINED_PATH = STORMS_DIR / "ncei_storms_all.parquet"
STORMS_PER_YEAR_GLOB = "ncei_storms_*.parquet"

OUTPUT_COLUMNS = [
    "fips", "year", "wildfire_score", "whp_mean", "pct_high_hazard", "whp_max",
    "fire_event_count", "fire_damage", "wildfire_activity_score",
]

METADATA_SOURCE = "USFS_WHP+NCEI_STORMS"
METADATA_CONFIDENCE = "B"
METADATA_ATTRIBUTION = "proxy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_wildfire_scores(
    raster_path: str | Path,
    county_boundaries: gpd.GeoDataFrame,
    year: int | None = None,
    storm_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute zonal statistics from WHP raster for each county, blended
    with annual wildfire activity from NCEI storm events.

    Args:
        raster_path: Path to the WHP GeoTIFF raster file.
        county_boundaries: GeoDataFrame with county polygons. Must have
            ``GEOID`` and ``geometry`` columns.
        year: Scoring year for the output. Defaults to current year from
            settings if not provided.
        storm_events: Optional DataFrame of NCEI storm events with columns
            ``fips``, ``date``, ``event_type``, ``property_damage``,
            ``crop_damage``. If None, wildfire activity score is 0.

    Returns:
        DataFrame with columns: ``fips``, ``year``, ``wildfire_score``,
        ``whp_mean``, ``pct_high_hazard``, ``whp_max``,
        ``fire_event_count``, ``fire_damage``, ``wildfire_activity_score``.
    """
    import rasterio

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(
            f"WHP raster file not found: {raster_path}"
        )

    if year is None:
        from config.settings import get_settings
        year = get_settings().scoring_year

    # --- Validate county boundaries ---
    _validate_county_boundaries(county_boundaries)

    if county_boundaries.empty:
        logger.warning("Empty county boundaries — returning empty result.")
        return _empty_output()

    # --- Prepare county polygons ---
    counties = _prepare_counties(county_boundaries)

    # --- CRS alignment: reproject counties to match raster ---
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_nodata = src.nodata

    counties = _align_crs(counties, raster_crs)

    # --- Repair invalid geometries ---
    counties = _repair_geometries(counties)

    # --- Compute zonal statistics ---
    results = _compute_zonal_stats(raster_path, counties, raster_nodata)

    # --- Compute static WHP composite score ---
    results = _compute_composite_score(results)

    # --- Compute wildfire activity from NCEI storm events ---
    activity = _compute_wildfire_activity(storm_events, year)

    # --- Merge activity into results ---
    results = _merge_activity(results, activity)

    # --- Blend WHP + activity into final wildfire_score ---
    results = _blend_scores(results)

    # --- Add year column and enforce output schema ---
    results["year"] = year
    results = results[OUTPUT_COLUMNS].copy()

    logger.info(
        "Wildfire scoring complete: %d counties scored, %d with valid WHP data, "
        "%d with fire activity",
        len(results),
        results["whp_mean"].notna().sum(),
        (results["fire_event_count"] > 0).sum(),
    )

    return results


def run(year: int | None = None) -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    raster_path = _locate_raster()
    county_boundaries = _load_county_polygons()
    storm_events = _load_storm_events()

    result = compute_wildfire_scores(
        raster_path, county_boundaries, year=year, storm_events=storm_events,
    )

    if result.empty:
        logger.warning("No wildfire scoring results to write.")
        return result

    # Determine scoring year from results
    scoring_year = result["year"].iloc[0]

    # Save outputs
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = HARMONIZED_DIR / f"wildfire_scoring_{scoring_year}.parquet"
    metadata_path = HARMONIZED_DIR / f"wildfire_scoring_{scoring_year}_metadata.json"

    # Write via file handle to avoid pyarrow/GDAL filesystem registration
    # conflict (ArrowKeyError on 'file' scheme) when rasterio is loaded.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(result)
    with open(parquet_path, "wb") as f:
        pq.write_table(table, f)
    logger.info("Wrote %s (%d counties)", parquet_path, len(result))

    has_activity = storm_events is not None and not storm_events.empty
    _write_metadata(metadata_path, scoring_year, raster_path, has_activity=has_activity)
    logger.info("Wrote %s", metadata_path)

    return result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_county_boundaries(gdf: gpd.GeoDataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    if gdf.empty:
        return
    required = {"GEOID", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"County boundaries missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading helpers (for run() only)
# ---------------------------------------------------------------------------
def _locate_raster() -> Path:
    """Find the WHP GeoTIFF raster in the raw data directory."""
    matches = sorted(RASTER_DIR.glob(RASTER_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No WHP raster file matching {RASTER_GLOB} in {RASTER_DIR}"
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple raster files found in %s — using %s",
            RASTER_DIR, matches[0].name,
        )
    raster_path = matches[0]
    logger.info("Located WHP raster: %s", raster_path)
    return raster_path


def _load_county_polygons() -> gpd.GeoDataFrame:
    """Load county boundary polygons from the census blocks raw directory."""
    matches = sorted(COUNTY_BOUNDARY_DIR.glob(COUNTY_BOUNDARY_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No county boundary file matching {COUNTY_BOUNDARY_GLOB} "
            f"in {COUNTY_BOUNDARY_DIR}"
        )
    county_path = matches[-1]  # most recent year
    logger.info("Loading county boundaries from %s", county_path)
    return gpd.read_file(county_path)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _prepare_counties(county_boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract FIPS from GEOID and keep only needed columns."""
    gdf = county_boundaries.copy()
    gdf["fips"] = gdf["GEOID"].apply(normalize_fips)
    return gdf[["fips", "geometry"]].copy()


def _align_crs(
    counties: gpd.GeoDataFrame,
    raster_crs,
) -> gpd.GeoDataFrame:
    """Reproject county polygons to match the raster CRS if needed."""
    if counties.crs is None:
        logger.warning("County boundaries have no CRS — assuming EPSG:4269")
        counties = counties.set_crs("EPSG:4269")

    if counties.crs != raster_crs:
        logger.info(
            "Reprojecting county boundaries from %s to %s",
            counties.crs, raster_crs,
        )
        counties = counties.to_crs(raster_crs)

    return counties


def _repair_geometries(counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid geometries using buffer(0)."""
    invalid_mask = ~counties.geometry.is_valid
    n_invalid = invalid_mask.sum()

    if n_invalid > 0:
        logger.warning(
            "Repairing %d invalid county geometries with buffer(0)", n_invalid
        )
        for idx in counties.index[invalid_mask]:
            fips = counties.loc[idx, "fips"]
            logger.debug("Repairing invalid geometry for county %s", fips)
            counties.loc[idx, "geometry"] = counties.loc[idx, "geometry"].buffer(0)

        # Check if repair succeeded
        still_invalid = ~counties.geometry.is_valid
        n_still_invalid = still_invalid.sum()
        if n_still_invalid > 0:
            failed_fips = counties.loc[still_invalid, "fips"].tolist()
            logger.warning(
                "Geometry repair failed for %d counties — skipping: %s",
                n_still_invalid, failed_fips,
            )
            counties = counties[~still_invalid].copy()

    return counties


def _compute_zonal_stats(
    raster_path: Path,
    counties: gpd.GeoDataFrame,
    raster_nodata: float | None,
) -> pd.DataFrame:
    """Compute zonal statistics for each county from the WHP raster.

    Uses rasterstats.zonal_stats which handles windowed reading
    efficiently without loading the full raster into memory.
    """
    logger.info("Computing zonal statistics for %d counties...", len(counties))

    # Compute zonal stats: mean, max, count, and categorical counts
    stats = zonal_stats(
        counties,
        str(raster_path),
        stats=["mean", "max", "count"],
        categorical=True,
        nodata=raster_nodata,
        all_touched=False,
    )

    # Build results DataFrame
    records = []
    no_valid_count = 0
    outside_raster_count = 0

    for i, (stat_dict, fips) in enumerate(zip(stats, counties["fips"])):
        valid_count = stat_dict.get("count", 0) or 0
        raw_mean = stat_dict.get("mean")
        raw_max = stat_dict.get("max")

        # Get categorical pixel counts for valid WHP values and out-of-range
        categorical_counts = {
            k: v for k, v in stat_dict.items()
            if isinstance(k, (int, float)) and k not in ("count", "mean", "max", "min")
        }

        # Separate valid hazard pixels from non-hazard / unexpected values
        valid_pixel_counts = {}
        non_hazard_count = 0
        unexpected_count = 0
        for pixel_val, count in categorical_counts.items():
            int_val = int(pixel_val)
            if VALID_WHP_RANGE[0] <= int_val <= VALID_WHP_RANGE[1]:
                valid_pixel_counts[int_val] = count
            elif int_val in NON_HAZARD_CLASSES:
                non_hazard_count += count  # silently exclude
            else:
                unexpected_count += count

        if unexpected_count > 0:
            logger.warning(
                "County %s: %d pixels with unexpected values outside %s and non-hazard %s — treated as NoData",
                fips, unexpected_count, VALID_WHP_RANGE, NON_HAZARD_CLASSES,
            )

        # Recompute stats using only valid-range pixels
        total_valid = sum(valid_pixel_counts.values())

        if total_valid == 0:
            # No valid pixels
            if valid_count == 0 and not categorical_counts:
                outside_raster_count += 1
            else:
                no_valid_count += 1

            records.append({
                "fips": fips,
                "wildfire_score": 0.0,
                "whp_mean": np.nan,
                "pct_high_hazard": 0.0,
                "whp_max": np.nan,
            })
            continue

        # Compute mean from categorical counts (more precise for classified data)
        whp_mean = sum(val * cnt for val, cnt in valid_pixel_counts.items()) / total_valid

        # Compute max from valid pixel values
        whp_max = float(max(valid_pixel_counts.keys()))

        # Compute pct_high_hazard
        high_hazard_count = sum(
            cnt for val, cnt in valid_pixel_counts.items()
            if val in HIGH_HAZARD_CLASSES
        )
        pct_high_hazard = (high_hazard_count / total_valid) * 100.0

        records.append({
            "fips": fips,
            "wildfire_score": 0.0,  # placeholder — computed next
            "whp_mean": whp_mean,
            "pct_high_hazard": pct_high_hazard,
            "whp_max": whp_max,
        })

    if no_valid_count > 0:
        logger.info(
            "%d counties have no valid WHP pixels (all NoData / non-burnable)",
            no_valid_count,
        )

    if outside_raster_count > 0:
        logger.warning(
            "%d counties fall entirely outside the WHP raster extent "
            "(possibly Alaska, Hawaii, or territories)",
            outside_raster_count,
        )

    return pd.DataFrame(records)


def _compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite wildfire_score from whp_mean and pct_high_hazard.

    Formula: wildfire_score = (whp_mean * MEAN_WEIGHT) +
             (pct_high_hazard / 100 * 5 * HIGH_HAZARD_WEIGHT)
    """
    result = df.copy()
    has_data = result["whp_mean"].notna()

    result.loc[has_data, "wildfire_score"] = (
        result.loc[has_data, "whp_mean"] * MEAN_WEIGHT
        + (result.loc[has_data, "pct_high_hazard"] / 100.0 * 5.0) * HIGH_HAZARD_WEIGHT
    )
    # Counties with no data keep wildfire_score = 0.0

    return result


def _compute_wildfire_activity(
    storm_events: pd.DataFrame | None,
    year: int,
) -> pd.DataFrame:
    """Compute trailing 5-year wildfire activity score per county.

    Returns a DataFrame with columns: fips, fire_event_count, fire_damage,
    wildfire_activity_score. Counties with no wildfire events are not included
    (they get filled with zeros during merge).
    """
    if storm_events is None or storm_events.empty:
        logger.info("No storm events provided — wildfire activity scores will be 0.")
        return pd.DataFrame(columns=["fips", "fire_event_count", "fire_damage", "wildfire_activity_score"])

    df = storm_events.copy()

    # Filter to wildfire events only
    if "event_type" not in df.columns:
        logger.warning("Storm events missing 'event_type' column — skipping activity.")
        return pd.DataFrame(columns=["fips", "fire_event_count", "fire_damage", "wildfire_activity_score"])

    df = df[df["event_type"] == "Wildfire"].copy()
    if df.empty:
        logger.info("No wildfire events found in storm data.")
        return pd.DataFrame(columns=["fips", "fire_event_count", "fire_damage", "wildfire_activity_score"])

    # Extract year from date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["event_year"] = df["date"].dt.year

    # Trailing 5-year window: [year - 4, year]
    window_start = year - ACTIVITY_WINDOW_YEARS + 1
    df = df[(df["event_year"] >= window_start) & (df["event_year"] <= year)]

    if df.empty:
        logger.info("No wildfire events in %d-%d window.", window_start, year)
        return pd.DataFrame(columns=["fips", "fire_event_count", "fire_damage", "wildfire_activity_score"])

    # Normalize FIPS
    df["fips"] = df["fips"].apply(normalize_fips)

    # Compute total damage per event
    prop_dmg = pd.to_numeric(df.get("property_damage", 0), errors="coerce").fillna(0)
    crop_dmg = pd.to_numeric(df.get("crop_damage", 0), errors="coerce").fillna(0)
    df["total_damage"] = prop_dmg + crop_dmg

    # Aggregate per county
    agg = df.groupby("fips").agg(
        fire_event_count=("fips", "size"),
        fire_damage=("total_damage", "sum"),
    ).reset_index()

    # Compute activity score: log-scaled relative to national p95, capped at 5
    p95_damage = agg["fire_damage"].quantile(0.95) if len(agg) > 0 else 1.0
    if p95_damage <= 0:
        p95_damage = 1.0  # avoid log(0)

    agg["wildfire_activity_score"] = np.minimum(
        5.0,
        np.log1p(agg["fire_damage"]) / np.log1p(p95_damage) * 5.0,
    )

    logger.info(
        "Wildfire activity: %d counties with events in %d-%d window, "
        "p95 damage=$%.0f",
        len(agg), window_start, year, p95_damage,
    )

    return agg[["fips", "fire_event_count", "fire_damage", "wildfire_activity_score"]]


def _merge_activity(results: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    """Merge wildfire activity scores into WHP results, filling missing with 0."""
    if activity.empty:
        results["fire_event_count"] = 0
        results["fire_damage"] = 0.0
        results["wildfire_activity_score"] = 0.0
        return results

    merged = results.merge(activity, on="fips", how="left")
    merged["fire_event_count"] = merged["fire_event_count"].fillna(0).astype(int)
    merged["fire_damage"] = merged["fire_damage"].fillna(0.0)
    merged["wildfire_activity_score"] = merged["wildfire_activity_score"].fillna(0.0)
    return merged


def _blend_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Blend static WHP score with annual activity score.

    wildfire_score = WHP_WEIGHT * whp_score + ACTIVITY_WEIGHT * activity_score
    """
    result = df.copy()
    # whp_score was stored in wildfire_score by _compute_composite_score
    whp_score = result["wildfire_score"].copy()
    result["wildfire_score"] = (
        WHP_WEIGHT * whp_score
        + ACTIVITY_WEIGHT * result["wildfire_activity_score"]
    )
    return result


def _load_storm_events() -> pd.DataFrame | None:
    """Load NCEI storm events from cached parquet files. Returns None if unavailable."""
    try:
        if STORMS_COMBINED_PATH.exists():
            logger.info("Loading storm events from %s", STORMS_COMBINED_PATH)
            return pd.read_parquet(STORMS_COMBINED_PATH)

        per_year = sorted(STORMS_DIR.glob(STORMS_PER_YEAR_GLOB))
        per_year = [p for p in per_year if "all" not in p.name]
        if not per_year:
            logger.info("No NCEI storm event files found — using WHP-only scoring.")
            return None

        logger.info("Loading %d per-year storm event files", len(per_year))
        dfs = [pd.read_parquet(p) for p in per_year]
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        logger.warning("Failed to load storm events: %s — using WHP-only scoring.", e)
        return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "wildfire_score": float,
        "whp_mean": float,
        "pct_high_hazard": float,
        "whp_max": float,
        "fire_event_count": int,
        "fire_damage": float,
        "wildfire_activity_score": float,
    })


def _write_metadata(
    path: Path, year: int, raster_path: Path, *, has_activity: bool = False,
) -> None:
    """Write metadata JSON sidecar alongside the parquet output."""
    sources = ["USFS_WHP"]
    if has_activity:
        sources.append("NCEI_STORM_EVENTS")

    meta = {
        "source": "+".join(sources),
        "confidence": METADATA_CONFIDENCE,
        "attribution": METADATA_ATTRIBUTION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "data_vintage": raster_path.stem,
        "blending_weights": {
            "whp": WHP_WEIGHT,
            "activity": ACTIVITY_WEIGHT,
            "activity_window_years": ACTIVITY_WINDOW_YEARS,
        },
        "description": (
            f"County-level wildfire hazard scores for scoring year {year}. "
            f"Blended from USFS WHP raster (structural risk, weight={WHP_WEIGHT}) "
            f"and NCEI wildfire events trailing {ACTIVITY_WINDOW_YEARS}-year "
            f"activity (weight={ACTIVITY_WEIGHT}). "
            f"WHP composite: whp_mean={MEAN_WEIGHT}, "
            f"pct_high_hazard={HIGH_HAZARD_WEIGHT}."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
