"""Map EPA air quality monitors to counties using point-in-polygon spatial join.

Input:
    - Monitor metadata (lat, lon) from ingest/epa_airnow.py:
      ``data/raw/epa_airnow/epa_aqs_monitors_all.parquet`` (combined) or
      ``data/raw/epa_airnow/epa_aqs_monitors_{year}.parquet`` (per-year)
    - County Cartographic Boundary shapefile from ingest/census_blocks.py:
      ``data/raw/census_blocks/cb_{year}_us_county_500k.zip``

Output:
    - ``data/harmonized/monitor_to_county.parquet``
    - ``data/harmonized/monitor_to_county_metadata.json``

Output columns:
    monitor_id (str), fips (str), lat (float), lon (float), monitor_count (int)

Confidence: A
Attribution: none
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from config.settings import HARMONIZED_DIR, RAW_DIR
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRS_NAD83 = "EPSG:4269"
CRS_ALBERS = "EPSG:5070"  # NAD83/Conus Albers Equal Area — for distance calculations
NEAREST_DISTANCE_THRESHOLD_M = 50_000  # 50 km in meters

MONITOR_METADATA_COMBINED = RAW_DIR / "epa_airnow" / "epa_aqs_monitors_all.parquet"
MONITOR_METADATA_DIR = RAW_DIR / "epa_airnow"
MONITOR_METADATA_GLOB = "epa_aqs_monitors_*.parquet"

COUNTY_BOUNDARY_DIR = RAW_DIR / "census_blocks"
COUNTY_BOUNDARY_GLOB = "cb_*_us_county_500k.zip"

OUTPUT_PARQUET = HARMONIZED_DIR / "monitor_to_county.parquet"
OUTPUT_METADATA = HARMONIZED_DIR / "monitor_to_county_metadata.json"

OUTPUT_COLUMNS = ["monitor_id", "fips", "lat", "lon", "monitor_count"]

METADATA_SOURCE = "EPA_AQS_MONITORS"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def map_monitors_to_counties(
    monitor_metadata: pd.DataFrame | None = None,
    county_boundaries_path: str | Path | None = None,
) -> pd.DataFrame:
    """Assign each EPA monitor to its containing county via spatial join.

    Args:
        monitor_metadata: DataFrame with columns ``monitor_id``, ``lat``,
            ``lon``.  If *None*, reads from the default cached parquet.
        county_boundaries_path: Path to the Census county boundary shapefile
            zip.  If *None*, auto-discovers from ``data/raw/census_blocks/``.

    Returns:
        DataFrame with columns: ``monitor_id``, ``fips``, ``lat``, ``lon``,
        ``monitor_count``.  Counties with zero monitors are included with
        ``monitor_id`` / ``lat`` / ``lon`` as NaN and ``monitor_count`` = 0.
    """
    # --- Load inputs -------------------------------------------------------
    monitors_gdf = _load_monitor_points(monitor_metadata)
    counties_gdf = _load_county_polygons(county_boundaries_path)

    if counties_gdf.empty:
        logger.warning("No county boundary polygons loaded — returning empty result.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # --- Spatial join: point-in-polygon ------------------------------------
    joined = _spatial_join(monitors_gdf, counties_gdf)

    # --- Monitor counts per county -----------------------------------------
    monitor_counts = _compute_monitor_counts(joined)

    # --- Flag zero-monitor counties ----------------------------------------
    all_fips = set(counties_gdf["fips"].unique())
    result = _add_zero_monitor_counties(joined, monitor_counts, all_fips)

    # --- Enforce output schema ---------------------------------------------
    result = result[OUTPUT_COLUMNS].copy()
    result["monitor_count"] = result["monitor_count"].astype(int)

    logger.info(
        "Monitor-to-county mapping: %d monitors → %d counties "
        "(%d counties with ≥1 monitor, %d with 0)",
        result["monitor_id"].notna().sum(),
        result["fips"].nunique(),
        (result.groupby("fips")["monitor_count"].first() > 0).sum(),
        (result.groupby("fips")["monitor_count"].first() == 0).sum(),
    )

    return result


def run(
    monitor_metadata: pd.DataFrame | None = None,
    county_boundaries_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full transform: compute mapping, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    result = map_monitors_to_counties(monitor_metadata, county_boundaries_path)

    # Save outputs
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Wrote %s", OUTPUT_PARQUET)

    _write_metadata()
    logger.info("Wrote %s", OUTPUT_METADATA)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_monitor_points(monitor_metadata: pd.DataFrame | None) -> gpd.GeoDataFrame:
    """Load monitor metadata and convert to GeoDataFrame with Point geometries."""
    if monitor_metadata is None:
        monitor_metadata = _read_monitor_metadata_from_cache()

    # Keep only needed columns
    cols = ["monitor_id", "lat", "lon"]
    missing_cols = [c for c in cols if c not in monitor_metadata.columns]
    if missing_cols:
        if monitor_metadata.empty:
            logger.warning("Empty monitor metadata — no monitors to map.")
            return gpd.GeoDataFrame(
                columns=["monitor_id", "lat", "lon", "geometry"],
                geometry="geometry",
                crs=CRS_NAD83,
            )
        raise ValueError(f"Monitor metadata missing columns: {missing_cols}")

    df = monitor_metadata[cols].copy()

    # Drop monitors with missing coordinates
    before = len(df)
    df = df.dropna(subset=["lat", "lon"])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d monitors with missing coordinates", dropped)

    if df.empty:
        logger.warning("No valid monitors after dropping missing coordinates.")
        return gpd.GeoDataFrame(
            columns=["monitor_id", "lat", "lon", "geometry"],
            geometry="geometry",
            crs=CRS_NAD83,
        )

    # Ensure monitor_id is string
    df["monitor_id"] = df["monitor_id"].astype(str)

    # Deduplicate by monitor_id (in case of per-year concatenation)
    before = len(df)
    df = df.drop_duplicates(subset=["monitor_id"], keep="first")
    deduped = before - len(df)
    if deduped:
        logger.info("Deduplicated %d monitor entries (same monitor_id across years)", deduped)

    # Build GeoDataFrame
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_NAD83)

    return gdf


def _read_monitor_metadata_from_cache() -> pd.DataFrame:
    """Read monitor metadata from cached parquet files."""
    # Prefer combined file
    if MONITOR_METADATA_COMBINED.exists():
        logger.info("Loading monitor metadata from %s", MONITOR_METADATA_COMBINED)
        return pd.read_parquet(MONITOR_METADATA_COMBINED)

    # Fall back to per-year files
    per_year = sorted(MONITOR_METADATA_DIR.glob(MONITOR_METADATA_GLOB))
    # Exclude the "all" file from glob results if it somehow appeared
    per_year = [p for p in per_year if "monitors_all" not in p.name]
    if not per_year:
        raise FileNotFoundError(
            f"No monitor metadata files found in {MONITOR_METADATA_DIR}"
        )

    logger.info("Loading %d per-year monitor metadata files", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate by monitor_id
    combined = combined.drop_duplicates(subset=["monitor_id"], keep="first")
    return combined


def _load_county_polygons(county_boundaries_path: str | Path | None) -> gpd.GeoDataFrame:
    """Load county boundary polygons and normalize FIPS codes."""
    if county_boundaries_path is None:
        matches = sorted(COUNTY_BOUNDARY_DIR.glob(COUNTY_BOUNDARY_GLOB))
        if not matches:
            raise FileNotFoundError(
                f"No county boundary file matching {COUNTY_BOUNDARY_GLOB} "
                f"in {COUNTY_BOUNDARY_DIR}"
            )
        county_boundaries_path = matches[-1]  # most recent year
        logger.info("Auto-discovered county boundaries: %s", county_boundaries_path)

    county_boundaries_path = Path(county_boundaries_path)
    gdf = gpd.read_file(county_boundaries_path)

    if gdf.empty:
        return gpd.GeoDataFrame(columns=["fips", "geometry"], geometry="geometry", crs=CRS_NAD83)

    # Normalize FIPS from GEOID column
    gdf["fips"] = gdf["GEOID"].apply(normalize_fips)

    # Ensure CRS is NAD83
    if gdf.crs is None or gdf.crs.to_epsg() != 4269:
        logger.info("Reprojecting county boundaries to %s", CRS_NAD83)
        gdf = gdf.to_crs(CRS_NAD83)

    return gdf


def _spatial_join(
    monitors_gdf: gpd.GeoDataFrame,
    counties_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Point-in-polygon join with nearest-county fallback for unmatched monitors.

    Monitors beyond the distance threshold (50 km) from any county are
    dropped with a warning.
    """
    if monitors_gdf.empty:
        return pd.DataFrame(columns=["monitor_id", "fips", "lat", "lon"])

    # Primary join: monitor within county polygon
    joined = gpd.sjoin(
        monitors_gdf,
        counties_gdf[["fips", "geometry"]],
        how="left",
        predicate="within",
    )

    # Identify unmatched monitors (fips is NaN after left join)
    unmatched_mask = joined["fips"].isna()
    n_unmatched = unmatched_mask.sum()

    if n_unmatched > 0:
        logger.info(
            "%d monitors fell outside all county polygons — using nearest-county fallback",
            n_unmatched,
        )
        # Get original unmatched monitor points
        unmatched_ids = joined.loc[unmatched_mask, "monitor_id"].values
        unmatched_gdf = monitors_gdf[monitors_gdf["monitor_id"].isin(unmatched_ids)].copy()

        # Nearest-county fallback — reproject to Albers for accurate distances
        nearest = gpd.sjoin_nearest(
            unmatched_gdf.to_crs(CRS_ALBERS),
            counties_gdf[["fips", "geometry"]].to_crs(CRS_ALBERS),
            how="left",
            distance_col="_dist_m",
        )

        # Log each fallback match for auditing
        for _, row in nearest.iterrows():
            dist_km = row["_dist_m"] / 1000.0
            logger.info(
                "Fallback match: monitor_id=%s → fips=%s (distance=%.1f km)",
                row["monitor_id"],
                row["fips"],
                dist_km,
            )

        # Drop monitors beyond distance threshold
        beyond = nearest["_dist_m"] > NEAREST_DISTANCE_THRESHOLD_M
        if beyond.any():
            dropped_ids = nearest.loc[beyond, "monitor_id"].values
            logger.warning(
                "Dropped %d monitors beyond %.0f km threshold: %s",
                beyond.sum(),
                NEAREST_DISTANCE_THRESHOLD_M / 1000,
                list(dropped_ids),
            )
            nearest = nearest[~beyond]

        # Replace unmatched rows with nearest results
        matched = joined.loc[~unmatched_mask].copy()
        result = pd.concat(
            [
                matched[["monitor_id", "fips", "lat", "lon"]],
                nearest[["monitor_id", "fips", "lat", "lon"]],
            ],
            ignore_index=True,
        )
    else:
        result = joined[["monitor_id", "fips", "lat", "lon"]].copy()

    # Deduplicate: a monitor should map to exactly one county.
    # sjoin can produce duplicates if a point falls on a polygon boundary.
    result = result.drop_duplicates(subset=["monitor_id"], keep="first")

    return result.reset_index(drop=True)


def _compute_monitor_counts(joined: pd.DataFrame) -> pd.Series:
    """Count monitors per county FIPS."""
    if joined.empty:
        return pd.Series(dtype=int, name="monitor_count")
    return joined.groupby("fips")["monitor_id"].count().rename("monitor_count")


def _add_zero_monitor_counties(
    joined: pd.DataFrame,
    monitor_counts: pd.Series,
    all_fips: set[str],
) -> pd.DataFrame:
    """Add rows for counties with zero monitors."""
    if joined.empty:
        # All counties are zero-monitor
        rows = [
            {
                "monitor_id": np.nan,
                "fips": fips,
                "lat": np.nan,
                "lon": np.nan,
                "monitor_count": 0,
            }
            for fips in sorted(all_fips)
        ]
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Add monitor_count to joined rows
    result = joined.merge(
        monitor_counts.reset_index().rename(columns={"fips": "fips"}),
        on="fips",
        how="left",
    )

    # Find counties with no monitors
    covered_fips = set(joined["fips"].unique())
    missing_fips = all_fips - covered_fips

    if missing_fips:
        logger.info("%d counties have zero air quality monitors", len(missing_fips))
        zero_rows = pd.DataFrame(
            {
                "monitor_id": np.nan,
                "fips": sorted(missing_fips),
                "lat": np.nan,
                "lon": np.nan,
                "monitor_count": 0,
            }
        )
        result = pd.concat([result, zero_rows], ignore_index=True)

    return result


def _write_metadata() -> None:
    """Write metadata JSON sidecar alongside the parquet output."""
    meta = {
        "source": METADATA_SOURCE,
        "confidence": METADATA_CONFIDENCE,
        "attribution": METADATA_ATTRIBUTION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "data_vintage": "static",
        "description": (
            "Monitor-to-county spatial mapping via point-in-polygon join. "
            "Each EPA AQS air quality monitor is assigned to its containing "
            "county using Census Cartographic Boundary polygons."
        ),
    }
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(meta, f, indent=2)
