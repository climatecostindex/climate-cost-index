"""Map NOAA weather stations to counties using point-in-polygon spatial join.

Input:
    - Station metadata (lat, lon) from ingest/noaa_ncei.py:
      ``data/raw/noaa_ncei/noaa_ncei_stations.parquet``
    - County Cartographic Boundary shapefile from ingest/census_blocks.py:
      ``data/raw/census_blocks/cb_{year}_us_county_500k.zip``

Output:
    - ``data/harmonized/station_to_county.parquet``
    - ``data/harmonized/station_to_county_metadata.json``

Output columns:
    station_id (str), fips (str), lat (float), lon (float), station_count (int)

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
STATION_METADATA_PATH = RAW_DIR / "noaa_ncei" / "noaa_ncei_stations.parquet"
COUNTY_BOUNDARY_DIR = RAW_DIR / "census_blocks"
COUNTY_BOUNDARY_GLOB = "cb_*_us_county_500k.zip"

OUTPUT_PARQUET = HARMONIZED_DIR / "station_to_county.parquet"
OUTPUT_METADATA = HARMONIZED_DIR / "station_to_county_metadata.json"

OUTPUT_COLUMNS = ["station_id", "fips", "lat", "lon", "station_count"]

METADATA_SOURCE = "NOAA_NCEI_STATIONS"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def map_stations_to_counties(
    station_metadata: pd.DataFrame | None = None,
    county_boundaries_path: str | Path | None = None,
) -> pd.DataFrame:
    """Assign each NOAA station to its containing county via spatial join.

    Args:
        station_metadata: DataFrame with columns ``station_id``, ``lat``,
            ``lon``.  If *None*, reads from the default cached parquet.
        county_boundaries_path: Path to the Census county boundary shapefile
            zip.  If *None*, auto-discovers from ``data/raw/census_blocks/``.

    Returns:
        DataFrame with columns: ``station_id``, ``fips``, ``lat``, ``lon``,
        ``station_count``.  Counties with zero stations are included with
        ``station_id`` / ``lat`` / ``lon`` as NaN and ``station_count`` = 0.
    """
    # --- Load inputs -------------------------------------------------------
    stations_gdf = _load_station_points(station_metadata)
    counties_gdf = _load_county_polygons(county_boundaries_path)

    if counties_gdf.empty:
        logger.warning("No county boundary polygons loaded — returning empty result.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # --- Spatial join: point-in-polygon ------------------------------------
    joined = _spatial_join(stations_gdf, counties_gdf)

    # --- Station counts per county -----------------------------------------
    station_counts = _compute_station_counts(joined)

    # --- Flag zero-station counties ----------------------------------------
    all_fips = set(counties_gdf["fips"].unique())
    result = _add_zero_station_counties(joined, station_counts, all_fips)

    # --- Enforce output schema ---------------------------------------------
    result = result[OUTPUT_COLUMNS].copy()
    result["station_count"] = result["station_count"].astype(int)

    logger.info(
        "Station-to-county mapping: %d stations → %d counties "
        "(%d counties with ≥1 station, %d with 0)",
        result["station_id"].notna().sum(),
        result["fips"].nunique(),
        (result.groupby("fips")["station_count"].first() > 0).sum(),
        (result.groupby("fips")["station_count"].first() == 0).sum(),
    )

    return result


def run(
    station_metadata: pd.DataFrame | None = None,
    county_boundaries_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full transform: compute mapping, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    result = map_stations_to_counties(station_metadata, county_boundaries_path)

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
def _load_station_points(station_metadata: pd.DataFrame | None) -> gpd.GeoDataFrame:
    """Load station metadata and convert to GeoDataFrame with Point geometries."""
    if station_metadata is None:
        logger.info("Loading station metadata from %s", STATION_METADATA_PATH)
        station_metadata = pd.read_parquet(STATION_METADATA_PATH)

    # Keep only needed columns
    cols = ["station_id", "lat", "lon"]
    missing_cols = [c for c in cols if c not in station_metadata.columns]
    if missing_cols:
        if station_metadata.empty:
            logger.warning("Empty station metadata — no stations to map.")
            return gpd.GeoDataFrame(
                columns=["station_id", "lat", "lon", "geometry"],
                geometry="geometry",
                crs=CRS_NAD83,
            )
        raise ValueError(f"Station metadata missing columns: {missing_cols}")

    df = station_metadata[cols].copy()

    # Drop stations with missing coordinates
    before = len(df)
    df = df.dropna(subset=["lat", "lon"])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d stations with missing coordinates", dropped)

    if df.empty:
        logger.warning("No valid stations after dropping missing coordinates.")
        return gpd.GeoDataFrame(
            columns=["station_id", "lat", "lon", "geometry"],
            geometry="geometry",
            crs=CRS_NAD83,
        )

    # Ensure station_id is string
    df["station_id"] = df["station_id"].astype(str)

    # Build GeoDataFrame
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_NAD83)

    return gdf


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
    stations_gdf: gpd.GeoDataFrame,
    counties_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Point-in-polygon join with nearest-county fallback for unmatched stations."""
    if stations_gdf.empty:
        return pd.DataFrame(columns=["station_id", "fips", "lat", "lon"])

    # Primary join: station within county polygon
    joined = gpd.sjoin(
        stations_gdf,
        counties_gdf[["fips", "geometry"]],
        how="left",
        predicate="within",
    )

    # Identify unmatched stations (fips is NaN after left join)
    unmatched_mask = joined["fips"].isna()
    n_unmatched = unmatched_mask.sum()

    if n_unmatched > 0:
        logger.info(
            "%d stations fell outside all county polygons — using nearest-county fallback",
            n_unmatched,
        )
        # Get original unmatched station points
        unmatched_ids = joined.loc[unmatched_mask, "station_id"].values
        unmatched_gdf = stations_gdf[stations_gdf["station_id"].isin(unmatched_ids)].copy()

        # Nearest-county fallback — reproject to Albers for accurate distances
        nearest = gpd.sjoin_nearest(
            unmatched_gdf.to_crs(CRS_ALBERS),
            counties_gdf[["fips", "geometry"]].to_crs(CRS_ALBERS),
            how="left",
        )

        # Replace unmatched rows with nearest results
        matched = joined.loc[~unmatched_mask].copy()
        result = pd.concat(
            [
                matched[["station_id", "fips", "lat", "lon"]],
                nearest[["station_id", "fips", "lat", "lon"]],
            ],
            ignore_index=True,
        )
    else:
        result = joined[["station_id", "fips", "lat", "lon"]].copy()

    # Deduplicate: a station should map to exactly one county.
    # sjoin can produce duplicates if a point falls on a polygon boundary.
    result = result.drop_duplicates(subset=["station_id"], keep="first")

    return result.reset_index(drop=True)


def _compute_station_counts(joined: pd.DataFrame) -> pd.Series:
    """Count stations per county FIPS."""
    if joined.empty:
        return pd.Series(dtype=int, name="station_count")
    return joined.groupby("fips")["station_id"].count().rename("station_count")


def _add_zero_station_counties(
    joined: pd.DataFrame,
    station_counts: pd.Series,
    all_fips: set[str],
) -> pd.DataFrame:
    """Add rows for counties with zero stations."""
    if joined.empty:
        # All counties are zero-station
        rows = [
            {
                "station_id": np.nan,
                "fips": fips,
                "lat": np.nan,
                "lon": np.nan,
                "station_count": 0,
            }
            for fips in sorted(all_fips)
        ]
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Add station_count to joined rows
    result = joined.merge(
        station_counts.reset_index().rename(columns={"fips": "fips"}),
        on="fips",
        how="left",
    )

    # Find counties with no stations
    covered_fips = set(joined["fips"].unique())
    missing_fips = all_fips - covered_fips

    if missing_fips:
        logger.info("%d counties have zero weather stations", len(missing_fips))
        zero_rows = pd.DataFrame(
            {
                "station_id": np.nan,
                "fips": sorted(missing_fips),
                "lat": np.nan,
                "lon": np.nan,
                "station_count": 0,
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
            "Station-to-county spatial mapping via point-in-polygon join. "
            "Each NOAA GHCN-Daily station is assigned to its containing "
            "county using Census Cartographic Boundary polygons."
        ),
    }
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(meta, f, indent=2)
