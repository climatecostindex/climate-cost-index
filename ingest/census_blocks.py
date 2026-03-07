"""Fetch block-group-level housing unit counts and centroids from Census ACS + TIGER CB.

Source: Census Bureau — ACS 5-Year Estimates (Block-Group Level)
        + Cartographic Boundary Files (block-group internal-point coordinates)
ACS API: https://api.census.gov/data/{year}/acs/acs5
CB File: https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_bg_500k.zip

Note: The Census Gazetteer does NOT publish block-group-level files (only tracts
and above). Block-group centroids are sourced from the Cartographic Boundary
shapefile, which includes INTPTLAT/INTPTLON attributes — the same internal-point
coordinates used by the Gazetteer for other geographies.

Fetches raw block-group housing unit counts and internal-point centroids ONLY.
Does NOT compute flood zone overlays, point-in-polygon tests, county-level
aggregations, or any derived metrics — those belong in transform/flood_zone_scoring.py.

Output columns: block_group_fips, county_fips, state_fips, housing_units, lat, lon
Confidence: A
Attribution: none (reference data for flood zone transform)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ingest.base import BaseIngester
from ingest.utils import download_file

# geopandas used only for reading the Cartographic Boundary shapefile
import geopandas as gpd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Census ACS API configuration (block-group level)
# ---------------------------------------------------------------------------
CENSUS_ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"

# ACS variable: total housing units
ACS_VARIABLE = "B25001_001E"

# Census suppression sentinel value
CENSUS_SUPPRESSION_VALUE = -666666666

# Cartographic Boundary File URL for block-group centroids (~93MB national file)
# The Census Gazetteer does NOT publish block-group files, so we use the CB
# shapefile which includes INTPTLAT/INTPTLON attributes for each block group.
CB_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
    "cb_{year}_us_bg_500k.zip"
)

# County-level Cartographic Boundary File — used by transforms that need
# county polygons (station_to_county, monitor_to_county, wildfire_scoring,
# flood_zone_scoring). Downloaded here so all TIGER/CB files live together.
CB_COUNTY_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
    "cb_{year}_us_county_500k.zip"
)

# CB shapefile attribute columns
CB_GEOID_COL = "GEOID"
CB_LAT_COL = "INTPTLAT"
CB_LON_COL = "INTPTLON"

# All U.S. state/territory FIPS codes (2-digit, zero-padded)
# Includes 50 states + DC + territories (PR, VI, GU, AS, MP)
STATE_FIPS_CODES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56", "72",
]

# Conservative rate limit — 500 req/day without key
ACS_CALLS_PER_SECOND = 1.0

# Delay between state queries (seconds) to be polite
STATE_QUERY_DELAY = 0.5


class CensusBlocksIngester(BaseIngester):
    """Ingest block-group housing unit counts and centroid coordinates.

    Queries the Census ACS API state-by-state for block-group-level housing
    unit counts (~220,000 block groups), then joins with the Census
    Cartographic Boundary File for internal-point centroid coordinates.

    This data is consumed by transform/flood_zone_scoring.py to determine
    what percentage of a county's housing units sit in flood zones.
    """

    source_name = "census_blocks"
    confidence = "A"
    attribution = "none"
    calls_per_second = ACS_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "block_group_fips": str,
        "county_fips": str,
        "state_fips": str,
        "housing_units": float,
        "lat": float,
        "lon": float,
    }

    def _default_years(self) -> list[int]:
        """Return the most recent ACS vintage year available.

        ACS 5-year estimates lag ~2 years behind current date.
        For block-group data, a single vintage is sufficient for v1.
        """
        latest = datetime.now().year - 2
        return [latest]

    def _get_api_key(self) -> str | None:
        """Return the Census API key from environment, or None."""
        key = os.getenv("CENSUS_API_KEY", "")
        return key if key else None

    def _fetch_state_block_groups(
        self, year: int, state_fips: str
    ) -> pd.DataFrame:
        """Fetch block-group housing unit counts for a single state.

        Args:
            year: ACS vintage year.
            state_fips: 2-digit zero-padded state FIPS code.

        Returns:
            DataFrame with columns: block_group_fips, county_fips,
            state_fips, housing_units.
        """
        url = CENSUS_ACS_BASE_URL.format(year=year)
        params: dict[str, str] = {
            "get": ACS_VARIABLE,
            "for": "block group:*",
            "in": f"state:{state_fips} county:* tract:*",
        }
        api_key = self._get_api_key()
        if api_key:
            params["key"] = api_key

        resp = self.api_get(url, params=params)
        data = resp.json()

        if not data or len(data) < 2:
            logger.warning(
                "CENSUS_BLOCKS: empty response for state %s, year %d",
                state_fips, year,
            )
            return pd.DataFrame(
                columns=["block_group_fips", "county_fips", "state_fips", "housing_units"]
            )

        return self._parse_acs_response(data)

    def _parse_acs_response(self, data: list[list[str]]) -> pd.DataFrame:
        """Parse Census ACS JSON array into block-group DataFrame.

        Args:
            data: JSON array where data[0] is headers, data[1:] is rows.

        Returns:
            DataFrame with block_group_fips, county_fips, state_fips,
            housing_units columns.
        """
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)

        # Build 12-digit block-group FIPS: state(2) + county(3) + tract(6) + block_group(1)
        df["block_group_fips"] = (
            df["state"].str.zfill(2)
            + df["county"].str.zfill(3)
            + df["tract"].str.zfill(6)
            + df["block group"].str.zfill(1)
        )

        # Extract 5-digit county FIPS and 2-digit state FIPS
        df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
        df["state_fips"] = df["state"].str.zfill(2)

        # Parse housing units: handle suppression values and nulls
        df["housing_units"] = pd.to_numeric(df[ACS_VARIABLE], errors="coerce")
        df.loc[df["housing_units"] == CENSUS_SUPPRESSION_VALUE, "housing_units"] = np.nan

        return df[["block_group_fips", "county_fips", "state_fips", "housing_units"]].copy()

    def _fetch_all_states(self, year: int) -> pd.DataFrame:
        """Fetch block-group data for all states, one at a time.

        Partial failures are logged — successful states are retained.

        Args:
            year: ACS vintage year.

        Returns:
            Combined DataFrame across all successful states.
        """
        all_frames: list[pd.DataFrame] = []
        failed_states: list[str] = []

        for state_fips in STATE_FIPS_CODES:
            try:
                df = self._fetch_state_block_groups(year, state_fips)
                if not df.empty:
                    all_frames.append(df)
                    logger.debug(
                        "CENSUS_BLOCKS: fetched %d block groups for state %s",
                        len(df), state_fips,
                    )
                time.sleep(STATE_QUERY_DELAY)
            except Exception:
                logger.warning(
                    "CENSUS_BLOCKS: failed to fetch state %s for year %d, skipping",
                    state_fips, year,
                    exc_info=True,
                )
                failed_states.append(state_fips)

        if failed_states:
            logger.warning(
                "CENSUS_BLOCKS: %d states failed: %s",
                len(failed_states), failed_states,
            )

        if not all_frames:
            logger.error("CENSUS_BLOCKS: no data retrieved for any state")
            return pd.DataFrame(
                columns=["block_group_fips", "county_fips", "state_fips", "housing_units"]
            )

        return pd.concat(all_frames, ignore_index=True)

    def _fetch_centroids(self, year: int) -> pd.DataFrame:
        """Download and parse the Cartographic Boundary File for BG centroids.

        The Census Gazetteer does not publish block-group-level files.
        Instead, we use the Cartographic Boundary shapefile (500k resolution,
        ~93MB national file) which includes INTPTLAT/INTPTLON attributes —
        the same internal-point coordinates used by the Gazetteer for tracts.

        Args:
            year: CB file year (should match ACS vintage).

        Returns:
            DataFrame with columns: block_group_fips, lat, lon.
        """
        url = CB_URL_TEMPLATE.format(year=year)
        dest = self.cache_dir() / f"cb_{year}_us_bg_500k.zip"

        download_file(url, dest)

        # Read shapefile from zip
        logger.info("CENSUS_BLOCKS: reading block-group centroids from CB file")
        gdf = gpd.read_file(dest)

        # CB files may include INTPTLAT/INTPTLON as attributes, but recent
        # vintages only have geometry. Compute representative_point() which
        # is guaranteed to fall inside the polygon (unlike simple centroid).
        if CB_LAT_COL in gdf.columns and CB_LON_COL in gdf.columns:
            centroids = pd.DataFrame({
                "block_group_fips": gdf[CB_GEOID_COL].astype(str).str.strip(),
                "lat": pd.to_numeric(gdf[CB_LAT_COL], errors="coerce"),
                "lon": pd.to_numeric(gdf[CB_LON_COL], errors="coerce"),
            })
        else:
            logger.info(
                "CENSUS_BLOCKS: INTPTLAT/INTPTLON not in CB file, "
                "computing representative points from geometry"
            )
            rep_points = gdf.geometry.representative_point()
            centroids = pd.DataFrame({
                "block_group_fips": gdf[CB_GEOID_COL].astype(str).str.strip(),
                "lat": rep_points.y,
                "lon": rep_points.x,
            })

        logger.info(
            "CENSUS_BLOCKS: parsed %d block-group centroids from CB file",
            len(centroids),
        )

        return centroids

    def _download_county_boundaries(self, year: int) -> Path:
        """Download and cache county-level Cartographic Boundary data.

        Downloads the county CB shapefile and caches a parquet with county
        reference attributes (FIPS, name, centroid, area). The zip is also
        retained for transforms that need county polygon geometries.

        Args:
            year: CB file year.

        Returns:
            Path to the downloaded zip file.
        """
        url = CB_COUNTY_URL_TEMPLATE.format(year=year)
        dest = self.cache_dir() / f"cb_{year}_us_county_500k.zip"
        download_file(url, dest)

        # Parse county attributes and cache as parquet
        logger.info("CENSUS_BLOCKS: reading county boundaries from CB file")
        gdf = gpd.read_file(dest)

        # County CB files don't include INTPTLAT/INTPTLON — compute
        # representative points from geometry (guaranteed inside polygon)
        rep_points = gdf.geometry.representative_point()

        county_df = pd.DataFrame({
            "county_fips": gdf["GEOID"].astype(str).str.zfill(5),
            "state_fips": gdf["STATEFP"].astype(str).str.zfill(2),
            "county_name": gdf["NAME"].astype(str),
            "lat": rep_points.y,
            "lon": rep_points.x,
            "land_area_sqm": gdf["ALAND"].astype(float),
            "water_area_sqm": gdf["AWATER"].astype(float),
        })

        self.cache_raw(
            county_df,
            label=f"county_boundaries_{year}",
            data_vintage=f"TIGER/CB {year}",
        )

        logger.info(
            "CENSUS_BLOCKS: cached %d county boundaries from CB file",
            len(county_df),
        )
        return dest

    def log_completeness(self, df: pd.DataFrame) -> None:
        """Log block-group and state coverage instead of county coverage."""
        n_block_groups = len(df)
        n_states = df["state_fips"].nunique() if "state_fips" in df.columns else 0
        n_counties = df["county_fips"].nunique() if "county_fips" in df.columns else 0
        logger.info(
            "%s completeness: %d block groups across %d states and %d counties",
            self.source_name,
            n_block_groups,
            n_states,
            n_counties,
        )

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch block-group housing units and centroid coordinates.

        Queries the ACS API state-by-state for housing unit counts, then
        joins with the CB file for centroid coordinates. Caches the
        result as a single national parquet file.

        Args:
            years: ACS vintage years. Defaults to most recent available.
                   For block-group data, typically a single year.

        Returns:
            DataFrame with columns: block_group_fips, county_fips,
            state_fips, housing_units, lat, lon.
        """
        if years is None:
            years = self._default_years()

        # Use the first (and typically only) vintage year
        year = years[0]

        # Fetch ACS housing unit counts for all states
        acs_df = self._fetch_all_states(year)

        if acs_df.empty:
            logger.error("CENSUS_BLOCKS: no ACS data retrieved")
            return pd.DataFrame(columns=list(self.required_columns))

        # Download county-level CB file for transforms that need county polygons
        try:
            self._download_county_boundaries(year)
        except Exception:
            logger.warning(
                "CENSUS_BLOCKS: county CB file download failed for year %d, "
                "trying year %d",
                year, year - 1,
                exc_info=True,
            )
            try:
                self._download_county_boundaries(year - 1)
            except Exception:
                logger.warning(
                    "CENSUS_BLOCKS: county CB file download failed for both years",
                    exc_info=True,
                )

        # Fetch centroid coordinates from Cartographic Boundary File
        try:
            gaz_df = self._fetch_centroids(year)
        except Exception:
            logger.warning(
                "CENSUS_BLOCKS: CB file download failed for year %d, "
                "trying year %d",
                year, year - 1,
                exc_info=True,
            )
            try:
                gaz_df = self._fetch_centroids(year - 1)
                logger.warning(
                    "CENSUS_BLOCKS: using CB year %d with ACS year %d — "
                    "minor geographic mismatches possible",
                    year - 1, year,
                )
            except Exception:
                logger.error(
                    "CENSUS_BLOCKS: CB file download failed for both years, "
                    "proceeding without coordinates",
                    exc_info=True,
                )
                gaz_df = pd.DataFrame(columns=["block_group_fips", "lat", "lon"])

        # Outer join: preserve all block groups from both sources
        result = acs_df.merge(gaz_df, on="block_group_fips", how="outer")

        # Fill missing state_fips/county_fips for Gazetteer-only rows
        mask_no_county = result["county_fips"].isna()
        if mask_no_county.any():
            result.loc[mask_no_county, "state_fips"] = (
                result.loc[mask_no_county, "block_group_fips"].str[:2]
            )
            result.loc[mask_no_county, "county_fips"] = (
                result.loc[mask_no_county, "block_group_fips"].str[:5]
            )
            n_gaz_only = mask_no_county.sum()
            logger.info(
                "CENSUS_BLOCKS: %d block groups in CB file but not in ACS",
                n_gaz_only,
            )

        n_acs_only = result["lat"].isna().sum()
        if n_acs_only > 0:
            logger.info(
                "CENSUS_BLOCKS: %d block groups in ACS but not in CB file",
                n_acs_only,
            )

        # Ensure correct dtypes
        result["block_group_fips"] = result["block_group_fips"].astype(str)
        result["county_fips"] = result["county_fips"].astype(str)
        result["state_fips"] = result["state_fips"].astype(str)
        result["housing_units"] = result["housing_units"].astype(float)
        result["lat"] = result["lat"].astype(float)
        result["lon"] = result["lon"].astype(float)

        # Keep only output schema columns
        result = result[list(self.required_columns)].copy()

        # Cache
        self.cache_raw(
            result,
            label=f"census_blocks_{year}",
            data_vintage=f"ACS 5-year {year}",
        )

        return result
