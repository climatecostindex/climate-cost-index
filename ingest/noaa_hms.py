"""Fetch HMS smoke plume polygon data from NOAA.

Source: NOAA Hazard Mapping System (HMS) Smoke Product
URL: https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/
Format: Daily zip archives containing shapefile components (smoke plume polygons)

Actual server directory structure:
    .../Shapefile/{YYYY}/{MM}/hms_smoke{YYYYMMDD}.zip

Fetches raw smoke plume polygons ONLY. Does NOT compute county-level
smoke day counts or overlay with PM2.5 readings — those belong in
transform/air_quality_scoring.py.

Output: GeoDataFrame with columns: date, geometry, density
Confidence: B
Attribution: proxy
Minimum history: 12 years
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOAA HMS configuration
# ---------------------------------------------------------------------------

# Base URL for HMS smoke plume polygon shapefiles
HMS_BASE_URL = (
    "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/"
    "Smoke_Polygons/Shapefile/"
)

# Regex to find month subdirectories in year listing (e.g., href="08/")
HMS_MONTH_PATTERN = re.compile(r'href="(\d{2})/"')

# Regex to extract date strings from zip filenames in month listings
HMS_FILENAME_PATTERN = re.compile(r"hms_smoke(\d{8})\.zip")

# Rate limit: 1 req/sec — polite for federal NOAA server
HMS_CALLS_PER_SECOND = 1.0

# Minimum trailing history depth (years)
HMS_MIN_HISTORY_YEARS = 12


class NOAAHMSIngester(BaseIngester):
    """Ingest daily HMS smoke plume polygon shapefiles.

    Downloads daily smoke plume zip archives from NOAA's Hazard
    Mapping System. Each archive contains a shapefile with smoke
    plume polygons carrying a density classification (Light, Medium, Heavy).

    Server layout: {base}/{YYYY}/{MM}/hms_smoke{YYYYMMDD}.zip

    This is the PRIMARY source for wildfire-attributable smoke days.
    Raw plume geometries are cached as GeoParquet per year. The
    transform layer handles county overlay and PM2.5 correlation.
    """

    source_name = "noaa_hms"
    confidence = "B"
    attribution = "proxy"
    calls_per_second = HMS_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "date": object,
        "geometry": object,
        "density": str,
    }

    # -- Directory listing & file discovery ------------------------------------

    def _get_months_for_year(self, year: int) -> list[str]:
        """Fetch the year directory and return available month subdirectories.

        Args:
            year: Data year.

        Returns:
            Sorted list of month strings (e.g., ["01", "02", ..., "12"]).
        """
        url = f"{HMS_BASE_URL}{year}/"
        logger.info("NOAA_HMS: fetching year listing for %d", year)
        resp = self.api_get(url)
        months = sorted(HMS_MONTH_PATTERN.findall(resp.text))
        logger.info("NOAA_HMS: found %d month dirs for year %d", len(months), year)
        return months

    def _get_dates_for_month(self, year: int, month: str) -> list[str]:
        """Fetch a month directory and return available date strings.

        Args:
            year: Data year.
            month: Two-digit month string (e.g., "08").

        Returns:
            Sorted list of date strings (YYYYMMDD).
        """
        url = f"{HMS_BASE_URL}{year}/{month}/"
        resp = self.api_get(url)
        dates = sorted(set(HMS_FILENAME_PATTERN.findall(resp.text)))
        return dates

    # -- Local staging paths ---------------------------------------------------

    def _shapefile_staging_dir(self) -> Path:
        """Return the root staging directory for downloaded shapefiles."""
        return self.cache_dir() / "shapefiles"

    def _date_staging_dir(self, year: int, date_str: str) -> Path:
        """Return the staging directory for a specific date, creating if needed."""
        d = self._shapefile_staging_dir() / str(year) / date_str
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _is_date_cached(self, year: int, date_str: str) -> bool:
        """Check if a date's shapefile components are already downloaded."""
        d = self._shapefile_staging_dir() / str(year) / date_str
        if not d.exists():
            return False
        return any(d.glob("*.shp"))

    def _find_shp_path(self, year: int, date_str: str) -> Path | None:
        """Find the .shp file for a cached date.

        Args:
            year: Data year.
            date_str: Date string (YYYYMMDD).

        Returns:
            Path to .shp file, or None if not found.
        """
        d = self._shapefile_staging_dir() / str(year) / date_str
        if not d.exists():
            return None
        # Try the expected name first
        shp = d / f"hms_smoke{date_str}.shp"
        if shp.exists():
            return shp
        # Fallback: any .shp file (zip extraction may use different names)
        shp_files = list(d.glob("*.shp"))
        return shp_files[0] if shp_files else None

    # -- Download --------------------------------------------------------------

    def _download_date(self, year: int, date_str: str) -> bool:
        """Download and extract the zip archive for one date.

        URL format: {base}/{YYYY}/{MM}/hms_smoke{YYYYMMDD}.zip

        Args:
            year: Data year.
            date_str: Date string (YYYYMMDD).

        Returns:
            True on success, False on failure.
        """
        month = date_str[4:6]
        url = f"{HMS_BASE_URL}{year}/{month}/hms_smoke{date_str}.zip"
        dest_dir = self._date_staging_dir(year, date_str)

        try:
            resp = self.api_get(url)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(dest_dir)
            return True
        except Exception:
            logger.warning(
                "NOAA_HMS: failed to download/extract %s",
                date_str,
                exc_info=True,
            )
            return False

    # -- Parsing ---------------------------------------------------------------

    def _parse_shapefile(
        self, shp_path: Path, date_str: str
    ) -> gpd.GeoDataFrame | None:
        """Parse a single date's shapefile into a GeoDataFrame.

        Reads the shapefile, extracts geometry and density columns,
        normalizes the density column name, and attaches the observation
        date extracted from the filename.

        Args:
            shp_path: Path to the .shp file.
            date_str: Date string (YYYYMMDD) for this observation.

        Returns:
            GeoDataFrame with (date, geometry, density), or None if
            the file is empty or malformed.
        """
        try:
            gdf = gpd.read_file(shp_path)
        except Exception:
            logger.warning(
                "NOAA_HMS: failed to read shapefile for %s at %s",
                date_str,
                shp_path,
                exc_info=True,
            )
            return None

        if gdf.empty:
            logger.info(
                "NOAA_HMS: empty shapefile for %s (no smoke observed)", date_str
            )
            return None

        # Find density column (handle variant names)
        density_col = None
        for col in gdf.columns:
            if col.lower() == "density":
                density_col = col
                break

        if density_col is None:
            logger.warning(
                "NOAA_HMS: no density column in shapefile for %s, columns: %s",
                date_str,
                list(gdf.columns),
            )
            return None

        # Parse date from filename string
        obs_date = datetime.strptime(date_str, "%Y%m%d").date()

        return gpd.GeoDataFrame(
            {
                "date": obs_date,
                "density": gdf[density_col].astype(str),
            },
            geometry=gdf.geometry,
            crs=gdf.crs,
        )

    # -- Year-level fetch ------------------------------------------------------

    def _fetch_year(self, year: int) -> gpd.GeoDataFrame | None:
        """Fetch and parse all available dates for one year.

        Traverses monthly subdirectories, discovers daily zip files,
        and supports incremental downloads (skips dates already cached).

        Args:
            year: Data year to fetch.

        Returns:
            GeoDataFrame for the year, or None if no data available.
        """
        try:
            months = self._get_months_for_year(year)
        except Exception:
            logger.warning(
                "NOAA_HMS: failed to fetch year listing for %d",
                year,
                exc_info=True,
            )
            return None

        if not months:
            logger.info("NOAA_HMS: no month directories for year %d", year)
            return None

        # Collect all available dates across months
        all_dates: list[str] = []
        for month in months:
            try:
                dates = self._get_dates_for_month(year, month)
                all_dates.extend(dates)
            except Exception:
                logger.warning(
                    "NOAA_HMS: failed to list month %s/%d, skipping",
                    month,
                    year,
                    exc_info=True,
                )

        if not all_dates:
            logger.info("NOAA_HMS: no dates found for year %d", year)
            return None

        logger.info(
            "NOAA_HMS: found %d dates for year %d", len(all_dates), year
        )

        day_frames: list[gpd.GeoDataFrame] = []
        cached_count = 0
        failed_count = 0
        empty_count = 0

        for date_str in sorted(all_dates):
            # Download if not already cached
            if not self._is_date_cached(year, date_str):
                if not self._download_date(year, date_str):
                    failed_count += 1
                    continue
            else:
                cached_count += 1

            # Parse the shapefile
            shp_path = self._find_shp_path(year, date_str)
            if shp_path is None:
                failed_count += 1
                continue

            gdf = self._parse_shapefile(shp_path, date_str)
            if gdf is not None:
                day_frames.append(gdf)
            else:
                empty_count += 1

        logger.info(
            "NOAA_HMS: year %d — %d dates with plumes, %d cached, "
            "%d failed, %d empty",
            year,
            len(day_frames),
            cached_count,
            failed_count,
            empty_count,
        )

        if not day_frames:
            return None

        year_gdf = gpd.GeoDataFrame(
            pd.concat(day_frames, ignore_index=True), geometry="geometry"
        )

        self.cache_raw(
            year_gdf,
            label=f"noaa_hms_{year}",
            data_vintage=f"NOAA HMS smoke plumes {year}",
        )

        return year_gdf

    # -- Main fetch ------------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch HMS smoke plume polygon data.

        Downloads daily smoke plume zip archives for each year,
        parses them, and concatenates into a combined GeoDataFrame.

        Args:
            years: If provided, fetch only these years. If None,
                   fetch trailing 12 years from current year.

        Returns:
            GeoDataFrame with columns: date, geometry, density.
        """
        if years is None:
            current_year = date.today().year
            years = list(range(
                current_year - HMS_MIN_HISTORY_YEARS, current_year + 1
            ))

        frames: list[gpd.GeoDataFrame] = []

        for year in sorted(years):
            year_gdf = self._fetch_year(year)
            if year_gdf is not None and not year_gdf.empty:
                frames.append(year_gdf)

        if not frames:
            logger.warning("NOAA_HMS: no data parsed from any year")
            return gpd.GeoDataFrame(
                columns=["date", "geometry", "density"],
                geometry="geometry",
            )

        gdf = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry="geometry"
        )

        # Cache combined
        year_min, year_max = min(years), max(years)
        self.cache_raw(
            gdf,
            label="noaa_hms_all",
            data_vintage=f"NOAA HMS smoke plumes {year_min} to {year_max}",
        )

        logger.info(
            "NOAA_HMS: total %d plume records across %d years",
            len(gdf),
            len(frames),
        )

        return gdf
