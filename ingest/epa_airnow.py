"""Fetch daily PM2.5 monitor readings from EPA AQS pre-generated files.

Source: EPA Air Quality System (AQS)
Primary: Pre-generated annual summary files (bulk CSV downloads)
URL: https://aqs.epa.gov/aqsweb/airdata/download_files.html
Fallback: AQS API at https://aqs.epa.gov/data/api/ (not yet implemented)

Fetches raw monitor readings ONLY. Does NOT compute county-level averages,
annual summaries, smoke day counts, or AQI threshold exceedances — those
belong in transform/air_quality_scoring.py.

Output — readings table:
    monitor_id, fips, date, pm25_value, aqi_value, lat, lon
Output — monitor metadata table:
    monitor_id, lat, lon, county_fips, state
Confidence: A
Attribution: proxy
Minimum history: 12 years
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date

import pandas as pd

from ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EPA AQS configuration
# ---------------------------------------------------------------------------

# Base URL for pre-generated annual PM2.5 daily summary files.
# Files are zipped CSVs: daily_88101_{year}.zip
EPA_AQS_BULK_URL = "https://aqs.epa.gov/aqsweb/airdata"

# EPA parameter code for PM2.5 - Local Conditions
EPA_PM25_PARAM_CODE = "88101"

# Rate limit: bulk downloads have no formal limit, be polite (1 req/sec)
EPA_BULK_CALLS_PER_SECOND = 1.0

# Minimum trailing history depth (years)
EPA_MIN_HISTORY_YEARS = 12

# Columns to read from the EPA bulk CSV files
EPA_CSV_USECOLS = [
    "State Code",
    "County Code",
    "Site Num",
    "POC",
    "Latitude",
    "Longitude",
    "Date Local",
    "Arithmetic Mean",
    "AQI",
]


class EPAAirNowIngester(BaseIngester):
    """Ingest daily PM2.5 and AQI monitor readings from EPA AQS.

    Downloads pre-generated annual summary files (daily_88101_{year}.zip)
    containing daily PM2.5 readings for all national monitors. Produces
    two outputs: a readings table (daily observations) and a monitor
    metadata table (one row per unique monitor with location info).

    This is reference data for the air quality component. The transform
    layer handles county-level aggregation and smoke day identification.
    """

    source_name = "epa_airnow"
    confidence = "A"
    attribution = "proxy"
    calls_per_second = EPA_BULK_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "monitor_id": str,
        "fips": str,
        "date": object,  # Python date objects
        "pm25_value": float,
        "aqi_value": float,
        "lat": float,
        "lon": float,
    }

    metadata_columns: dict[str, type] = {
        "monitor_id": str,
        "lat": float,
        "lon": float,
        "county_fips": str,
        "state": str,
    }

    # -- Download & parse ------------------------------------------------------

    def _download_annual_zip(self, year: int) -> bytes:
        """Download the annual PM2.5 daily summary zip file.

        Args:
            year: Data year to download.

        Returns:
            Raw bytes of the zip file.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
        """
        url = f"{EPA_AQS_BULK_URL}/daily_{EPA_PM25_PARAM_CODE}_{year}.zip"
        logger.info("EPA_AQS: downloading %s", url)
        resp = self.api_get(url)
        return resp.content

    def _parse_annual_csv(self, zip_bytes: bytes) -> pd.DataFrame:
        """Parse the annual PM2.5 daily summary from a zip file.

        Extracts the CSV from the zip archive, reads the needed columns,
        and constructs output fields (monitor_id, fips, date).

        Args:
            zip_bytes: Raw zip file bytes.

        Returns:
            DataFrame with readings-table columns.

        Raises:
            ValueError: If the zip contains no CSV files.
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError("EPA_AQS: zip file contains no CSV files")
            csv_name = csv_names[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    usecols=EPA_CSV_USECOLS,
                    dtype={
                        "State Code": str,
                        "County Code": str,
                        "Site Num": str,
                        "POC": str,
                    },
                    low_memory=False,
                )

        # Construct monitor_id: state-county-site-poc
        df["monitor_id"] = (
            df["State Code"].str.zfill(2) + "-"
            + df["County Code"].str.zfill(3) + "-"
            + df["Site Num"].str.zfill(4) + "-"
            + df["POC"]
        )

        # Construct 5-digit FIPS
        df["fips"] = (
            df["State Code"].str.zfill(2) + df["County Code"].str.zfill(3)
        )

        # Parse date
        df["date"] = pd.to_datetime(df["Date Local"], errors="coerce").dt.date

        # PM2.5 and AQI
        df["pm25_value"] = pd.to_numeric(df["Arithmetic Mean"], errors="coerce")
        df["aqi_value"] = pd.to_numeric(df["AQI"], errors="coerce")

        # Coordinates
        df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")

        # Select output columns
        return df[list(self.required_columns)].copy()

    def _extract_monitor_metadata(self, readings: pd.DataFrame) -> pd.DataFrame:
        """Extract unique monitor metadata from readings.

        Takes one representative row per monitor_id, keeping location
        and FIPS information.

        Args:
            readings: Readings DataFrame with monitor_id, lat, lon, fips.

        Returns:
            DataFrame with monitor metadata columns.
        """
        monitors = (
            readings[["monitor_id", "lat", "lon", "fips"]]
            .drop_duplicates(subset="monitor_id", keep="first")
            .copy()
        )
        monitors["county_fips"] = monitors["fips"]
        monitors["state"] = monitors["fips"].str[:2]
        return monitors[list(self.metadata_columns)].copy()

    # -- Main fetch ------------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch EPA AQS PM2.5 daily readings.

        Downloads pre-generated annual summary files for each year,
        parses them, caches readings and monitor metadata separately.

        Args:
            years: If provided, fetch only these years. If None,
                   fetch trailing 12 years from current year.

        Returns:
            Readings DataFrame with columns: monitor_id, fips, date,
            pm25_value, aqi_value, lat, lon.
        """
        if years is None:
            current_year = date.today().year
            years = list(range(
                current_year - EPA_MIN_HISTORY_YEARS, current_year + 1
            ))

        frames: list[pd.DataFrame] = []

        for year in sorted(years):
            try:
                zip_bytes = self._download_annual_zip(year)
                frame = self._parse_annual_csv(zip_bytes)
                frames.append(frame)

                # Cache readings per year
                self.cache_raw(
                    frame,
                    label=f"epa_aqs_readings_{year}",
                    data_vintage=f"EPA AQS PM2.5 daily readings {year}",
                )

                # Cache monitor metadata per year
                meta = self._extract_monitor_metadata(frame)
                self.cache_raw(
                    meta,
                    label=f"epa_aqs_monitors_{year}",
                    data_vintage=f"EPA AQS monitor metadata {year}",
                )

                logger.info(
                    "EPA_AQS: parsed %d readings, %d monitors for year %d",
                    len(frame), len(meta), year,
                )

            except Exception:
                logger.warning(
                    "EPA_AQS: failed to fetch/parse year %d, skipping",
                    year,
                    exc_info=True,
                )

        if not frames:
            logger.warning("EPA_AQS: no data parsed from any year")
            return pd.DataFrame(columns=list(self.required_columns))

        df = pd.concat(frames, ignore_index=True)

        # Cache combined readings
        year_min, year_max = min(years), max(years)
        self.cache_raw(
            df,
            label="epa_aqs_readings_all",
            data_vintage=f"EPA AQS PM2.5 daily readings {year_min} to {year_max}",
        )

        # Cache combined monitor metadata
        all_meta = self._extract_monitor_metadata(df)
        self.cache_raw(
            all_meta,
            label="epa_aqs_monitors_all",
            data_vintage=f"EPA AQS monitor metadata {year_min} to {year_max}",
        )

        logger.info(
            "EPA_AQS: total %d readings, %d unique monitors across %d years",
            len(df), df["monitor_id"].nunique(), len(frames),
        )

        return df

    def fetch_monitor_metadata(self) -> pd.DataFrame:
        """Load cached monitor metadata if available.

        Returns the combined monitor metadata from the most recent
        fetch(). If no cached data exists, returns an empty DataFrame.

        Returns:
            DataFrame with columns: monitor_id, lat, lon, county_fips, state.
        """
        cached = self.load_raw("epa_aqs_monitors_all")
        if cached is not None:
            return cached
        logger.warning("EPA_AQS: no cached monitor metadata found. Run fetch() first.")
        return pd.DataFrame(columns=list(self.metadata_columns))
