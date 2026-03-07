"""Fetch daily weather station observations from NOAA GHCN-Daily.

Source: NOAA GHCN-Daily (Global Historical Climatology Network — Daily)
Primary path: Bulk download from https://www.ncei.noaa.gov/pub/data/ghcn/daily/
Fallback path: CDO API at https://www.ncei.noaa.gov/cdo-web/api/v2/ (backup only)

Uses the recommended by-year CSV approach to avoid downloading the
full ~3GB ghcnd_all.tar.gz archive. Downloads per-year CSV.gz files
from https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/.

Fetches raw daily observations ONLY. Does NOT compute degree days, HDD,
CDD, temperature anomalies, extreme heat day counts, station-to-county
mappings, monthly/annual averages, or ANY derived metrics — those belong
in the transform layer.

Three output tables:
  - Daily observations: station_id, date, tmax, tmin, prcp, q_flag_*
  - Station metadata: station_id, lat, lon, elevation, state, name
  - Climate normals: station_id, month, normal_tmax, normal_tmin

Confidence: A
Attribution: proxy
Minimum history: 15 years
"""

from __future__ import annotations

import io
import logging
import tarfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.base import BaseIngester
from ingest.utils import download_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOAA GHCN-Daily configuration
# ---------------------------------------------------------------------------

# Base URL for GHCN-Daily bulk downloads
GHCN_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"

# By-year CSV files URL
GHCN_BY_YEAR_URL = f"{GHCN_BASE_URL}/by_year"

# Station metadata URL
GHCN_STATIONS_URL = f"{GHCN_BASE_URL}/ghcnd-stations.txt"

# 1991-2020 climate normals — bulk archive (primary) and per-station (fallback)
NORMALS_ARCHIVE_URL = (
    "https://www.ncei.noaa.gov/data/normals-monthly/1991-2020/archive/"
    "us-climate-normals_1991-2020_v1.0.1_monthly_multivariate_by-station_c20230404.tar.gz"
)
NORMALS_ACCESS_URL = (
    "https://www.ncei.noaa.gov/data/normals-monthly/1991-2020/access"
)

# Rate limit for static file downloads (be polite)
GHCN_CALLS_PER_SECOND = 1.0

# Minimum trailing history depth (years)
GHCN_MIN_HISTORY_YEARS = 15

# Elements to extract from daily data
GHCN_ELEMENTS = frozenset({"TMAX", "TMIN", "PRCP"})

# GHCN missing value sentinel
GHCN_MISSING_VALUE = -9999

# Unit conversion factor: GHCN stores values in tenths
GHCN_UNIT_DIVISOR = 10.0

# By-year CSV column names (source files have no header)
BY_YEAR_COLUMNS = [
    "station_id", "date_str", "element", "value",
    "m_flag", "q_flag", "s_flag", "obs_time",
]

# Station metadata fixed-width column specs (from GHCN docs)
# ID: chars 1-11, Lat: 13-20, Lon: 22-30, Elev: 32-37, State: 39-40, Name: 42-71
STATIONS_COLSPECS = [
    (0, 11),   # station_id (11 chars)
    (12, 20),  # latitude   (8 chars)
    (21, 30),  # longitude  (9 chars)
    (31, 37),  # elevation  (6 chars)
    (38, 40),  # state      (2 chars)
    (41, 71),  # name       (30 chars)
]
STATIONS_COLNAMES = ["station_id", "lat", "lon", "elevation", "state", "name"]

# Normals CSV column names from NCEI normals-monthly product
NORMALS_TMAX_COL = "MLY-TMAX-NORMAL"
NORMALS_TMIN_COL = "MLY-TMIN-NORMAL"


class NOAANCEIIngester(BaseIngester):
    """Ingest daily weather station observations from NOAA GHCN-Daily.

    Downloads by-year CSV files from GHCN-Daily, station metadata from
    fixed-width files, and 1991-2020 climate normals. Produces three
    output tables cached to data/raw/noaa_ncei/.

    Uses the recommended by-year CSV approach to avoid downloading the
    full ~3GB ghcnd_all.tar.gz archive.
    """

    source_name = "noaa_ncei"
    confidence = "A"
    attribution = "proxy"
    calls_per_second = GHCN_CALLS_PER_SECOND

    # Primary output schema: daily observations
    required_columns: dict[str, type] = {
        "station_id": str,
        "date": object,  # Python date objects
        "tmax": float,
        "tmin": float,
        "prcp": float,
        "q_flag_tmax": str,
        "q_flag_tmin": str,
        "q_flag_prcp": str,
    }

    # Secondary output: station metadata
    station_metadata_columns: dict[str, type] = {
        "station_id": str,
        "lat": float,
        "lon": float,
        "elevation": float,
        "state": str,
        "name": str,
    }

    # Secondary output: climate normals
    normals_columns: dict[str, type] = {
        "station_id": str,
        "month": int,
        "normal_tmax": float,
        "normal_tmin": float,
    }

    # -- Station metadata ------------------------------------------------------

    def _download_stations(self) -> pd.DataFrame:
        """Download and parse ghcnd-stations.txt (fixed-width format).

        Returns:
            DataFrame with station_id, lat, lon, elevation, state, name.
        """
        logger.info("NOAA_NCEI: downloading station metadata from %s", GHCN_STATIONS_URL)
        resp = self.api_get(GHCN_STATIONS_URL)

        df = pd.read_fwf(
            io.StringIO(resp.text),
            colspecs=STATIONS_COLSPECS,
            names=STATIONS_COLNAMES,
            dtype=str,
        )

        df["station_id"] = df["station_id"].str.strip()
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")
        df["state"] = df["state"].fillna("").str.strip()
        df["name"] = df["name"].fillna("").str.strip()

        logger.info("NOAA_NCEI: parsed %d stations total", len(df))
        return df

    def _filter_us_stations(self, stations: pd.DataFrame) -> pd.DataFrame:
        """Filter to U.S. stations only.

        U.S. stations have IDs starting with "US" or have a valid
        2-letter U.S. state code in the state column.

        Args:
            stations: Full station metadata DataFrame.

        Returns:
            Filtered DataFrame containing only U.S. stations.
        """
        us_id_mask = stations["station_id"].str.startswith("US")
        has_state = stations["state"].str.len() == 2
        filtered = stations[us_id_mask | has_state].copy()

        logger.info(
            "NOAA_NCEI: filtered to %d U.S. stations (from %d total)",
            len(filtered), len(stations),
        )
        return filtered

    # -- Daily observations (by-year CSVs) -------------------------------------

    def _download_year_csv(self, year: int) -> Path:
        """Download a by-year CSV.gz file from GHCN-Daily to local cache.

        Uses the download_file utility which skips if already downloaded
        and streams to disk with a progress bar.

        Args:
            year: Data year to download.

        Returns:
            Path to the cached .csv.gz file.
        """
        url = f"{GHCN_BY_YEAR_URL}/{year}.csv.gz"
        dest = self.cache_dir() / f"ghcn_daily_{year}.csv.gz"
        return download_file(url, dest)

    def _parse_year_csv(
        self, gz_path: Path, us_station_ids: set[str],
    ) -> pd.DataFrame:
        """Parse a by-year CSV.gz file into the output schema.

        Reads in chunks to manage memory, filters to U.S. stations and
        target elements, pivots elements into columns, converts units,
        and handles missing values.

        Args:
            gz_path: Path to the cached .csv.gz file.
            us_station_ids: Set of U.S. station IDs for filtering.

        Returns:
            DataFrame with daily observation columns.
        """
        reader = pd.read_csv(
            gz_path,
            compression="gzip",
            header=None,
            names=BY_YEAR_COLUMNS,
            dtype=str,
            chunksize=500_000,
        )

        filtered_chunks: list[pd.DataFrame] = []
        for chunk in reader:
            mask = (
                chunk["station_id"].isin(us_station_ids)
                & chunk["element"].isin(GHCN_ELEMENTS)
            )
            if mask.any():
                filtered_chunks.append(chunk[mask])

        if not filtered_chunks:
            return pd.DataFrame(columns=list(self.required_columns))

        df = pd.concat(filtered_chunks, ignore_index=True)

        if df.empty:
            return pd.DataFrame(columns=list(self.required_columns))

        # Convert value to numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Handle missing value sentinel (-9999 → NaN)
        df.loc[df["value"] == GHCN_MISSING_VALUE, "value"] = np.nan

        # Convert units: tenths → standard (°C for temp, mm for precip)
        df["value"] = df["value"] / GHCN_UNIT_DIVISOR

        # Fill missing quality flags with empty string
        df["q_flag"] = df["q_flag"].fillna("")

        # Split by element, deduplicate, and merge into wide format
        tmax = (
            df[df["element"] == "TMAX"][["station_id", "date_str", "value", "q_flag"]]
            .drop_duplicates(subset=["station_id", "date_str"], keep="first")
            .rename(columns={"value": "tmax", "q_flag": "q_flag_tmax"})
        )
        tmin = (
            df[df["element"] == "TMIN"][["station_id", "date_str", "value", "q_flag"]]
            .drop_duplicates(subset=["station_id", "date_str"], keep="first")
            .rename(columns={"value": "tmin", "q_flag": "q_flag_tmin"})
        )
        prcp = (
            df[df["element"] == "PRCP"][["station_id", "date_str", "value", "q_flag"]]
            .drop_duplicates(subset=["station_id", "date_str"], keep="first")
            .rename(columns={"value": "prcp", "q_flag": "q_flag_prcp"})
        )

        # Merge on station_id + date (outer join for partial observations)
        result = tmax.merge(tmin, on=["station_id", "date_str"], how="outer")
        result = result.merge(prcp, on=["station_id", "date_str"], how="outer")

        # Ensure float types for value columns
        for col in ["tmax", "tmin", "prcp"]:
            result[col] = result[col].astype(float)

        # Fill missing quality flags with empty string
        for col in ["q_flag_tmax", "q_flag_tmin", "q_flag_prcp"]:
            result[col] = result[col].fillna("")

        # Parse date from YYYYMMDD string
        result["date"] = pd.to_datetime(
            result["date_str"], format="%Y%m%d", errors="coerce"
        ).dt.date

        result = result.drop(columns=["date_str"])
        return result[list(self.required_columns)].copy()

    # -- Climate normals -------------------------------------------------------

    def _parse_normals_csv(
        self, csv_content: str, station_id: str,
    ) -> pd.DataFrame | None:
        """Parse a single station's normals CSV.

        Extracts monthly TMAX and TMIN normals and converts from °F to °C
        for consistency with the daily observations.

        Args:
            csv_content: Raw CSV text content.
            station_id: Station ID for labeling rows.

        Returns:
            DataFrame with normals columns, or None if parsing fails.
        """
        try:
            stn_df = pd.read_csv(io.StringIO(csv_content))
        except Exception:
            return None

        if NORMALS_TMAX_COL not in stn_df.columns or NORMALS_TMIN_COL not in stn_df.columns:
            return None

        # Extract month — the normals-monthly CSV has a lowercase "month"
        # column with values "01".."12", and a "DATE" column with the same.
        # Use the explicit month column first, then DATE as integer fallback.
        if "month" in stn_df.columns:
            months = pd.to_numeric(stn_df["month"], errors="coerce").astype("Int64")
        elif "DATE" in stn_df.columns:
            months = pd.to_numeric(stn_df["DATE"], errors="coerce").astype("Int64")
        else:
            months = pd.Series(range(1, len(stn_df) + 1))

        tmax_f = pd.to_numeric(stn_df[NORMALS_TMAX_COL], errors="coerce")
        tmin_f = pd.to_numeric(stn_df[NORMALS_TMIN_COL], errors="coerce")

        # NCEI normals-monthly product reports in °F; convert to °C
        normal_tmax = (tmax_f - 32.0) * 5.0 / 9.0
        normal_tmin = (tmin_f - 32.0) * 5.0 / 9.0

        return pd.DataFrame({
            "station_id": station_id,
            "month": months.values,
            "normal_tmax": normal_tmax.values,
            "normal_tmin": normal_tmin.values,
        })

    def _download_normals_archive(
        self, us_station_ids: set[str],
    ) -> pd.DataFrame:
        """Download normals via the bulk archive (~30MB tar.gz).

        Downloads the full NCEI normals-monthly archive and extracts
        CSVs for U.S. stations that are in our station set.

        Args:
            us_station_ids: Set of station IDs to extract normals for.

        Returns:
            DataFrame with normals columns.

        Raises:
            Exception: If the archive download or parsing fails.
        """
        dest = self.cache_dir() / "normals_archive.tar.gz"
        download_file(NORMALS_ARCHIVE_URL, dest)

        logger.info("NOAA_NCEI: extracting normals from archive (%s)", dest)
        frames: list[pd.DataFrame] = []
        fetched = 0

        with tarfile.open(dest, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".csv"):
                    continue
                # Extract station ID from path (e.g. ".../USW00094728.csv")
                basename = Path(member.name).stem
                if basename not in us_station_ids:
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                csv_content = f.read().decode("utf-8")
                normals = self._parse_normals_csv(csv_content, basename)
                if normals is not None and not normals.empty:
                    frames.append(normals)
                    fetched += 1

        logger.info("NOAA_NCEI: extracted normals for %d stations from archive", fetched)

        if not frames:
            return pd.DataFrame(columns=list(self.normals_columns))

        return pd.concat(frames, ignore_index=True)

    def _download_normals_per_station(
        self, station_ids: set[str],
    ) -> pd.DataFrame:
        """Download normals one station at a time (fallback).

        Only used if the bulk archive download fails. Limited to
        stations that actually appear in the daily data.

        Args:
            station_ids: Station IDs to fetch normals for.

        Returns:
            DataFrame with normals columns.
        """
        frames: list[pd.DataFrame] = []
        fetched = 0
        skipped = 0

        for station_id in sorted(station_ids):
            url = f"{NORMALS_ACCESS_URL}/{station_id}.csv"
            try:
                resp = self.api_get(url)
                normals = self._parse_normals_csv(resp.text, station_id)
                if normals is not None and not normals.empty:
                    frames.append(normals)
                    fetched += 1
            except Exception:
                skipped += 1
                continue

            if fetched % 500 == 0 and fetched > 0:
                logger.info(
                    "NOAA_NCEI: normals progress — %d fetched, %d skipped",
                    fetched, skipped,
                )

        logger.info(
            "NOAA_NCEI: normals complete — %d fetched, %d skipped",
            fetched, skipped,
        )

        if not frames:
            return pd.DataFrame(columns=list(self.normals_columns))

        return pd.concat(frames, ignore_index=True)

    def _download_normals(self, us_station_ids: set[str]) -> pd.DataFrame:
        """Download 1991-2020 climate normals for U.S. stations.

        Primary: downloads the bulk archive (~30MB) and extracts all
        U.S. station normals in one pass.
        Fallback: per-station CSV download if the archive fails.

        Args:
            us_station_ids: Set of U.S. station IDs to fetch normals for.

        Returns:
            DataFrame with station_id, month, normal_tmax, normal_tmin.
        """
        # Try bulk archive first (30MB vs 85,000 individual requests)
        try:
            return self._download_normals_archive(us_station_ids)
        except Exception:
            logger.warning(
                "NOAA_NCEI: bulk normals archive failed, falling back to per-station",
                exc_info=True,
            )

        # Fallback: per-station download
        return self._download_normals_per_station(us_station_ids)

    # -- Main fetch ------------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch NOAA GHCN-Daily data.

        Downloads station metadata, by-year daily CSVs, and climate
        normals. Caches all three output tables to data/raw/noaa_ncei/.
        Returns the daily observations DataFrame as the primary output.

        Args:
            years: Years to fetch. Defaults to trailing 15 years.

        Returns:
            Daily observations DataFrame matching required_columns.
        """
        if years is None:
            current_year = date.today().year
            years = list(range(
                current_year - GHCN_MIN_HISTORY_YEARS, current_year + 1
            ))

        # Step 1: Download and cache station metadata
        all_stations = self._download_stations()
        us_stations = self._filter_us_stations(all_stations)
        us_station_ids = set(us_stations["station_id"].values)

        station_meta = us_stations[list(self.station_metadata_columns)].copy()
        self.cache_raw(
            station_meta,
            label="noaa_ncei_stations",
            data_vintage="GHCN-Daily station metadata",
        )

        # Step 2: Download and parse daily observations by year
        frames: list[pd.DataFrame] = []

        for year in sorted(years):
            try:
                gz_path = self._download_year_csv(year)
                frame = self._parse_year_csv(gz_path, us_station_ids)

                if not frame.empty:
                    frames.append(frame)
                    self.cache_raw(
                        frame,
                        label=f"noaa_ncei_{year}",
                        data_vintage=f"GHCN-Daily observations {year}",
                    )
                    logger.info(
                        "NOAA_NCEI: parsed %d observations for year %d",
                        len(frame), year,
                    )
                else:
                    logger.warning(
                        "NOAA_NCEI: no U.S. observations for year %d", year,
                    )

            except Exception:
                logger.warning(
                    "NOAA_NCEI: failed to fetch/parse year %d, skipping",
                    year,
                    exc_info=True,
                )

        # Step 3: Download and cache climate normals
        # Only fetch normals for stations that appeared in daily data
        observed_station_ids = set()
        for frame in frames:
            observed_station_ids.update(frame["station_id"].unique())
        logger.info(
            "NOAA_NCEI: fetching normals for %d stations with daily data "
            "(from %d total US stations)",
            len(observed_station_ids), len(us_station_ids),
        )
        try:
            normals = self._download_normals(observed_station_ids)
            if not normals.empty:
                self.cache_raw(
                    normals,
                    label="noaa_ncei_normals",
                    data_vintage="1991-2020 monthly climate normals",
                )
                logger.info(
                    "NOAA_NCEI: cached normals for %d station-months",
                    len(normals),
                )
        except Exception:
            logger.warning(
                "NOAA_NCEI: failed to fetch climate normals",
                exc_info=True,
            )

        # Combine all years
        if not frames:
            logger.warning("NOAA_NCEI: no daily data parsed from any year")
            return pd.DataFrame(columns=list(self.required_columns))

        df = pd.concat(frames, ignore_index=True)

        # Cache combined observations
        year_min, year_max = min(years), max(years)
        self.cache_raw(
            df,
            label="noaa_ncei_observations_all",
            data_vintage=f"GHCN-Daily observations {year_min} to {year_max}",
        )

        logger.info(
            "NOAA_NCEI: total %d observations, %d stations across %d years",
            len(df), df["station_id"].nunique(), len(frames),
        )

        return df

    def fetch_station_metadata(self) -> pd.DataFrame:
        """Load cached station metadata.

        Returns:
            DataFrame with station metadata columns.
        """
        cached = self.load_raw("noaa_ncei_stations")
        if cached is not None:
            return cached
        logger.warning("NOAA_NCEI: no cached station metadata. Run fetch() first.")
        return pd.DataFrame(columns=list(self.station_metadata_columns))

    def fetch_normals(self) -> pd.DataFrame:
        """Load cached climate normals.

        Returns:
            DataFrame with normals columns.
        """
        cached = self.load_raw("noaa_ncei_normals")
        if cached is not None:
            return cached
        logger.warning("NOAA_NCEI: no cached normals. Run fetch() first.")
        return pd.DataFrame(columns=list(self.normals_columns))
