"""Fetch storm event records from NCEI Storm Events Database.

Source: NCEI Storm Events Database
URL: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
Format: Bulk CSV download — annual gzipped files

Fetches raw event records ONLY. Does NOT compute severity scores, tier
classifications, or county aggregations — those belong in
transform/event_severity_tiers.py and transform/storm_severity.py.

CRITICAL: Preserve raw event_type strings exactly as NCEI provides.
Do NOT remap, clean, or normalize event type names during ingestion.

Output columns: event_id, fips, date, event_type, property_damage,
               crop_damage, injuries_direct, deaths_direct, magnitude,
               begin_lat, begin_lon
Confidence: B
Attribution: proxy
Minimum history: 15 years
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date

import numpy as np
import pandas as pd

from pathlib import Path

from ingest.base import BaseIngester
from ingest.utils import fips_5digit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NCEI Storm Events configuration
# ---------------------------------------------------------------------------

# Base URL for the bulk CSV file directory
NCEI_STORMS_BASE_URL = (
    "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
)

# Filename patterns — data year is embedded as d{YEAR}, compilation date varies.
# Example: StormEvents_details-ftp_v1.0_d2023_c20260116.csv.gz
NCEI_DETAILS_PATTERN = re.compile(
    r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz"
)
NCEI_LOCATIONS_PATTERN = re.compile(
    r"StormEvents_locations-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz"
)

# Rate limit: static file downloads, be polite (1 req/sec)
NCEI_CALLS_PER_SECOND = 1.0

# Minimum trailing history depth (years)
NCEI_MIN_HISTORY_YEARS = 15

# Damage value suffix multipliers
DAMAGE_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
}

# NWS forecast zone → county FIPS crosswalk
ZONE_CROSSWALK_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "nws_zones" / "zone_county_crosswalk.parquet"

# Damage fields that must be split when distributing zone events across counties
_DAMAGE_FIELDS = ("property_damage", "crop_damage")
_COUNT_FIELDS = ("injuries_direct", "deaths_direct")

# Columns to extract from the details CSV
DETAILS_USECOLS = [
    "EVENT_ID",
    "STATE_FIPS",
    "CZ_TYPE",
    "CZ_FIPS",
    "BEGIN_YEARMONTH",
    "BEGIN_DAY",
    "EVENT_TYPE",
    "DAMAGE_PROPERTY",
    "DAMAGE_CROPS",
    "INJURIES_DIRECT",
    "DEATHS_DIRECT",
    "MAGNITUDE",
]

# Columns to extract from the locations CSV
LOCATIONS_USECOLS = ["EVENT_ID", "LATITUDE", "LONGITUDE"]


class NCEIStormsIngester(BaseIngester):
    """Ingest storm event records from NCEI bulk CSV files.

    Downloads annual StormEvents_details and StormEvents_locations
    gzipped CSV files from NCEI. Parses event records, normalizes
    FIPS codes for county-type records, and joins location coordinates.

    CRITICAL: Preserves raw event_type strings exactly as NCEI provides.
    Do NOT remap, clean, or normalize event type names.
    """

    source_name = "ncei_storms"
    confidence = "B"
    attribution = "proxy"
    calls_per_second = NCEI_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "event_id": str,
        "fips": str,
        "date": object,  # Python date objects
        "event_type": str,
        "property_damage": float,
        "crop_damage": float,
        "injuries_direct": int,
        "deaths_direct": int,
        "magnitude": float,
        "begin_lat": float,
        "begin_lon": float,
    }

    # -- Directory listing & file discovery ------------------------------------

    def _get_file_listing(self) -> str:
        """Fetch the NCEI CSV directory listing HTML.

        Returns:
            HTML content of the directory listing page.
        """
        logger.info("NCEI_STORMS: fetching directory listing from %s", NCEI_STORMS_BASE_URL)
        resp = self.api_get(NCEI_STORMS_BASE_URL)
        return resp.text

    def _find_files_for_year(
        self, listing_html: str, year: int
    ) -> tuple[str | None, str | None]:
        """Find details and locations filenames for a given data year.

        If multiple files exist for the same year (different compilation
        dates), the one with the latest compilation date is returned.

        Args:
            listing_html: HTML content of the directory listing.
            year: Data year to find files for.

        Returns:
            Tuple of (details_filename, locations_filename). Either may
            be None if not found in the listing.
        """
        details_file = None
        details_compile = ""
        for match in NCEI_DETAILS_PATTERN.finditer(listing_html):
            if int(match.group(1)) == year:
                if match.group(2) > details_compile:
                    details_file = match.group(0)
                    details_compile = match.group(2)

        locations_file = None
        locations_compile = ""
        for match in NCEI_LOCATIONS_PATTERN.finditer(listing_html):
            if int(match.group(1)) == year:
                if match.group(2) > locations_compile:
                    locations_file = match.group(0)
                    locations_compile = match.group(2)

        return details_file, locations_file

    # -- Download & parse ------------------------------------------------------

    def _download_csv(self, filename: str) -> bytes:
        """Download a gzipped CSV file from NCEI.

        Args:
            filename: Filename within the NCEI csvfiles directory.

        Returns:
            Raw bytes of the gzipped CSV file.
        """
        url = NCEI_STORMS_BASE_URL + filename
        logger.info("NCEI_STORMS: downloading %s", url)
        resp = self.api_get(url)
        return resp.content

    def _parse_details(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse the StormEvents_details gzipped CSV.

        Reads only the columns needed for our output schema. All columns
        are read as strings to avoid type-coercion surprises.

        Args:
            raw_bytes: Raw gzipped CSV content.

        Returns:
            DataFrame with DETAILS_USECOLS columns, all string dtype.
        """
        return pd.read_csv(
            io.BytesIO(raw_bytes),
            compression="gzip",
            usecols=DETAILS_USECOLS,
            dtype=str,
            low_memory=False,
        )

    def _parse_locations(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse the StormEvents_locations gzipped CSV.

        If multiple location records exist for one event, keeps only
        the first (events like tornadoes may have multiple track points).

        Args:
            raw_bytes: Raw gzipped CSV content.

        Returns:
            DataFrame with EVENT_ID, LATITUDE, LONGITUDE — one row per event.
        """
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            compression="gzip",
            usecols=LOCATIONS_USECOLS,
            dtype={"EVENT_ID": str},
            low_memory=False,
        )
        return df.drop_duplicates(subset="EVENT_ID", keep="first")

    # -- Value parsing ---------------------------------------------------------

    @staticmethod
    def _parse_damage(val: object) -> float:
        """Parse NCEI damage shorthand string to numeric dollars.

        Handles: "25K" → 25000.0, "1.5M" → 1500000.0,
        "0.5B" → 500000000.0, "0" → 0.0, "" → 0.0, NaN → 0.0.

        Args:
            val: Raw damage cell value.

        Returns:
            Numeric damage in dollars.
        """
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0
        s = str(val).strip()
        if not s or s.lower() == "nan":
            return 0.0
        # Check for K/M/B suffix
        suffix = s[-1].upper()
        if suffix in DAMAGE_MULTIPLIERS:
            try:
                return float(s[:-1]) * DAMAGE_MULTIPLIERS[suffix]
            except (ValueError, TypeError):
                return 0.0
        # Plain numeric
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    # -- Zone-to-county resolution ---------------------------------------------

    def _load_zone_crosswalk(self) -> pd.DataFrame | None:
        """Load the NWS zone-to-county FIPS crosswalk.

        Returns:
            DataFrame with columns zone_id, fips, area_fraction.
            None if the crosswalk file is not available.
        """
        if not ZONE_CROSSWALK_PATH.exists():
            logger.warning(
                "NCEI_STORMS: zone crosswalk not found at %s — "
                "zone events (CZ_TYPE=Z) will be dropped",
                ZONE_CROSSWALK_PATH,
            )
            return None
        return pd.read_parquet(ZONE_CROSSWALK_PATH)

    def _resolve_zone_events(
        self, df: pd.DataFrame, crosswalk: pd.DataFrame,
    ) -> pd.DataFrame:
        """Resolve zone events to county FIPS using the crosswalk.

        For zones mapping to a single county, assigns the county FIPS directly.
        For zones mapping to multiple counties, duplicates the event row for
        each county and splits damage/casualties proportionally by area.

        Args:
            df: DataFrame with zone events (fips is NaN, zone_id is set).
            crosswalk: Zone-to-county crosswalk with zone_id, fips, area_fraction.

        Returns:
            DataFrame with zone events resolved to county-level FIPS codes.
            Event IDs are suffixed with _z{N} for multi-county splits.
        """
        if df.empty:
            return df

        # Join zone events to crosswalk
        zone_events = df.drop(columns=["fips"]).merge(
            crosswalk, on="zone_id", how="inner",
        )

        if zone_events.empty:
            logger.warning(
                "NCEI_STORMS: no zone events matched the crosswalk "
                "(%d events had zone_id values not in crosswalk)",
                len(df),
            )
            return pd.DataFrame(columns=df.columns)

        unmatched = set(df["zone_id"]) - set(crosswalk["zone_id"])
        if unmatched:
            logger.info(
                "NCEI_STORMS: %d unique zone_ids not in crosswalk (dropped)",
                len(unmatched),
            )

        # Split damage and casualty fields proportionally by area_fraction
        for col in _DAMAGE_FIELDS + _COUNT_FIELDS:
            if col in zone_events.columns:
                zone_events[col] = zone_events[col] * zone_events["area_fraction"]

        # For multi-county zones, suffix event_id to maintain uniqueness
        counts = zone_events.groupby("event_id").cumcount()
        multi_mask = zone_events.duplicated(subset="event_id", keep=False)
        zone_events.loc[multi_mask, "event_id"] = (
            zone_events.loc[multi_mask, "event_id"]
            + "_z"
            + counts[multi_mask].astype(str)
        )

        zone_events = zone_events.drop(columns=["zone_id", "area_fraction"])

        logger.info(
            "NCEI_STORMS: resolved %d zone events → %d county-level rows",
            len(df), len(zone_events),
        )

        return zone_events

    # -- Row-level transforms --------------------------------------------------

    def _process_year(
        self, details_bytes: bytes, locations_bytes: bytes | None,
        zone_crosswalk: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Process one year of storm data into the output schema.

        Parses details and locations CSVs, joins them on EVENT_ID,
        normalizes FIPS codes, resolves forecast-zone events to county
        FIPS via the NWS crosswalk, parses damage strings, and builds dates.

        Args:
            details_bytes: Raw gzipped details CSV.
            locations_bytes: Raw gzipped locations CSV, or None.
            zone_crosswalk: NWS zone-to-county crosswalk, or None.

        Returns:
            DataFrame conforming to required_columns.
        """
        details = self._parse_details(details_bytes)

        # Parse and join locations
        if locations_bytes is not None:
            locations = self._parse_locations(locations_bytes)
        else:
            locations = pd.DataFrame(columns=LOCATIONS_USECOLS)
        df = details.merge(locations, on="EVENT_ID", how="left")

        # -- event_id --
        df["event_id"] = df["EVENT_ID"].astype(str)

        # -- FIPS normalization --
        cz_type = df["CZ_TYPE"].str.strip().str.upper()
        state_fips = pd.to_numeric(df["STATE_FIPS"], errors="coerce")
        cz_fips = pd.to_numeric(df["CZ_FIPS"], errors="coerce")

        county_mask = (cz_type == "C") & state_fips.notna() & cz_fips.notna()
        zone_mask = (cz_type == "Z") & state_fips.notna() & cz_fips.notna()
        marine_mask = cz_type == "M"

        fips_series = pd.Series(pd.NA, index=df.index, dtype="string")
        if county_mask.any():
            fips_series[county_mask] = (
                state_fips[county_mask].astype(int).map(lambda x: f"{x:02d}")
                + cz_fips[county_mask].astype(int).map(lambda x: f"{x:03d}")
            )
        df["fips"] = fips_series

        # Build zone_id for zone events (STATE_FIPS + CZ_FIPS as 5-char string)
        zone_id_series = pd.Series(pd.NA, index=df.index, dtype="string")
        if zone_mask.any():
            zone_id_series[zone_mask] = (
                state_fips[zone_mask].astype(int).map(lambda x: f"{x:02d}")
                + cz_fips[zone_mask].astype(int).map(lambda x: f"{x:03d}")
            )
        df["zone_id"] = zone_id_series

        n_zone = zone_mask.sum()
        n_marine = marine_mask.sum()
        if n_marine > 0:
            logger.info(
                "NCEI_STORMS: %d marine-zone records (CZ_TYPE=M) excluded",
                n_marine,
            )

        # -- date --
        ym = df["BEGIN_YEARMONTH"].astype(str)
        year_col = pd.to_numeric(ym.str[:4], errors="coerce")
        month_col = pd.to_numeric(ym.str[4:6], errors="coerce")
        day_col = pd.to_numeric(df["BEGIN_DAY"], errors="coerce")

        date_df = pd.DataFrame({
            "year": year_col, "month": month_col, "day": day_col,
        })
        df["date"] = pd.to_datetime(date_df, errors="coerce").dt.date

        # -- event_type — PRESERVED EXACTLY --
        df["event_type"] = df["EVENT_TYPE"].astype(str)

        # -- damage parsing --
        df["property_damage"] = df["DAMAGE_PROPERTY"].apply(self._parse_damage)
        df["crop_damage"] = df["DAMAGE_CROPS"].apply(self._parse_damage)

        # -- numeric fields --
        df["injuries_direct"] = (
            pd.to_numeric(df["INJURIES_DIRECT"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["deaths_direct"] = (
            pd.to_numeric(df["DEATHS_DIRECT"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["magnitude"] = pd.to_numeric(df["MAGNITUDE"], errors="coerce").astype(float)

        # -- coordinates from locations file --
        df["begin_lat"] = pd.to_numeric(df["LATITUDE"], errors="coerce").astype(float)
        df["begin_lon"] = pd.to_numeric(df["LONGITUDE"], errors="coerce").astype(float)

        # -- Resolve zone events to county FIPS --
        output_cols = list(self.required_columns)

        county_events = df[df["fips"].notna()][output_cols].copy()

        if zone_crosswalk is not None and n_zone > 0:
            zone_events = df[df["zone_id"].notna()].copy()
            resolved = self._resolve_zone_events(zone_events, zone_crosswalk)
            if not resolved.empty:
                resolved = resolved[output_cols].copy()
                return pd.concat([county_events, resolved], ignore_index=True)
        elif n_zone > 0:
            logger.info(
                "NCEI_STORMS: %d zone events dropped (no crosswalk available)",
                n_zone,
            )

        return county_events

    # -- Main fetch ------------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch NCEI Storm Events data.

        Downloads details and locations CSV files for each year,
        parses them, and merges into a single DataFrame.

        Args:
            years: If provided, fetch only these years. If None,
                   fetch trailing 15 years from current year.

        Returns:
            DataFrame with columns: event_id, fips, date, event_type,
            property_damage, crop_damage, injuries_direct, deaths_direct,
            magnitude, begin_lat, begin_lon.
        """
        if years is None:
            current_year = date.today().year
            years = list(range(
                current_year - NCEI_MIN_HISTORY_YEARS, current_year + 1
            ))

        # Load zone-to-county crosswalk (once for all years)
        zone_crosswalk = self._load_zone_crosswalk()
        if zone_crosswalk is not None:
            logger.info(
                "NCEI_STORMS: loaded zone crosswalk (%d mappings, %d zones)",
                len(zone_crosswalk), zone_crosswalk["zone_id"].nunique(),
            )

        # Fetch directory listing to discover exact filenames
        try:
            listing_html = self._get_file_listing()
        except Exception:
            logger.error("NCEI_STORMS: failed to fetch directory listing")
            raise

        frames: list[pd.DataFrame] = []

        for year in sorted(years):
            details_file, locations_file = self._find_files_for_year(
                listing_html, year
            )

            if details_file is None:
                logger.warning(
                    "NCEI_STORMS: no details file found for year %d, skipping", year
                )
                continue

            try:
                details_bytes = self._download_csv(details_file)

                locations_bytes = None
                if locations_file is not None:
                    try:
                        locations_bytes = self._download_csv(locations_file)
                    except Exception:
                        logger.warning(
                            "NCEI_STORMS: failed to download locations for %d, "
                            "proceeding without coordinates",
                            year,
                            exc_info=True,
                        )

                frame = self._process_year(
                    details_bytes, locations_bytes, zone_crosswalk,
                )
                frames.append(frame)

                self.cache_raw(
                    frame,
                    label=f"ncei_storms_{year}",
                    data_vintage=f"NCEI Storm Events {year}",
                )

                logger.info(
                    "NCEI_STORMS: parsed %d events for year %d", len(frame), year
                )

            except Exception:
                logger.warning(
                    "NCEI_STORMS: failed to process year %d, skipping",
                    year,
                    exc_info=True,
                )

        if not frames:
            logger.warning("NCEI_STORMS: no data parsed from any year")
            return pd.DataFrame(columns=list(self.required_columns))

        df = pd.concat(frames, ignore_index=True)

        # Cache combined
        year_min, year_max = min(years), max(years)
        self.cache_raw(
            df,
            label="ncei_storms_all",
            data_vintage=f"NCEI Storm Events {year_min} to {year_max}",
        )

        logger.info(
            "NCEI_STORMS: total %d events across %d years",
            len(df),
            len(frames),
        )

        return df
