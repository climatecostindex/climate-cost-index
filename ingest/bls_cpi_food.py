"""Fetch regional/metro Food at Home CPI indices from BLS.

Source: BLS Consumer Price Index — Food
API URL: https://api.bls.gov/publicAPI/v2/timeseries/data/
Bulk area codes: https://download.bls.gov/pub/time.series/cu/cu.area
Series: CPI-U Food at Home by region and metro area
Format: JSON (API)
API key: Optional via BLS_API_KEY in .env

Required for CCI-Expanded. Food is excluded from CCI-Core but included
in CCI-Expanded per SSRN Section 4.2.

Fetches raw regional/metro CPI food indices ONLY. Does NOT compute
county-level allocations, climate attribution, rolling averages, trend
decomposition, or inflation adjustments — those belong in transform.

Year-over-year percentage change is a standard CPI reporting convention
and is acceptable to compute here.

Output columns: area_code, area_name, geo_type, year, food_cpi_index,
                food_cpi_yoy_change, period_type
Confidence: B (regional/metro, not county)
Attribution: none (general cost proxy, not climate-attributed in v1)
Rate limit: 25 requests/day (v1 unregistered), 500/day (v2 registered)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from ingest.base import BaseIngester, RETRYABLE_STATUS_CODES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BLS CPI API configuration
# ---------------------------------------------------------------------------

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_AREA_CODE_URL = "https://download.bls.gov/pub/time.series/cu/cu.area"

# Series ID template: CUUR{area_code}SAF11 = CPI-U, not seasonally adjusted,
# Food at Home.
# Note: SAF11 = "Food at home" per BLS item code taxonomy.
# SA0L1 = "All items less food and energy" — a common confusion.
SERIES_PREFIX = "CUUR"
SERIES_SUFFIX = "SAF11"

# Census region area codes
NATIONAL_AREA_CODE = "0000"
REGION_AREA_CODES = {
    "0100": "Northeast",
    "0200": "Midwest",
    "0300": "South",
    "0400": "West",
}

# v2 API limits (with key)
V2_MAX_SERIES_PER_BATCH = 50
V2_MAX_YEARS_PER_QUERY = 20

# v1 API limits (without key)
V1_MAX_SERIES_PER_BATCH = 25
V1_MAX_YEARS_PER_QUERY = 10

# Rate limit: conservative for both API tiers
BLS_CPI_CALLS_PER_SECOND = 0.5  # 2 req/sec max, stay polite

# Minimum history depth (years)
MIN_HISTORY_YEARS = 12

# Default start year — BLS CPI metro coverage varies, 2010 is safe minimum
DEFAULT_START_YEAR = 2010

# Current year for default end year
DEFAULT_END_YEAR = 2025

# BLS User-Agent (needed for bulk downloads)
BLS_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# BLS bulk download requires Accept header to avoid 403
BLS_BULK_HEADERS = {
    "User-Agent": BLS_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _build_series_id(area_code: str) -> str:
    """Build a BLS CPI series ID for Food at Home.

    Args:
        area_code: BLS area code (e.g., "0000", "0100", "S12A").

    Returns:
        Full series ID string.
    """
    return f"{SERIES_PREFIX}{area_code}{SERIES_SUFFIX}"


def _classify_geo_type(area_code: str) -> str:
    """Classify a BLS area code as national, region, or metro.

    Args:
        area_code: BLS area code.

    Returns:
        One of "national", "region", or "metro".
    """
    if area_code == NATIONAL_AREA_CODE:
        return "national"
    if area_code in REGION_AREA_CODES:
        return "region"
    return "metro"


class BLSCPIFoodIngester(BaseIngester):
    """Ingest BLS CPI Food at Home indices by region and metro area.

    Downloads Food at Home CPI-U index values (not seasonally adjusted) for:
    - National average
    - 4 Census regions (Northeast, Midwest, South, West)
    - Available metro areas (discovered from BLS area code reference)

    Computes annual averages from monthly data (preferring BLS-published M13
    values) and standard year-over-year percentage change.
    """

    source_name = "bls_cpi_food"
    confidence = "B"
    attribution = "none"
    calls_per_second = BLS_CPI_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "area_code": str,
        "area_name": str,
        "geo_type": str,
        "year": int,
        "food_cpi_index": float,
        "food_cpi_yoy_change": float,
        "period_type": str,
    }

    def _get_api_key(self) -> str | None:
        """Return the BLS API key from environment, or None."""
        return os.environ.get("BLS_API_KEY") or None

    def _api_limits(self) -> tuple[int, int]:
        """Return (max_series_per_batch, max_years_per_query) based on API key.

        Returns:
            Tuple of (series_limit, year_limit).
        """
        if self._get_api_key():
            return V2_MAX_SERIES_PER_BATCH, V2_MAX_YEARS_PER_QUERY
        logger.warning(
            "BLS_CPI_FOOD: no BLS_API_KEY set — using v1 API with lower limits "
            "(25 series/batch, 10 years/query, 25 queries/day)"
        )
        return V1_MAX_SERIES_PER_BATCH, V1_MAX_YEARS_PER_QUERY

    # -- Area code discovery ---------------------------------------------------

    def _fetch_area_codes(self) -> dict[str, str]:
        """Fetch and parse BLS CPI area codes from the bulk reference file.

        Downloads ``cu.area`` from the BLS bulk download site. Filters for
        area types that publish CPI data (types A = metro, 01-04 = region,
        0000 = national).

        Returns:
            Dict mapping area_code → area_name for all CPI-relevant areas.
        """
        cached_path = self.cache_dir() / "cu_area.txt"

        if cached_path.exists():
            logger.info("BLS_CPI_FOOD: loading cached area codes from %s", cached_path)
            raw_text = cached_path.read_text()
        else:
            try:
                logger.info("BLS_CPI_FOOD: downloading area codes from %s", BLS_AREA_CODE_URL)
                resp = self.api_get(
                    BLS_AREA_CODE_URL,
                    headers=BLS_BULK_HEADERS,
                )
                raw_text = resp.text
                cached_path.write_text(raw_text)
            except Exception:
                logger.warning(
                    "BLS_CPI_FOOD: failed to download area code reference file. "
                    "Falling back to region-only codes.",
                    exc_info=True,
                )
                return self._fallback_area_codes()

        return self._parse_area_codes(raw_text)

    def _parse_area_codes(self, raw_text: str) -> dict[str, str]:
        """Parse the BLS cu.area tab-separated file.

        The file may have two formats:
        1. With area_type column:
           area_type\\tarea_code\\tarea_text\\tdisplay_level\\tselectable\\tsort_sequence
        2. Without area_type column:
           area_code\\tarea_name\\tdisplay_level\\tselectable\\tsort_sequence

        We want national (0000), regions (0100-0400), and metro areas
        (codes starting with A or S followed by digits, e.g. S12A, A104).

        Args:
            raw_text: Raw text content of cu.area.

        Returns:
            Dict mapping area_code → area_name.
        """
        areas: dict[str, str] = {}

        lines = raw_text.splitlines()
        if not lines:
            return self._fallback_area_codes()

        # Detect format from header
        header = lines[0].strip().lower()
        has_area_type_col = header.startswith("area_type")

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            if has_area_type_col:
                if len(parts) < 3:
                    continue
                area_type = parts[0].strip()
                area_code = parts[1].strip()
                area_name = parts[2].strip()
            else:
                if len(parts) < 2:
                    continue
                area_code = parts[0].strip()
                area_name = parts[1].strip()
                area_type = ""

            # National
            if area_code == NATIONAL_AREA_CODE:
                areas[area_code] = area_name
                continue

            # Regions (4 Census regions only, not sub-regions)
            if area_code in REGION_AREA_CODES:
                areas[area_code] = area_name
                continue

            # Metro areas — codes like S12A, S35B, A104, A210 etc.
            # Exclude size-class codes (S000, S100, N000, D000, etc.)
            # and sub-region codes (0110, 0230, etc.)
            if has_area_type_col:
                if area_type == "A":
                    areas[area_code] = area_name
            else:
                # Without area_type column, identify metros by code pattern:
                # - A-prefix codes are all metros (A104, A210, A311, etc.)
                # - S-prefix codes ending with a letter are metros (S12A, S35B)
                # - S/N/D-prefix codes ending with digits are size classes
                is_metro = False
                if area_code.startswith("A") and len(area_code) >= 3 and area_code[1].isdigit():
                    is_metro = True
                elif area_code.startswith("S") and len(area_code) >= 3 and area_code[-1].isalpha():
                    is_metro = True
                if is_metro:
                    areas[area_code] = area_name

        # Ensure national and all regions are present even if file format
        # differs slightly
        if NATIONAL_AREA_CODE not in areas:
            areas[NATIONAL_AREA_CODE] = "U.S. city average"
        for code, name in REGION_AREA_CODES.items():
            if code not in areas:
                areas[code] = name

        logger.info(
            "BLS_CPI_FOOD: found %d areas (%d metro, 4 regions, 1 national)",
            len(areas),
            len(areas) - 5,
        )

        return areas

    def _fallback_area_codes(self) -> dict[str, str]:
        """Return minimal area codes when the reference file is unavailable.

        Covers national + 4 Census regions. Metro areas are missing but
        regional coverage is sufficient for CCI computation.

        Returns:
            Dict mapping area_code → area_name.
        """
        areas = {NATIONAL_AREA_CODE: "U.S. city average"}
        areas.update(REGION_AREA_CODES)
        logger.warning(
            "BLS_CPI_FOOD: using fallback area codes (national + 4 regions only). "
            "Metro area data will not be available."
        )
        return areas

    # -- BLS API queries -------------------------------------------------------

    def _api_post(
        self,
        url: str,
        json_body: dict[str, Any],
    ) -> dict[str, Any]:
        """POST to BLS API with rate limiting and retry.

        Args:
            url: API endpoint URL.
            json_body: Request JSON body.

        Returns:
            Parsed JSON response dict.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
            ValueError: If the BLS API returns a non-SUCCESS status.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.rate_limit()
            try:
                resp = self.client.post(url, json=json_body, timeout=60.0)
                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    wait = self.retry_backoff_base ** attempt
                    logger.warning(
                        "BLS_CPI_FOOD: HTTP %d (attempt %d/%d), retrying in %.1fs",
                        resp.status_code,
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise
                wait = self.retry_backoff_base ** attempt
                logger.warning(
                    "BLS_CPI_FOOD: request error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    def _query_series_batch(
        self,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> dict[str, Any]:
        """Query a batch of BLS CPI series.

        Args:
            series_ids: List of BLS series IDs.
            start_year: Start year (inclusive).
            end_year: End year (inclusive).

        Returns:
            Parsed JSON response from BLS API.
        """
        body: dict[str, Any] = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
            "annualaverage": True,
        }

        api_key = self._get_api_key()
        if api_key:
            body["registrationkey"] = api_key

        logger.info(
            "BLS_CPI_FOOD: querying %d series for %d-%d",
            len(series_ids),
            start_year,
            end_year,
        )

        return self._api_post(BLS_API_URL, body)

    def _query_all_series(
        self,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Query all series, batching and splitting year ranges as needed.

        Args:
            series_ids: All series IDs to query.
            start_year: Start year (inclusive).
            end_year: End year (inclusive).

        Returns:
            List of series data dicts from all successful batches.
        """
        max_series, max_years = self._api_limits()

        # Split year range into chunks
        year_ranges: list[tuple[int, int]] = []
        y = start_year
        while y <= end_year:
            chunk_end = min(y + max_years - 1, end_year)
            year_ranges.append((y, chunk_end))
            y = chunk_end + 1

        # Split series into batches
        series_batches: list[list[str]] = []
        for i in range(0, len(series_ids), max_series):
            series_batches.append(series_ids[i : i + max_series])

        all_series_data: list[dict[str, Any]] = []

        for s_batch in series_batches:
            for yr_start, yr_end in year_ranges:
                try:
                    resp_data = self._query_series_batch(s_batch, yr_start, yr_end)
                    status = resp_data.get("status", "")
                    if status != "REQUEST_SUCCEEDED":
                        logger.warning(
                            "BLS_CPI_FOOD: API returned status '%s' for batch "
                            "(series %d-%d, years %d-%d). Message: %s",
                            status,
                            0,
                            len(s_batch),
                            yr_start,
                            yr_end,
                            resp_data.get("message", []),
                        )
                        # Still try to parse — partial data may be present
                    results = resp_data.get("Results", {})
                    series_list = results.get("series", [])
                    all_series_data.extend(series_list)
                except Exception:
                    logger.warning(
                        "BLS_CPI_FOOD: batch query failed (years %d-%d, %d series). "
                        "Caching data from other batches.",
                        yr_start,
                        yr_end,
                        len(s_batch),
                        exc_info=True,
                    )

        return all_series_data

    # -- Data parsing ----------------------------------------------------------

    def _parse_series_data(
        self,
        series_list: list[dict[str, Any]],
        area_map: dict[str, str],
    ) -> pd.DataFrame:
        """Parse BLS API series response into a DataFrame of annual averages.

        For each series-year, prefers the BLS-published annual average
        (period "M13") if available. Falls back to computing the mean of
        available monthly values.

        Args:
            series_list: List of series dicts from BLS API response.
            area_map: Mapping of area_code → area_name.

        Returns:
            DataFrame with annual average CPI values per area.
        """
        records: list[dict[str, Any]] = []

        for series in series_list:
            series_id = series.get("seriesID", "")
            # Extract area code from series ID: CUUR{area_code}SAF11
            area_code = self._extract_area_code(series_id)
            if area_code is None:
                logger.warning(
                    "BLS_CPI_FOOD: could not parse area code from series '%s', skipping",
                    series_id,
                )
                continue

            area_name = area_map.get(area_code, f"Unknown ({area_code})")
            geo_type = _classify_geo_type(area_code)

            observations = series.get("data", [])
            if not observations:
                continue

            # Group observations by year
            monthly_by_year: dict[int, list[float]] = {}
            m13_by_year: dict[int, float] = {}

            for obs in observations:
                year = int(obs.get("year", 0))
                period = obs.get("period", "")
                value_str = obs.get("value", "")

                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    continue

                if period == "M13":
                    m13_by_year[year] = value
                elif period.startswith("M") and period[1:].isdigit():
                    month_num = int(period[1:])
                    if 1 <= month_num <= 12:
                        monthly_by_year.setdefault(year, []).append(value)

            # Build annual records
            all_years = sorted(set(list(m13_by_year.keys()) + list(monthly_by_year.keys())))

            for year in all_years:
                if year in m13_by_year:
                    annual_avg = m13_by_year[year]
                    period_type = "M13"
                elif year in monthly_by_year and monthly_by_year[year]:
                    months = monthly_by_year[year]
                    annual_avg = sum(months) / len(months)
                    period_type = "computed"
                    if len(months) < 12:
                        logger.info(
                            "BLS_CPI_FOOD: %s (%s) year %d — computed annual average "
                            "from %d months (incomplete year)",
                            area_code,
                            area_name,
                            year,
                            len(months),
                        )
                else:
                    continue

                records.append({
                    "area_code": area_code,
                    "area_name": area_name,
                    "geo_type": geo_type,
                    "year": year,
                    "food_cpi_index": annual_avg,
                    "period_type": period_type,
                })

        if not records:
            return pd.DataFrame(columns=list(self.required_columns))

        df = pd.DataFrame(records)

        # Compute year-over-year change per area
        df = df.sort_values(["area_code", "year"]).reset_index(drop=True)
        df["food_cpi_yoy_change"] = np.nan

        for area_code in df["area_code"].unique():
            mask = df["area_code"] == area_code
            area_df = df.loc[mask].sort_values("year")
            indices = area_df.index

            for i in range(1, len(indices)):
                curr_idx = indices[i]
                prev_idx = indices[i - 1]
                prev_val = df.loc[prev_idx, "food_cpi_index"]
                curr_val = df.loc[curr_idx, "food_cpi_index"]
                if prev_val and prev_val != 0:
                    df.loc[curr_idx, "food_cpi_yoy_change"] = (
                        (curr_val - prev_val) / prev_val * 100.0
                    )

        # Enforce dtypes
        df["area_code"] = df["area_code"].astype(str)
        df["area_name"] = df["area_name"].astype(str)
        df["geo_type"] = df["geo_type"].astype(str)
        df["year"] = df["year"].astype(int)
        df["food_cpi_index"] = df["food_cpi_index"].astype(float)
        df["food_cpi_yoy_change"] = df["food_cpi_yoy_change"].astype(float)
        df["period_type"] = df["period_type"].astype(str)

        return df

    @staticmethod
    def _extract_area_code(series_id: str) -> str | None:
        """Extract the BLS area code from a CPI series ID.

        Series format: CUUR{area_code}SAF11
        Prefix is "CUUR" (4 chars), suffix is "SAF11" (5 chars).

        Args:
            series_id: Full BLS series ID.

        Returns:
            Area code string, or None if format doesn't match.
        """
        prefix = SERIES_PREFIX  # "CUUR"
        suffix = SERIES_SUFFIX  # "SAF11"

        if not series_id.startswith(prefix) or not series_id.endswith(suffix):
            return None

        area_code = series_id[len(prefix) : -len(suffix)]
        return area_code if area_code else None

    # -- Main fetch ------------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch BLS CPI Food at Home data for all areas and years.

        Args:
            years: If provided, return only these years. If None, fetch
                   DEFAULT_START_YEAR through DEFAULT_END_YEAR.

        Returns:
            DataFrame with columns: area_code, area_name, geo_type, year,
            food_cpi_index, food_cpi_yoy_change, period_type.
        """
        # Determine year range
        if years:
            start_year = min(years)
            end_year = max(years)
            # Need one extra prior year for YoY computation
            query_start = start_year - 1
        else:
            query_start = DEFAULT_START_YEAR
            end_year = DEFAULT_END_YEAR
            start_year = DEFAULT_START_YEAR

        # Discover area codes
        area_map = self._fetch_area_codes()

        # Build series IDs
        series_ids = [_build_series_id(code) for code in sorted(area_map.keys())]
        logger.info(
            "BLS_CPI_FOOD: built %d series IDs for %d areas",
            len(series_ids),
            len(area_map),
        )

        # Query BLS API
        series_data = self._query_all_series(series_ids, query_start, end_year)

        if not series_data:
            logger.warning("BLS_CPI_FOOD: no data returned from BLS API")
            return pd.DataFrame(columns=list(self.required_columns))

        # Parse into DataFrame
        df = self._parse_series_data(series_data, area_map)

        if df.empty:
            logger.warning("BLS_CPI_FOOD: no records parsed from API response")
            return pd.DataFrame(columns=list(self.required_columns))

        # Filter to requested years (drop the extra prior year used for YoY)
        if years:
            df = df[df["year"].isin(years)].copy()
        else:
            df = df[df["year"] >= start_year].copy()

        if df.empty:
            logger.warning("BLS_CPI_FOOD: no records match requested years")
            return pd.DataFrame(columns=list(self.required_columns))

        df = df.reset_index(drop=True)

        # Cache
        year_min, year_max = int(df["year"].min()), int(df["year"].max())
        self.cache_raw(
            df,
            label="bls_cpi_food_all",
            data_vintage=f"BLS CPI Food at Home {year_min} to {year_max}",
        )

        logger.info(
            "BLS_CPI_FOOD: fetched %d records — %d areas, %d years (%d-%d)",
            len(df),
            df["area_code"].nunique(),
            df["year"].nunique(),
            year_min,
            year_max,
        )

        return df
