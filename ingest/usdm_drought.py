"""Fetch weekly drought classifications from U.S. Drought Monitor.

Source: U.S. Drought Monitor
URL: https://droughtmonitor.unl.edu/DmData/DataDownload/
API: https://usdmdataservices.unl.edu/api/CountyStatistics/
Format: Weekly JSON/CSV with county-level classification

Fetches raw weekly classifications ONLY. Does NOT compute drought scores,
severity-area integrals, annual aggregations, or any derived metrics — those
belong in transform/drought_scoring.py.

Output columns: fips, date, d0_pct, d1_pct, d2_pct, d3_pct, d4_pct, none_pct
Confidence: A
Attribution: proxy
Minimum history: 12+ years (full available preferred)
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ingest.base import BaseIngester
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# USDM API configuration
# ---------------------------------------------------------------------------
USDM_API_BASE = (
    "https://usdmdataservices.unl.edu/api/CountyStatistics"
    "/GetDroughtSeverityStatisticsByAreaPercent"
)

# Categorical format (statisticsType=2): D0-D4 are non-overlapping categories.
# None + D0 + D1 + D2 + D3 + D4 ≈ 100%.
USDM_STATISTICS_TYPE = "2"

# USDM data begins January 4, 2000
USDM_START_YEAR = 2000

# Tolerance for percentage-sum validation (±2%)
PCT_SUM_TOLERANCE = 2.0

# All 50 US states + DC (used to iterate state-by-state requests)
US_STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY",
]

# Column rename mapping: USDM API (camelCase) → output schema
_API_COL_MAP = {
    "none": "none_pct",
    "d0": "d0_pct",
    "d1": "d1_pct",
    "d2": "d2_pct",
    "d3": "d3_pct",
    "d4": "d4_pct",
}

# Percentage columns in output schema
_PCT_COLS = ["d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "none_pct"]


class USDMDroughtIngester(BaseIngester):
    """Ingest weekly county-level drought classifications from USDM.

    Fetches categorical drought percentages (D0-D4 + None) from the USDM
    REST API, state by state, year by year. The API has a 1-year maximum
    date range per request.

    At 1 req/sec across 51 state codes, a full history fetch (~24 years)
    takes roughly 20 minutes.
    """

    source_name = "usdm"
    confidence = "A"
    attribution = "proxy"
    calls_per_second = 1.0  # No documented rate limit; be polite

    required_columns: dict[str, type] = {
        "fips": str,
        "date": object,  # Python date objects → pandas object dtype
        "d0_pct": float,
        "d1_pct": float,
        "d2_pct": float,
        "d3_pct": float,
        "d4_pct": float,
        "none_pct": float,
    }

    def _default_years(self) -> list[int]:
        """Return year range from USDM inception through current year."""
        return list(range(USDM_START_YEAR, datetime.now().year + 1))

    def _fetch_state_year(self, state: str, year: int) -> pd.DataFrame:
        """Fetch one state's county drought data for a single year."""
        params = {
            "aoi": state,
            "startdate": f"1/1/{year}",
            "enddate": f"12/31/{year}",
            "statisticsType": USDM_STATISTICS_TYPE,
        }
        resp = self.api_get(
            USDM_API_BASE,
            params=params,
            headers={"Accept": "application/json"},
        )
        records = resp.json()
        if not records:
            return pd.DataFrame(columns=list(self.required_columns))

        return self._parse_response(pd.DataFrame(records))

    def _parse_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw API response into the output schema."""
        df = df.rename(columns=_API_COL_MAP)

        # Parse mapDate (ISO datetime string e.g. "2024-02-27T00:00:00") → Python date
        df["date"] = pd.to_datetime(df["mapDate"]).dt.date

        # Normalize FIPS to 5-digit zero-padded string
        df["fips"] = df["fips"].apply(normalize_fips)

        # Coerce percentage columns to float
        for col in _PCT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Validate percentage sums
        self._check_percentage_sums(df)

        # Return ONLY the output schema columns
        return df[list(self.required_columns)].copy()

    @staticmethod
    def _check_percentage_sums(df: pd.DataFrame) -> None:
        """Warn if drought category percentages don't sum to ~100%."""
        row_sums = df[_PCT_COLS].sum(axis=1)
        bad_mask = (row_sums - 100.0).abs() > PCT_SUM_TOLERANCE
        n_bad = bad_mask.sum()
        if n_bad > 0:
            sample_fips = df.loc[bad_mask, "fips"].head(3).tolist()
            logger.warning(
                "USDM: %d rows have percentage sums outside ±%.1f%% of 100%%. "
                "Sample FIPS: %s",
                n_bad,
                PCT_SUM_TOLERANCE,
                sample_fips,
            )

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch county-level drought data for the given years.

        Downloads state-by-state, year-by-year (API max range = 1 year).
        If a state/year request fails, it is logged and skipped — data
        fetched successfully is still cached and returned.

        Args:
            years: List of years to fetch. Defaults to 2000–present.

        Returns:
            DataFrame with columns: fips, date, d0_pct–d4_pct, none_pct.
        """
        if years is None:
            years = self._default_years()

        all_frames: list[pd.DataFrame] = []
        failed_requests: list[tuple[str, int]] = []

        for year in years:
            year_frames: list[pd.DataFrame] = []

            for state in US_STATE_CODES:
                try:
                    df = self._fetch_state_year(state, year)
                    if not df.empty:
                        year_frames.append(df)
                except Exception:
                    logger.warning(
                        "USDM: failed to fetch %s/%d, skipping",
                        state,
                        year,
                        exc_info=True,
                    )
                    failed_requests.append((state, year))

            if year_frames:
                year_df = pd.concat(year_frames, ignore_index=True)
                all_frames.append(year_df)
                # Cache each year for partial-failure resilience
                self.cache_raw(
                    year_df,
                    label=f"usdm_{year}",
                    data_vintage=f"{year}-01-01 to {year}-12-31",
                )
                logger.info("USDM: cached %d rows for year %d", len(year_df), year)
            else:
                logger.warning("USDM: no data retrieved for year %d", year)

        if failed_requests:
            logger.warning(
                "USDM: %d state-year requests failed: %s",
                len(failed_requests),
                failed_requests[:10],
            )

        if not all_frames:
            logger.error("USDM: no data retrieved for any year")
            return pd.DataFrame(columns=list(self.required_columns))

        result = pd.concat(all_frames, ignore_index=True)
        # Deduplicate in case overlapping state boundaries produce duplicates
        result = result.drop_duplicates(subset=["fips", "date"])

        # Write combined file with full-range metadata
        year_range = f"{min(years)} to {max(years)}"
        self.cache_raw(result, label="usdm_all", data_vintage=year_range)

        return result
