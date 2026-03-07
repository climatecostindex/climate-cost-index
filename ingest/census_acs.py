"""Fetch county-level housing, population, and income data from Census ACS.

Source: Census Bureau American Community Survey (ACS) 5-Year Estimates
URL: https://api.census.gov/data/{year}/acs/acs5
Format: JSON array (first row = headers, remaining rows = data)

Fetches raw Census estimates ONLY. Does NOT compute per-capita rates,
per-housing-unit rates, income adjustments, percentile ranks, or any
derived metrics — those belong in the transform and score layers.

Output columns: fips, year, total_housing_units, owner_occupied_units,
               population, median_household_income
Confidence: A
Attribution: none (reference data, not a scored component)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from ingest.base import BaseIngester
from ingest.utils import fips_5digit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Census ACS API configuration
# ---------------------------------------------------------------------------
CENSUS_ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"

# ACS variable codes → human-readable output column names
ACS_VARIABLES = {
    "B25001_001E": "total_housing_units",
    "B25003_002E": "owner_occupied_units",
    "B01003_001E": "population",
    "B19013_001E": "median_household_income",
}

# Comma-separated variable list for the API request
_ACS_GET_FIELDS = ",".join(ACS_VARIABLES.keys())

# Census suppression sentinel value
CENSUS_SUPPRESSION_VALUE = "-666666666"

# Default year range: 5 most recent ACS vintages
# ACS 5-year estimates have ~2 year lag, so most recent is typically current_year - 2
DEFAULT_HISTORY_YEARS = 15

# Conservative rate limit — 500 req/day without key
ACS_CALLS_PER_SECOND = 0.5


class CensusACSIngester(BaseIngester):
    """Ingest ACS 5-year estimates for all U.S. counties.

    Queries the Census Bureau API for housing units, population, and
    median household income at county level. One request per year fetches
    all counties (~3,222 rows).
    """

    source_name = "census_acs"
    confidence = "A"
    attribution = "none"
    calls_per_second = ACS_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "fips": str,
        "year": int,
        "total_housing_units": float,
        "owner_occupied_units": float,
        "population": float,
        "median_household_income": float,
    }

    def _default_years(self) -> list[int]:
        """Return the most recent ACS vintage years available.

        ACS 5-year estimates lag ~2 years behind the current date.
        Downloads DEFAULT_HISTORY_YEARS vintages.
        """
        latest = datetime.now().year - 2
        return list(range(latest - DEFAULT_HISTORY_YEARS + 1, latest + 1))

    def _get_api_key(self) -> str | None:
        """Return the Census API key from environment, or None."""
        key = os.getenv("CENSUS_API_KEY", "")
        return key if key else None

    def _fetch_year(self, year: int) -> pd.DataFrame:
        """Fetch ACS data for a single vintage year (all counties)."""
        url = CENSUS_ACS_BASE_URL.format(year=year)
        params: dict[str, str] = {
            "get": _ACS_GET_FIELDS,
            "for": "county:*",
        }
        api_key = self._get_api_key()
        if api_key:
            params["key"] = api_key

        resp = self.api_get(url, params=params)
        data = resp.json()

        if not data or len(data) < 2:
            logger.warning("CENSUS_ACS: empty response for year %d", year)
            return pd.DataFrame(columns=list(self.required_columns))

        return self._parse_response(data, year)

    def _parse_response(self, data: list[list[str]], year: int) -> pd.DataFrame:
        """Parse the Census JSON array response into the output schema.

        Args:
            data: JSON array where data[0] is headers, data[1:] is rows.
            year: The ACS vintage year.
        """
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)

        # Build 5-digit FIPS from state + county columns
        df["fips"] = df.apply(
            lambda r: fips_5digit(r["state"], r["county"]), axis=1
        )
        df["year"] = year

        # Rename ACS variable codes to human-readable names
        df = df.rename(columns=ACS_VARIABLES)

        # Convert numeric columns: replace Census suppression values with NaN
        for col in ACS_VARIABLES.values():
            df[col] = df[col].replace(CENSUS_SUPPRESSION_VALUE, np.nan)
            df[col] = df[col].replace("null", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Keep only output schema columns
        return df[list(self.required_columns)].copy()

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch county-level ACS data for the given vintage years.

        One API call per year retrieves all ~3,222 counties.
        Partial failures are logged and skipped.

        Args:
            years: ACS vintage years to fetch. Defaults to 5 most recent.

        Returns:
            DataFrame with columns: fips, year, total_housing_units,
            owner_occupied_units, population, median_household_income.
        """
        if years is None:
            years = self._default_years()

        all_frames: list[pd.DataFrame] = []
        failed_years: list[int] = []

        for year in years:
            try:
                df = self._fetch_year(year)
                if not df.empty:
                    all_frames.append(df)
                    self.cache_raw(
                        df,
                        label=f"census_acs_{year}",
                        data_vintage=f"ACS 5-year {year}",
                    )
                    logger.info(
                        "CENSUS_ACS: cached %d rows for vintage %d", len(df), year
                    )
                else:
                    logger.warning("CENSUS_ACS: no data for vintage %d", year)
            except Exception:
                logger.warning(
                    "CENSUS_ACS: failed to fetch vintage %d, skipping",
                    year,
                    exc_info=True,
                )
                failed_years.append(year)

        if failed_years:
            logger.warning(
                "CENSUS_ACS: %d vintage years failed: %s",
                len(failed_years),
                failed_years,
            )

        if not all_frames:
            logger.error("CENSUS_ACS: no data retrieved for any year")
            return pd.DataFrame(columns=list(self.required_columns))

        result = pd.concat(all_frames, ignore_index=True)

        # Write combined file with metadata
        year_range = f"{min(years)} to {max(years)}"
        self.cache_raw(result, label="census_acs_all", data_vintage=year_range)

        return result
