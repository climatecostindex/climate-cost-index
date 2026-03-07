"""Fetch FEMA Individual Assistance disaster declarations from OpenFEMA.

Source: OpenFEMA API
URL: https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries
Format: JSON responses with $skip/$top pagination

This ingester fetches IA-declared disaster records at county level from the
DisasterDeclarationsSummaries endpoint, which is the only OpenFEMA dataset
with clean county FIPS codes. Each record represents a county designated
for Individual Assistance under a specific disaster.

NOTE on ia_amount and registrant_count:
    The DisasterDeclarationsSummaries endpoint does NOT contain IA payout
    dollar amounts or registrant counts. Those are available from the
    HousingAssistanceOwners/Renters endpoints (at city/zip level, not
    county level, and without FIPS codes). The transform layer
    (storm_severity.py) should enrich these fields by cross-referencing
    with HousingAssistance data.

Output columns: disaster_number, fips, year, ia_amount, registrant_count,
               disaster_type, declaration_date
Confidence: B
Attribution: proxy
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ingest.base import BaseIngester
from ingest.utils import fips_5digit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenFEMA API configuration
# ---------------------------------------------------------------------------
FEMA_DECLARATIONS_URL = (
    "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
)

# Filter to disasters where Individual/Household programs were declared.
# ihProgramDeclared is the current field; iaProgramDeclared is the legacy field.
FEMA_IA_FILTER = "ihProgramDeclared eq true or iaProgramDeclared eq true"

# Fields to request from the API
FEMA_SELECT_FIELDS = (
    "disasterNumber,fipsStateCode,fipsCountyCode,declarationDate,"
    "incidentType,state,designatedArea"
)

# Pagination settings
FEMA_PAGE_SIZE = 1000

# Rate limit: no documented limit, but be polite
FEMA_CALLS_PER_SECOND = 1.0


class FEMAIAIngester(BaseIngester):
    """Ingest FEMA Individual Assistance disaster declarations at county level.

    Paginates through DisasterDeclarationsSummaries filtered to IA-declared
    disasters. Each record represents a county designated for IA under a
    specific disaster declaration.

    Note: ia_amount and registrant_count are NaN in this ingester's output.
    They require enrichment from HousingAssistanceOwners/Renters in the
    transform layer, as those datasets lack county FIPS codes.
    """

    source_name = "fema_ia"
    confidence = "B"
    attribution = "proxy"
    calls_per_second = FEMA_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "disaster_number": str,
        "fips": str,
        "year": int,
        "ia_amount": float,
        "registrant_count": float,  # float to accommodate NaN
        "disaster_type": str,
        "declaration_date": object,  # Python date objects
    }

    def _fetch_all_pages(self) -> pd.DataFrame:
        """Paginate through the full IA declarations dataset.

        Uses $skip/$top pagination. Logs progress and handles partial
        failures by returning whatever was successfully fetched.
        """
        all_records: list[dict] = []
        skip = 0
        total_expected: int | None = None

        while True:
            params: dict[str, str] = {
                "$top": str(FEMA_PAGE_SIZE),
                "$skip": str(skip),
                "$filter": FEMA_IA_FILTER,
                "$select": FEMA_SELECT_FIELDS,
                "$inlinecount": "allpages",
            }

            try:
                resp = self.api_get(FEMA_DECLARATIONS_URL, params=params)
                data = resp.json()
            except Exception:
                logger.warning(
                    "FEMA_IA: pagination failed at skip=%d, stopping",
                    skip,
                    exc_info=True,
                )
                break

            # Extract metadata count on first page
            meta = data.get("metadata", {})
            if total_expected is None:
                total_expected = meta.get("count", 0)
                logger.info("FEMA_IA: total records expected: %s", total_expected)

            records = data.get("DisasterDeclarationsSummaries", [])
            if not records:
                break

            all_records.extend(records)
            skip += FEMA_PAGE_SIZE

            logger.info(
                "FEMA_IA: fetched %d / %s records",
                len(all_records),
                total_expected or "?",
            )

            if len(records) < FEMA_PAGE_SIZE:
                break  # Last page

        if not all_records:
            return pd.DataFrame(columns=list(self.required_columns))

        return self._parse_records(all_records)

    def _parse_records(self, records: list[dict]) -> pd.DataFrame:
        """Parse raw API records into the output schema."""
        df = pd.DataFrame(records)

        # Build 5-digit FIPS from state + county codes
        valid_fips_mask = (
            df["fipsStateCode"].notna()
            & df["fipsCountyCode"].notna()
            & (df["fipsCountyCode"] != "000")
        )

        # Initialize as object dtype so it can hold both strings and NaN
        fips_series = pd.Series([np.nan] * len(df), dtype="object")
        fips_series[valid_fips_mask] = df.loc[valid_fips_mask].apply(
            lambda r: fips_5digit(r["fipsStateCode"], r["fipsCountyCode"]),
            axis=1,
        )
        # Convert to StringDtype so is_string_dtype() passes validation
        # (pd.NA replaces np.nan for missing values in string columns)
        df["fips"] = fips_series.astype("string")

        # Log malformed FIPS
        n_bad_fips = (~valid_fips_mask).sum()
        if n_bad_fips > 0:
            logger.warning(
                "FEMA_IA: %d records with missing/malformed FIPS codes (stored as NaN)",
                n_bad_fips,
            )

        # Parse declaration date
        df["declaration_date"] = pd.to_datetime(df["declarationDate"]).dt.date

        # Extract year from declaration date
        df["year"] = pd.to_datetime(df["declarationDate"]).dt.year

        # Rename columns
        df["disaster_number"] = df["disasterNumber"].astype(str)
        df["disaster_type"] = df["incidentType"].fillna("Unknown")

        # IA amounts and registrant counts are not available from this endpoint
        df["ia_amount"] = np.nan
        df["registrant_count"] = np.nan

        # Keep only output schema columns
        return df[list(self.required_columns)].copy()

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch FEMA IA disaster declarations for all counties.

        Downloads the full history of IA-declared disasters. The ``years``
        parameter filters the result after fetching (the API filter is on
        IA declaration, not year).

        Args:
            years: If provided, filter to declarations in these years only.
                   If None, return all available history.

        Returns:
            DataFrame with columns: disaster_number, fips, year, ia_amount,
            registrant_count, disaster_type, declaration_date.
        """
        df = self._fetch_all_pages()

        if df.empty:
            return df

        # Filter to requested years if specified
        if years is not None:
            df = df[df["year"].isin(years)].copy()
            if df.empty:
                logger.warning("FEMA_IA: no records match requested years %s", years)
                return pd.DataFrame(columns=list(self.required_columns))

        # Cache by year for partial-failure resilience
        for yr in sorted(df["year"].unique()):
            year_df = df[df["year"] == yr]
            self.cache_raw(
                year_df,
                label=f"fema_ia_{yr}",
                data_vintage=f"FEMA IA declarations {yr}",
            )

        # Cache combined file
        year_min, year_max = df["year"].min(), df["year"].max()
        self.cache_raw(
            df,
            label="fema_ia_all",
            data_vintage=f"FEMA IA declarations {year_min} to {year_max}",
        )

        return df
