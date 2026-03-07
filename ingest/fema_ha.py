"""Fetch FEMA Housing Assistance payout data from OpenFEMA.

Source: OpenFEMA API
URLs:
    - https://www.fema.gov/api/open/v2/HousingAssistanceOwners
    - https://www.fema.gov/api/open/v2/HousingAssistanceRenters

Format: JSON responses with $skip/$top pagination

This ingester fetches IA payout amounts by disaster, state, and county from
the HousingAssistanceOwners and HousingAssistanceRenters endpoints. Records
are aggregated to (disasterNumber, state, county) grain before caching.

The existing fema_ia.py ingester provides disaster declarations with FIPS
codes but no dollar amounts. This ingester provides dollar amounts but
identifies counties by NAME (e.g., "Barbour (County)"). The transform layer
(storm_severity.py) merges the two using a county name → FIPS mapping.

Output columns: disaster_number, state, county, ia_amount, registrant_count
Confidence: B
Attribution: proxy
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenFEMA API configuration
# ---------------------------------------------------------------------------
FEMA_HA_OWNERS_URL = (
    "https://www.fema.gov/api/open/v2/HousingAssistanceOwners"
)
FEMA_HA_RENTERS_URL = (
    "https://www.fema.gov/api/open/v2/HousingAssistanceRenters"
)

# Fields to request from the API
FEMA_HA_SELECT_FIELDS = (
    "disasterNumber,state,county,totalApprovedIhpAmount,validRegistrations"
)

# Pagination settings
FEMA_HA_PAGE_SIZE = 1000

# Inter-page delay in seconds — FEMA throttles (503) at high $skip offsets
FEMA_HA_PAGE_DELAY = 2.0

# Progress logging interval (every N records)
FEMA_HA_LOG_INTERVAL = 10_000

# Rate limit: be polite to the federal API
FEMA_HA_CALLS_PER_SECOND = 0.5

# Endpoint entity name mapping (used as JSON data key in responses)
FEMA_HA_ENTITY_NAMES = {
    FEMA_HA_OWNERS_URL: "HousingAssistanceOwners",
    FEMA_HA_RENTERS_URL: "HousingAssistanceRenters",
}


class FEMAHAIngester(BaseIngester):
    """Ingest FEMA Housing Assistance payout data at county-disaster level.

    Paginates through both HousingAssistanceOwners and HousingAssistanceRenters
    endpoints, aggregates to (disasterNumber, state, county) grain, and caches
    the combined result.

    County names are preserved exactly as the API provides — FIPS mapping
    belongs in the transform layer.
    """

    source_name = "fema_ha"
    confidence = "B"
    attribution = "proxy"
    calls_per_second = FEMA_HA_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "disaster_number": str,
        "state": str,
        "county": str,
        "ia_amount": float,
        "registrant_count": float,
    }

    def _fetch_endpoint(self, url: str) -> list[dict]:
        """Paginate through a single HA endpoint, returning raw records.

        Args:
            url: The OpenFEMA endpoint URL.

        Returns:
            List of raw record dicts from the API.
        """
        entity_name = FEMA_HA_ENTITY_NAMES[url]
        all_records: list[dict] = []
        skip = 0
        total_expected: int | None = None
        last_logged_at = 0

        while True:
            params: dict[str, str] = {
                "$top": str(FEMA_HA_PAGE_SIZE),
                "$skip": str(skip),
                "$select": FEMA_HA_SELECT_FIELDS,
                "$inlinecount": "allpages",
            }

            try:
                resp = self.api_get(url, params=params)
                data = resp.json()
            except Exception:
                logger.warning(
                    "FEMA_HA: %s pagination failed at skip=%d, "
                    "keeping %d records fetched so far",
                    entity_name,
                    skip,
                    len(all_records),
                    exc_info=True,
                )
                break

            # Extract metadata count on first page
            meta = data.get("metadata", {})
            if total_expected is None:
                total_expected = meta.get("count", 0)
                logger.info(
                    "FEMA_HA: total %s records expected: %s",
                    entity_name,
                    total_expected,
                )

            records = data.get(entity_name, [])
            if not records:
                break

            all_records.extend(records)
            skip += FEMA_HA_PAGE_SIZE

            # Log progress at intervals
            if len(all_records) - last_logged_at >= FEMA_HA_LOG_INTERVAL:
                logger.info(
                    "FEMA_HA: fetched %d / %s %s records",
                    len(all_records),
                    total_expected or "?",
                    entity_name,
                )
                last_logged_at = len(all_records)

            if len(records) < FEMA_HA_PAGE_SIZE:
                break  # Last page

            # Mandatory inter-page delay to avoid FEMA 503 throttling
            time.sleep(FEMA_HA_PAGE_DELAY)

        logger.info(
            "FEMA_HA: finished %s — %d records fetched (expected %s)",
            entity_name,
            len(all_records),
            total_expected or "?",
        )
        return all_records

    def _parse_and_aggregate(self, records: list[dict]) -> pd.DataFrame:
        """Parse raw API records and aggregate to county-disaster grain.

        Args:
            records: Combined list of owner + renter records.

        Returns:
            DataFrame aggregated to (disaster_number, state, county).
        """
        if not records:
            return pd.DataFrame(columns=list(self.required_columns))

        df = pd.DataFrame(records)

        # Handle null/missing amounts — treat as 0
        df["totalApprovedIhpAmount"] = (
            pd.to_numeric(df["totalApprovedIhpAmount"], errors="coerce").fillna(0.0)
        )
        df["validRegistrations"] = (
            pd.to_numeric(df["validRegistrations"], errors="coerce").fillna(0)
        )

        # Trim whitespace from county names
        df["county"] = df["county"].astype(str).str.strip()

        # Convert disaster number to string
        df["disasterNumber"] = df["disasterNumber"].astype(str)

        # Aggregate to (disasterNumber, state, county) grain
        agg = (
            df.groupby(["disasterNumber", "state", "county"], as_index=False)
            .agg(
                ia_amount=("totalApprovedIhpAmount", "sum"),
                registrant_count=("validRegistrations", "sum"),
            )
        )

        # Rename to output schema
        agg = agg.rename(columns={"disasterNumber": "disaster_number"})

        # Ensure correct dtypes
        agg["disaster_number"] = agg["disaster_number"].astype(str)
        agg["state"] = agg["state"].astype(str)
        agg["county"] = agg["county"].astype(str)
        agg["ia_amount"] = agg["ia_amount"].astype(float)
        agg["registrant_count"] = agg["registrant_count"].astype(float)

        return agg[list(self.required_columns)].copy()

    def fetch(self, years: list[int] | None = None, force: bool = False) -> pd.DataFrame:
        """Fetch FEMA Housing Assistance payout data.

        Downloads all records from both HousingAssistanceOwners and
        HousingAssistanceRenters endpoints, aggregates to county-disaster
        grain, and caches the result.

        Args:
            years: Not used for this ingester (data is keyed by disaster,
                   not year). Accepted for interface compatibility.
            force: If False and cached data exists, return cached data.
                   If True, re-fetch from the API.

        Returns:
            DataFrame with columns: disaster_number, state, county,
            ia_amount, registrant_count.
        """
        # Return cached data if available
        if not force:
            cached = self.load_raw("fema_ha_all")
            if cached is not None:
                logger.info("FEMA_HA: using cached data (%d rows)", len(cached))
                return cached

        all_records: list[dict] = []

        # Fetch owners
        owners = self._fetch_endpoint(FEMA_HA_OWNERS_URL)
        all_records.extend(owners)

        # Fetch renters
        renters = self._fetch_endpoint(FEMA_HA_RENTERS_URL)
        all_records.extend(renters)

        if not all_records:
            logger.warning("FEMA_HA: both endpoints returned no data")
            return pd.DataFrame(columns=list(self.required_columns))

        df = self._parse_and_aggregate(all_records)

        if df.empty:
            return df

        # Cache combined file
        self.cache_raw(
            df,
            label="fema_ha_all",
            data_vintage="FEMA Housing Assistance (all disasters)",
        )

        return df
