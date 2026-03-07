"""Fetch heat-related ED visit counts from CDC EPHT Tracking Network.

Source: CDC Environmental Public Health Tracking Network
URL: https://ephtracking.cdc.gov/apigateway/api/v1/
Measure: Heat-related illness emergency department visits
Format: JSON responses from the Tracking Network API
API key required: No (but token reduces throttling)

Fetches raw visit counts ONLY. Does NOT compute per-capita rates,
per-100k rates, burden indices, normalized scores, annual trends, or
any derived metrics — those belong in transform/health_burden.py.

Every record is tagged with its geographic resolution level
(``geo_resolution``: "county" or "state") so the transform layer
knows whether to use the record directly or distribute the state
rate across all counties in that state.

CDC EPHT API uses a content area → indicator → measure hierarchy.
Data retrieval uses POST to getCoreHolder with JSON body filters.

Output columns: fips, state_fips, year, heat_ed_visits, population,
                geo_resolution
Confidence: B
Attribution: proxy (inherently climate-linked)
Minimum history: 10+ years where available
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CDC_EPHT_BASE_URL
from ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CDC EPHT API configuration
# ---------------------------------------------------------------------------

API_BASE = CDC_EPHT_BASE_URL

# Discovery endpoints (all GET, append /json for content areas only)
CONTENT_AREAS_URL = f"{API_BASE}/contentareas/json"
INDICATORS_URL = f"{API_BASE}/indicators"  # + /{contentAreaId}
MEASURES_URL = f"{API_BASE}/measures"  # + /{indicatorId}
GEO_TYPES_URL = f"{API_BASE}/geographicTypes"  # + /{measureId}
GEO_ITEMS_URL = f"{API_BASE}/geographicItems"  # + /{measureId}/{geoTypeId}/0
TEMPORAL_ITEMS_URL = f"{API_BASE}/temporalItems"  # + /{measureId}/{geoTypeId}/ALL/ALL
STRAT_LEVELS_URL = f"{API_BASE}/stratificationlevel"  # + /{measureId}/{geoTypeId}/{isSmoothed}

# Data retrieval (POST)
DATA_URL = f"{API_BASE}/getCoreHolder"  # + /{measureId}/{stratLevelId}/{smoothing}/0

# Discovery keywords
HEAT_CONTENT_KEYWORDS = ["heat", "extreme heat", "heat-related", "heat stress"]
HEAT_INDICATOR_KEYWORDS = ["emergency department", "ed visit", "hri"]
HEAT_MEASURE_KEYWORDS = ["annual number", "number of emergency", "count"]

# Geographic type IDs
GEO_TYPE_STATE = 1
GEO_TYPE_COUNTY = 2

# Use raw (unsmoothed) data
IS_SMOOTHED = 0

# Suppression markers
SUPPRESSION_MARKERS = {"*", "Suppressed", "suppressed", "N/A", "n/a", ""}

# Rate limit: 1 request/second (be polite to federal CDC API)
CDC_CALLS_PER_SECOND = 1.0

# Default history
DEFAULT_START_YEAR = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_state_fips(raw: str | int | float) -> str:
    """Normalize a state FIPS code to a 2-digit zero-padded string."""
    if isinstance(raw, float):
        raw = int(raw)
    code = str(raw).strip().lstrip("0") or "0"
    numeric = int(code)
    result = f"{numeric:02d}"
    if len(result) != 2 or numeric < 1 or numeric > 78:
        raise ValueError(f"State FIPS code out of range: {raw!r} → {result}")
    return result


def normalize_county_fips(raw: str | int | float) -> str:
    """Normalize a county FIPS code to a 5-digit zero-padded string."""
    if isinstance(raw, float):
        raw = int(raw)
    code = str(raw).strip().lstrip("0") or "0"
    numeric = int(code)
    result = f"{numeric:05d}"
    if len(result) != 5:
        raise ValueError(f"County FIPS code out of range: {raw!r} → {result}")
    return result


def _is_suppressed(value: Any) -> bool:
    """Check if a value represents a CDC-suppressed count."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in SUPPRESSION_MARKERS:
        return True
    return False


def _parse_count(value: Any) -> float:
    """Parse a visit count or population value. NaN for suppressed/missing."""
    if _is_suppressed(value):
        return np.nan
    try:
        # Remove commas from display values like "2,074"
        if isinstance(value, str):
            value = value.replace(",", "")
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------

class CDCEPHTIngester(BaseIngester):
    """Ingest heat-related ED visit counts from CDC EPHT Tracking Network.

    Discovers the correct measure ID programmatically via the content area →
    indicator → measure hierarchy, then fetches all available state-level
    data in a single POST call. Tags every record with its geographic
    resolution.
    """

    source_name = "cdc_epht"
    confidence = "B"
    attribution = "proxy"
    calls_per_second = CDC_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "fips": str,
        "state_fips": str,
        "year": int,
        "heat_ed_visits": float,
        "population": float,
        "geo_resolution": str,
    }

    def __init__(self) -> None:
        super().__init__()
        self._measure_id: str | None = None
        self._measure_name: str = ""

    # -- Measure discovery ----------------------------------------------------

    def _discover_content_area_id(self) -> str:
        """Find the content area ID for heat-related illness.

        Queries ``/contentareas/json`` and searches by keyword.

        Returns:
            The content area ID string.

        Raises:
            RuntimeError: If no heat-related content area is found.
        """
        resp = self.api_get(CONTENT_AREAS_URL)
        content_areas = resp.json()

        if not content_areas:
            raise RuntimeError(
                "CDC_EPHT: content areas endpoint returned empty response."
            )

        for area in content_areas:
            name = str(area.get("name") or "").lower()
            area_id = str(area.get("id") or "")
            for keyword in HEAT_CONTENT_KEYWORDS:
                if keyword in name:
                    logger.info(
                        "CDC_EPHT: found heat content area: id=%s, name='%s'",
                        area_id, area.get("name"),
                    )
                    return area_id

        area_names = [str(a.get("name", "unknown")) for a in content_areas]
        raise RuntimeError(
            f"CDC_EPHT: no heat-related content area found. "
            f"Available: {area_names}"
        )

    def _discover_indicator_id(self, content_area_id: str) -> str:
        """Find the indicator ID for heat-related ED visits.

        Queries ``/indicators/{contentAreaId}`` and searches by keyword.

        Args:
            content_area_id: The content area ID.

        Returns:
            The indicator ID string.

        Raises:
            RuntimeError: If no ED visit indicator is found.
        """
        url = f"{INDICATORS_URL}/{content_area_id}"
        resp = self.api_get(url)
        indicators = resp.json()

        if not indicators:
            raise RuntimeError(
                f"CDC_EPHT: indicators endpoint returned empty for "
                f"content area {content_area_id}."
            )

        for ind in indicators:
            name = str(ind.get("name") or ind.get("shortName") or "").lower()
            ind_id = str(ind.get("id") or "")
            for keyword in HEAT_INDICATOR_KEYWORDS:
                if keyword in name:
                    logger.info(
                        "CDC_EPHT: found ED visit indicator: id=%s, name='%s'",
                        ind_id, ind.get("name"),
                    )
                    return ind_id

        ind_names = [str(i.get("name", "unknown")) for i in indicators]
        raise RuntimeError(
            f"CDC_EPHT: no ED visit indicator found in content area "
            f"{content_area_id}. Available: {ind_names}"
        )

    def _discover_measure_id(self, indicator_id: str) -> str:
        """Find the measure ID for annual ED visit counts (raw counts).

        Queries ``/measures/{indicatorId}`` and looks for the raw count
        measure (not age-adjusted rates or crude rates).

        Args:
            indicator_id: The indicator ID.

        Returns:
            The measure ID string.

        Raises:
            RuntimeError: If no count measure is found.
        """
        url = f"{MEASURES_URL}/{indicator_id}"
        resp = self.api_get(url)
        measures = resp.json()

        if not measures:
            raise RuntimeError(
                f"CDC_EPHT: measures endpoint returned empty for "
                f"indicator {indicator_id}."
            )

        # Prefer "Annual Number" (raw counts) over rate measures
        for measure in measures:
            name = str(measure.get("name") or "").lower()
            m_id = str(measure.get("id") or "")
            for keyword in HEAT_MEASURE_KEYWORDS:
                if keyword in name and "rate" not in name:
                    self._measure_name = str(measure.get("name") or "")
                    logger.info(
                        "CDC_EPHT: discovered measure — id=%s, name='%s'",
                        m_id, self._measure_name,
                    )
                    return m_id

        # Fallback: take first measure that isn't a rate
        for measure in measures:
            name = str(measure.get("name") or "").lower()
            if "rate" not in name:
                m_id = str(measure.get("id") or "")
                self._measure_name = str(measure.get("name") or "")
                logger.info(
                    "CDC_EPHT: fallback measure — id=%s, name='%s'",
                    m_id, self._measure_name,
                )
                return m_id

        measure_names = [str(m.get("name", "unknown")) for m in measures]
        raise RuntimeError(
            f"CDC_EPHT: no count measure found for indicator "
            f"{indicator_id}. Available: {measure_names}"
        )

    def discover_measure(self) -> str:
        """Discover and cache the heat ED visit measure ID.

        Traverses: content areas → indicators → measures.

        Returns:
            The measure ID string.
        """
        if self._measure_id is not None:
            return self._measure_id

        content_area_id = self._discover_content_area_id()
        indicator_id = self._discover_indicator_id(content_area_id)
        self._measure_id = self._discover_measure_id(indicator_id)
        return self._measure_id

    # -- API metadata queries -------------------------------------------------

    def _get_available_geo_types(self, measure_id: str) -> list[dict[str, Any]]:
        """Get available geographic types for a measure.

        Args:
            measure_id: The measure ID.

        Returns:
            List of geographic type records with geographicTypeId.
        """
        url = f"{GEO_TYPES_URL}/{measure_id}"
        resp = self.api_get(url)
        data = resp.json()
        return data if isinstance(data, list) else []

    def _get_geographic_items(
        self, measure_id: str, geo_type_id: int,
    ) -> list[dict[str, Any]]:
        """Get available geographic items (states/counties) for a measure.

        Args:
            measure_id: The measure ID.
            geo_type_id: Geographic type (1=state, 2=county).

        Returns:
            List of geographic item records.
        """
        url = f"{GEO_ITEMS_URL}/{measure_id}/{geo_type_id}/0"
        resp = self.api_get(url)
        data = resp.json()
        return data if isinstance(data, list) else []

    def _get_temporal_items(
        self, measure_id: str, geo_type_id: int,
    ) -> list[dict[str, Any]]:
        """Get available temporal items (years) for a measure.

        Args:
            measure_id: The measure ID.
            geo_type_id: Geographic type.

        Returns:
            List of temporal item records with temporalId.
        """
        url = f"{TEMPORAL_ITEMS_URL}/{measure_id}/{geo_type_id}/ALL/ALL"
        resp = self.api_get(url)
        data = resp.json()
        return data if isinstance(data, list) else []

    def _get_stratification_level(
        self, measure_id: str, geo_type_id: int,
    ) -> str:
        """Get the base (non-stratified) stratification level ID.

        Args:
            measure_id: The measure ID.
            geo_type_id: Geographic type.

        Returns:
            Stratification level ID string (usually "1" for plain state).
        """
        url = f"{STRAT_LEVELS_URL}/{measure_id}/{geo_type_id}/{IS_SMOOTHED}"
        resp = self.api_get(url)
        data = resp.json()
        if isinstance(data, list) and data:
            # Return the simplest (no extra stratification) level
            for level in data:
                strat_types = level.get("stratificationType", [])
                if not strat_types:
                    return str(level.get("id", "1"))
            # Fallback to first
            return str(data[0].get("id", "1"))
        return "1"

    # -- Data fetching --------------------------------------------------------

    def _fetch_data_post(
        self,
        measure_id: str,
        strat_level_id: str,
        geo_type_id: int,
        geo_item_ids: list[int],
        temporal_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Fetch data via POST to getCoreHolder.

        Args:
            measure_id: The measure ID.
            strat_level_id: Stratification level ID.
            geo_type_id: Geographic type (1=state, 2=county).
            geo_item_ids: List of geographic item IDs.
            temporal_ids: List of temporal (year) IDs.

        Returns:
            List of data records from tableResult.
        """
        url = f"{DATA_URL}/{measure_id}/{strat_level_id}/{IS_SMOOTHED}/0"
        body = {
            "geographicTypeIdFilter": geo_type_id,
            "geographicItemsFilter": ",".join(str(g) for g in geo_item_ids),
            "temporalTypeIdFilter": 1,  # Year
            "temporalItemsFilter": ",".join(str(t) for t in temporal_ids),
        }

        self.rate_limit()
        resp = self.client.post(url, json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            return data.get("tableResult", []) or []
        if isinstance(data, list):
            return data
        return []

    def _parse_record(
        self,
        record: dict[str, Any],
        geo_type_id: int,
    ) -> dict[str, Any] | None:
        """Parse a single API record into the output schema.

        Args:
            record: Raw API record from tableResult.
            geo_type_id: 1 for state-level, 2 for county-level.

        Returns:
            Parsed record dict, or None if invalid.
        """
        # Extract year from temporal field
        year_raw = record.get("temporal") or record.get("temporalId")
        if year_raw is None:
            return None
        try:
            year = int(str(year_raw).strip()[:4])
        except (ValueError, TypeError):
            return None

        # Extract visit count — check suppressionFlag first
        suppression_flag = str(record.get("suppressionFlag", "0"))
        if suppression_flag != "0":
            heat_ed_visits = np.nan
        else:
            data_value = record.get("dataValue")
            heat_ed_visits = _parse_count(data_value)

        # Population not provided by measure 438; transform/health_burden.py sources it from census_acs.py
        population = np.nan

        # Extract geographic identifiers
        geo_id = record.get("geoId", "")

        if geo_type_id == GEO_TYPE_COUNTY:
            try:
                fips = normalize_county_fips(geo_id)
            except (ValueError, TypeError):
                return None
            state_fips = fips[:2]
            geo_resolution = "county"
        else:
            try:
                state_fips = normalize_state_fips(geo_id)
            except (ValueError, TypeError):
                return None
            fips = state_fips
            geo_resolution = "state"

        return {
            "fips": fips,
            "state_fips": state_fips,
            "year": year,
            "heat_ed_visits": heat_ed_visits,
            "population": population,
            "geo_resolution": geo_resolution,
        }

    # -- Main fetch -----------------------------------------------------------

    def _default_years(self) -> list[int]:
        """Return the default year range (2000–present)."""
        current_year = datetime.now().year
        return list(range(DEFAULT_START_YEAR, current_year + 1))

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch heat-related ED visit data from CDC EPHT.

        Discovers the correct measure ID, determines available geographic
        types and years, then fetches all data via bulk POST calls.
        Tags every record with its geographic resolution.

        Args:
            years: Calendar years to fetch. Defaults to 2000–present.

        Returns:
            DataFrame with columns: fips, state_fips, year,
            heat_ed_visits, population, geo_resolution.
        """
        if years is None:
            years = self._default_years()

        # Step 1: Discover measure ID
        measure_id = self.discover_measure()

        # Step 2: Discover available geographic types
        geo_types = self._get_available_geo_types(measure_id)
        available_geo_type_ids = {
            gt.get("geographicTypeId") for gt in geo_types
        }
        logger.info(
            "CDC_EPHT: available geo types for measure %s: %s",
            measure_id, available_geo_type_ids,
        )

        all_records: list[dict[str, Any]] = []
        reporting_states: dict[int, list[str]] = {}
        county_level_states: dict[int, list[str]] = {}

        # Step 3: Fetch state-level data
        if GEO_TYPE_STATE in available_geo_type_ids:
            try:
                state_records = self._fetch_geo_type_data(
                    measure_id, GEO_TYPE_STATE, years,
                )
                all_records.extend(state_records)

                for rec in state_records:
                    yr = rec["year"]
                    sf = rec["state_fips"]
                    reporting_states.setdefault(yr, [])
                    if sf not in reporting_states[yr]:
                        reporting_states[yr].append(sf)
            except Exception:
                logger.error(
                    "CDC_EPHT: failed to fetch state-level data",
                    exc_info=True,
                )

        # Step 4: Fetch county-level data (if available)
        if GEO_TYPE_COUNTY in available_geo_type_ids:
            try:
                county_records = self._fetch_geo_type_data(
                    measure_id, GEO_TYPE_COUNTY, years,
                )
                all_records.extend(county_records)

                for rec in county_records:
                    yr = rec["year"]
                    sf = rec["state_fips"]
                    reporting_states.setdefault(yr, [])
                    if sf not in reporting_states[yr]:
                        reporting_states[yr].append(sf)
                    county_level_states.setdefault(yr, [])
                    if sf not in county_level_states[yr]:
                        county_level_states[yr].append(sf)
            except Exception:
                logger.warning(
                    "CDC_EPHT: failed to fetch county-level data",
                    exc_info=True,
                )
        else:
            logger.info(
                "CDC_EPHT: county-level data not available for measure %s",
                measure_id,
            )

        if not all_records:
            logger.error("CDC_EPHT: no data retrieved")
            return pd.DataFrame(columns=list(self.required_columns))

        # Build DataFrame
        df = pd.DataFrame(all_records)
        df["fips"] = df["fips"].astype(str)
        df["state_fips"] = df["state_fips"].astype(str)
        df["year"] = df["year"].astype(int)
        df["heat_ed_visits"] = df["heat_ed_visits"].astype(float)
        df["population"] = df["population"].astype(float)
        df["geo_resolution"] = df["geo_resolution"].astype(str)
        df = df[list(self.required_columns)].copy()

        # Cache with coverage metadata
        year_range_str = f"{min(years)} to {max(years)}"
        self._cache_with_coverage_metadata(
            df,
            label="cdc_epht_all",
            data_vintage=year_range_str,
            reporting_states=reporting_states,
            county_level_states=county_level_states,
        )

        return df

    def _fetch_geo_type_data(
        self,
        measure_id: str,
        geo_type_id: int,
        years: list[int],
    ) -> list[dict[str, Any]]:
        """Fetch all data for a geographic type via bulk POST.

        Queries the API for available geographic items and temporal items,
        filters temporals to the requested years, then makes a single
        POST call to retrieve all data.

        Args:
            measure_id: The measure ID.
            geo_type_id: Geographic type (1=state, 2=county).
            years: Requested years.

        Returns:
            List of parsed record dicts.
        """
        # Get available geographic items
        geo_items = self._get_geographic_items(measure_id, geo_type_id)
        if not geo_items:
            logger.info(
                "CDC_EPHT: no geographic items for geo type %d", geo_type_id,
            )
            return []

        geo_item_ids = [
            g.get("parentGeographicId") or g.get("id")
            for g in geo_items
        ]
        geo_item_ids = [g for g in geo_item_ids if g is not None]

        # Get available temporal items
        temporal_items = self._get_temporal_items(measure_id, geo_type_id)
        if not temporal_items:
            logger.warning(
                "CDC_EPHT: no temporal items for geo type %d", geo_type_id,
            )
            return []

        # Filter to requested years
        available_years = {
            t.get("temporalId") for t in temporal_items
            if t.get("temporalId") is not None
        }
        requested_temporal_ids = [y for y in years if y in available_years]
        if not requested_temporal_ids:
            logger.info(
                "CDC_EPHT: none of the requested years %s are available "
                "(available: %s)",
                years, sorted(available_years),
            )
            return []

        # Get stratification level
        strat_level_id = self._get_stratification_level(
            measure_id, geo_type_id,
        )

        # Fetch data in one POST call
        logger.info(
            "CDC_EPHT: fetching geo_type=%d, %d geo items, %d years, "
            "strat_level=%s",
            geo_type_id, len(geo_item_ids), len(requested_temporal_ids),
            strat_level_id,
        )
        raw_records = self._fetch_data_post(
            measure_id,
            strat_level_id,
            geo_type_id,
            geo_item_ids,
            requested_temporal_ids,
        )

        # Parse records
        parsed: list[dict[str, Any]] = []
        suppressed_count = 0
        for rec in raw_records:
            result = self._parse_record(rec, geo_type_id)
            if result is not None:
                parsed.append(result)
                if np.isnan(result["heat_ed_visits"]):
                    suppressed_count += 1

        if suppressed_count > 0:
            logger.info(
                "CDC_EPHT: %d suppressed records for geo_type %d",
                suppressed_count, geo_type_id,
            )

        geo_label = "state" if geo_type_id == GEO_TYPE_STATE else "county"
        logger.info(
            "CDC_EPHT: parsed %d %s-level records", len(parsed), geo_label,
        )
        return parsed

    # -- Custom completeness logging ------------------------------------------

    def log_completeness(self, df: pd.DataFrame) -> None:
        """Log state-level coverage instead of county coverage."""
        n_states = (
            df["state_fips"].nunique() if "state_fips" in df.columns else 0
        )
        n_rows = len(df)
        n_county = (
            len(df[df["geo_resolution"] == "county"])
            if "geo_resolution" in df.columns
            else 0
        )
        n_state = (
            len(df[df["geo_resolution"] == "state"])
            if "geo_resolution" in df.columns
            else 0
        )

        logger.info(
            "%s completeness: %d rows (%d county-level, %d state-level), "
            "%d reporting states (of 51 incl. DC)",
            self.source_name, n_rows, n_county, n_state, n_states,
        )

    # -- Caching with coverage metadata ---------------------------------------

    def _cache_with_coverage_metadata(
        self,
        df: pd.DataFrame,
        label: str,
        data_vintage: str,
        reporting_states: dict[int, list[str]],
        county_level_states: dict[int, list[str]],
    ) -> Path:
        """Save data and write metadata with coverage documentation."""
        parquet_path = self.cache_dir() / f"{label}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(
            "%s: saved %d rows to %s",
            self.source_name, len(df), parquet_path,
        )

        total_reporting = set()
        for states in reporting_states.values():
            total_reporting.update(states)

        total_county = set()
        for states in county_level_states.values():
            total_county.update(states)

        reporting_by_year = {
            str(yr): sorted(states)
            for yr, states in sorted(reporting_states.items())
        }
        county_by_year = {
            str(yr): sorted(states)
            for yr, states in sorted(county_level_states.items())
        }

        metadata = {
            "source": "CDC_EPHT",
            "confidence": self.confidence,
            "attribution": self.attribution,
            "retrieved_at": datetime.now().isoformat(),
            "data_vintage": data_vintage,
            "measure_id": self._measure_id,
            "measure_name": self._measure_name,
            "coverage": {
                "total_reporting_states": len(total_reporting),
                "total_states_with_county_data": len(total_county),
                "reporting_states_by_year": reporting_by_year,
                "county_level_states_by_year": county_by_year,
            },
        }

        meta_path = self.cache_dir() / f"{label}_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info("%s: wrote metadata to %s", self.source_name, meta_path)

        return parquet_path
