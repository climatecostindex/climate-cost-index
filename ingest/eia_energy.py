"""Fetch residential energy price/consumption data from EIA and RECS microdata.

Source: EIA Open Data API (v2) + Residential Energy Consumption Survey (RECS)
EIA API URL: https://api.eia.gov/v2/
RECS URL: https://www.eia.gov/consumption/residential/data/
Format: JSON (EIA API) + CSV (RECS bulk download)
API key: Required — EIA_API_KEY in .env

This is the ONLY "attributed" data source in CCI v1. The energy component
is the only one with formal causal attribution methodology (degree-day
regression isolating climate-driven consumption).

Fetches raw state-level prices/consumption and RECS household microdata ONLY.
Does NOT compute climate-attributed costs, degree-day regressions, panel
regressions, county-level allocations, rate-case structural break detection,
consumption-per-household normalization, or ANY derived metrics — all of
that belongs in transform/energy_attribution.py.

Output — State aggregate table:
    state_fips, state_abbr, year, electricity_price_cents_kwh,
    electricity_consumption_mwh, natural_gas_price
Output — RECS microdata table:
    household_id, census_division, census_division_name, dwelling_type,
    dwelling_type_code, square_footage, heating_fuel, heating_fuel_code,
    num_occupants, annual_electricity_kwh, annual_gas_ccf, sample_weight,
    recs_year

State aggregate confidence: A, attribution: attributed
RECS microdata confidence: B, attribution: attributed
Minimum history: 25 years (state aggregate, provides buffer for rate-case detection)
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EIA API configuration
# ---------------------------------------------------------------------------

EIA_API_BASE_URL = "https://api.eia.gov/v2"

# API routes
EIA_ELECTRICITY_ROUTE = "electricity/retail-sales"
EIA_NATURAL_GAS_ROUTE = "natural-gas/pri/sum"

# Rate limit: 100 requests/hour → ~1.7 req/min. Use 1 req/sec to be safe.
EIA_CALLS_PER_SECOND = 1.0

# Minimum trailing history depth (years)
EIA_MIN_HISTORY_YEARS = 25

# Residential sector facet code
EIA_RESIDENTIAL_SECTOR = "RES"

# ---------------------------------------------------------------------------
# RECS configuration
# ---------------------------------------------------------------------------

RECS_2020_URL = (
    "https://www.eia.gov/consumption/residential/data/2020/csv/"
    "recs2020_public_v7.csv"
)
RECS_VINTAGE_YEAR = 2020

# RECS column mappings (2020 RECS public use file)
RECS_COLUMNS = {
    "DOEID": "household_id",
    "DIVISION": "census_division",
    "TYPEHUQ": "dwelling_type_code",
    "TOTSQFT_EN": "square_footage",
    "FUELHEAT": "heating_fuel_code",
    "NHSLDMEM": "num_occupants",
    "KWH": "annual_electricity_kwh",
    "CUFEETNG": "annual_gas_cf",
    "NWEIGHT": "sample_weight",
}

# Census division code → name mapping (traditional 9-division scheme)
CENSUS_DIVISION_NAMES: dict[int, str] = {
    1: "New England",
    2: "Middle Atlantic",
    3: "East North Central",
    4: "West North Central",
    5: "South Atlantic",
    6: "East South Central",
    7: "West South Central",
    8: "Mountain",
    9: "Pacific",
}

# 2020 RECS uses string division names (and splits Mountain into two).
# Map back to traditional Census division codes for consistency.
RECS_DIVISION_NAME_TO_CODE: dict[str, int] = {
    "New England": 1,
    "Middle Atlantic": 2,
    "East North Central": 3,
    "East South Central": 6,
    "Mountain North": 8,
    "Mountain South": 8,
    "Pacific": 9,
    "South Atlantic": 5,
    "West North Central": 4,
    "West South Central": 7,
}

# RECS dwelling type code → label mapping (2020 RECS TYPEHUQ)
DWELLING_TYPE_LABELS: dict[int, str] = {
    1: "Mobile home",
    2: "Single-family detached",
    3: "Single-family attached",
    4: "Apartment in 2-4 unit building",
    5: "Apartment in 5+ unit building",
}

# RECS heating fuel code → label mapping (2020 RECS FUELHEAT)
HEATING_FUEL_LABELS: dict[int, str] = {
    1: "Natural gas",
    2: "Propane",
    3: "Fuel oil",
    4: "Kerosene",
    5: "Electricity",
    7: "Wood",
    21: "Other",
    -2: "Not applicable",
}

# ---------------------------------------------------------------------------
# State abbreviation → FIPS mapping
# ---------------------------------------------------------------------------

STATE_ABBR_TO_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}

# Total U.S. states + DC for completeness logging
TOTAL_US_STATES = 51


class EIAEnergyIngester(BaseIngester):
    """Ingest EIA residential energy data and RECS microdata.

    Produces TWO separate output tables:
    1. State aggregate: residential electricity/gas prices and consumption
    2. RECS microdata: household-level energy characteristics

    These MUST remain separate — the transform layer uses them differently.
    """

    source_name = "eia_energy"
    confidence = "A"
    attribution = "attributed"
    calls_per_second = EIA_CALLS_PER_SECOND

    # Schema for the state aggregate table
    required_columns: dict[str, type] = {
        "state_fips": str,
        "state_abbr": str,
        "year": int,
        "electricity_price_cents_kwh": float,
        "electricity_consumption_mwh": float,
        "natural_gas_price": float,
    }

    # Schema for the RECS microdata table
    recs_required_columns: dict[str, type] = {
        "household_id": str,
        "census_division": int,
        "census_division_name": str,
        "dwelling_type": str,
        "dwelling_type_code": int,
        "square_footage": float,
        "heating_fuel": str,
        "heating_fuel_code": int,
        "num_occupants": int,
        "annual_electricity_kwh": float,
        "annual_gas_ccf": float,
        "sample_weight": float,
        "recs_year": int,
    }

    def __init__(self) -> None:
        super().__init__()
        self._api_key: str = os.getenv("EIA_API_KEY", "")

    # -- API key validation ----------------------------------------------------

    def _require_api_key(self) -> str:
        """Return the EIA API key or raise if missing.

        Raises:
            RuntimeError: If EIA_API_KEY is not set.
        """
        if not self._api_key:
            raise RuntimeError(
                "EIA_API_KEY environment variable is not set. "
                "Register for a free key at "
                "https://www.eia.gov/opendata/register.php"
            )
        return self._api_key

    # -- EIA API queries -------------------------------------------------------

    def _query_eia_api(
        self,
        route: str,
        data_field: str,
        facets: dict[str, list[str]],
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Query the EIA API v2 and return the response data records.

        Args:
            route: API route (e.g. "electricity/retail-sales").
            data_field: Data field to request (e.g. "price", "sales").
            facets: Facet filters (e.g. {"sectorid": ["RES"]}).
            start_year: Start year (inclusive).
            end_year: End year (inclusive).

        Returns:
            List of data records from the API response.
        """
        api_key = self._require_api_key()
        url = f"{EIA_API_BASE_URL}/{route}/data"

        params: dict[str, Any] = {
            "api_key": api_key,
            "frequency": "annual",
            "data[0]": data_field,
            "start": str(start_year),
            "end": str(end_year),
            "length": "5000",
        }

        # Add facets
        for facet_key, facet_values in facets.items():
            for val in facet_values:
                params[f"facets[{facet_key}][]"] = val

        logger.info(
            "EIA_ENERGY: querying %s for %s (%d–%d)",
            route, data_field, start_year, end_year,
        )

        resp = self.api_get(url, params=params)
        body = resp.json()

        response = body.get("response", {})
        records = response.get("data", [])
        # data can be a dict (metadata response) or list (data records)
        if isinstance(records, dict):
            records = []
        total = int(response.get("total", 0))

        # Handle pagination if needed
        all_records = list(records)
        offset = len(records)
        while offset < total:
            params["offset"] = str(offset)
            resp = self.api_get(url, params=params)
            body = resp.json()
            page_records = body.get("response", {}).get("data", [])
            if not page_records:
                break
            all_records.extend(page_records)
            offset += len(page_records)

        logger.info(
            "EIA_ENERGY: received %d records from %s/%s",
            len(all_records), route, data_field,
        )
        return all_records

    def _fetch_electricity_prices(
        self, start_year: int, end_year: int,
    ) -> pd.DataFrame:
        """Fetch residential electricity prices by state.

        Args:
            start_year: Start year (inclusive).
            end_year: End year (inclusive).

        Returns:
            DataFrame with columns: state_abbr, year,
            electricity_price_cents_kwh.
        """
        records = self._query_eia_api(
            route=EIA_ELECTRICITY_ROUTE,
            data_field="price",
            facets={"sectorid": [EIA_RESIDENTIAL_SECTOR]},
            start_year=start_year,
            end_year=end_year,
        )

        rows = []
        for rec in records:
            state_id = str(rec.get("stateid", "")).upper()
            if state_id not in STATE_ABBR_TO_FIPS:
                continue

            period = rec.get("period")
            value = rec.get("price") if "price" in rec else rec.get("value")

            try:
                year = int(period)
                price = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                continue

            rows.append({
                "state_abbr": state_id,
                "year": year,
                "electricity_price_cents_kwh": price,
            })

        if not rows:
            return pd.DataFrame(
                columns=["state_abbr", "year", "electricity_price_cents_kwh"]
            )
        return pd.DataFrame(rows)

    def _fetch_electricity_consumption(
        self, start_year: int, end_year: int,
    ) -> pd.DataFrame:
        """Fetch residential electricity consumption (sales) by state.

        Args:
            start_year: Start year (inclusive).
            end_year: End year (inclusive).

        Returns:
            DataFrame with columns: state_abbr, year,
            electricity_consumption_mwh.
        """
        records = self._query_eia_api(
            route=EIA_ELECTRICITY_ROUTE,
            data_field="sales",
            facets={"sectorid": [EIA_RESIDENTIAL_SECTOR]},
            start_year=start_year,
            end_year=end_year,
        )

        rows = []
        for rec in records:
            state_id = str(rec.get("stateid", "")).upper()
            if state_id not in STATE_ABBR_TO_FIPS:
                continue

            period = rec.get("period")
            value = rec.get("sales") if "sales" in rec else rec.get("value")

            try:
                year = int(period)
                consumption = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                continue

            rows.append({
                "state_abbr": state_id,
                "year": year,
                "electricity_consumption_mwh": consumption,
            })

        if not rows:
            return pd.DataFrame(
                columns=["state_abbr", "year", "electricity_consumption_mwh"]
            )
        return pd.DataFrame(rows)

    def _fetch_natural_gas_prices(
        self, start_year: int, end_year: int,
    ) -> pd.DataFrame:
        """Fetch residential natural gas prices by state.

        Not all states report residential natural gas prices. Missing values
        are expected and stored as NaN.

        Args:
            start_year: Start year (inclusive).
            end_year: End year (inclusive).

        Returns:
            DataFrame with columns: state_abbr, year, natural_gas_price.
        """
        records = self._query_eia_api(
            route=EIA_NATURAL_GAS_ROUTE,
            data_field="value",
            facets={"process": ["PRS"]},
            start_year=start_year,
            end_year=end_year,
        )

        rows = []
        for rec in records:
            # EIA natural gas uses duoarea codes like "SCA" for state CA.
            duoarea = str(rec.get("duoarea", ""))
            if len(duoarea) == 3 and duoarea.startswith("S"):
                abbr = duoarea[1:].upper()
            else:
                abbr = str(rec.get("stateid", duoarea)).upper()

            if abbr not in STATE_ABBR_TO_FIPS:
                continue

            period = rec.get("period")
            value = rec.get("value")

            try:
                year = int(period)
                price = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                continue

            rows.append({
                "state_abbr": abbr,
                "year": year,
                "natural_gas_price": price,
            })

        if not rows:
            return pd.DataFrame(
                columns=["state_abbr", "year", "natural_gas_price"]
            )
        return pd.DataFrame(rows)

    # -- RECS microdata --------------------------------------------------------

    def _download_recs_csv(self) -> bytes:
        """Download the RECS public use microdata CSV.

        Returns:
            Raw CSV file bytes.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
        """
        logger.info(
            "EIA_ENERGY: downloading RECS %d microdata from %s",
            RECS_VINTAGE_YEAR, RECS_2020_URL,
        )
        resp = self.api_get(RECS_2020_URL)
        return resp.content

    def _parse_recs_csv(self, csv_bytes: bytes) -> pd.DataFrame:
        """Parse the RECS public use microdata CSV.

        Extracts household-level records with energy characteristics,
        maps numeric codes to human-readable labels, and preserves
        raw codes for transform layer use.

        Args:
            csv_bytes: Raw CSV file bytes.

        Returns:
            DataFrame with RECS microdata columns.
        """
        source_cols_upper = [k.upper() for k in RECS_COLUMNS]
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            usecols=lambda c: c.upper() in source_cols_upper,
            low_memory=False,
        )

        # Normalize column names to uppercase
        df.columns = [c.upper() for c in df.columns]

        # Rename to output names
        rename_map = {k.upper(): v for k, v in RECS_COLUMNS.items()}
        df = df.rename(columns=rename_map)

        # Map census division — 2020 RECS uses string names, not codes.
        # Detect whether column is numeric or string-based.
        raw_div = df["census_division"]
        if raw_div.dtype == object or pd.api.types.is_string_dtype(raw_div):
            # String division names (2020 RECS format)
            df["census_division_name"] = raw_div.astype(str)
            df["census_division"] = (
                raw_div.map(RECS_DIVISION_NAME_TO_CODE).fillna(0).astype(int)
            )
        else:
            # Numeric division codes (older RECS format)
            df["census_division"] = pd.to_numeric(
                raw_div, errors="coerce"
            ).fillna(0).astype(int)
            df["census_division_name"] = (
                df["census_division"]
                .map(CENSUS_DIVISION_NAMES)
                .fillna("Unknown")
            )

        # Map dwelling type codes to labels
        df["dwelling_type_code"] = pd.to_numeric(
            df["dwelling_type_code"], errors="coerce"
        ).fillna(0).astype(int)
        df["dwelling_type"] = (
            df["dwelling_type_code"]
            .map(DWELLING_TYPE_LABELS)
            .fillna("Unknown")
        )

        # Map heating fuel codes to labels
        df["heating_fuel_code"] = pd.to_numeric(
            df["heating_fuel_code"], errors="coerce"
        ).fillna(0).astype(int)
        df["heating_fuel"] = (
            df["heating_fuel_code"]
            .map(HEATING_FUEL_LABELS)
            .fillna("Unknown")
        )

        # Preserve raw RECS gas value (CUFEETNG) in hundred cubic feet (CCF).
        # CCF → therms conversion (×1.036) belongs in the transform layer.
        if "annual_gas_cf" in df.columns:
            df["annual_gas_ccf"] = pd.to_numeric(
                df["annual_gas_cf"], errors="coerce"
            )
            df = df.drop(columns=["annual_gas_cf"])
        else:
            df["annual_gas_ccf"] = float("nan")

        # Ensure correct types
        df["household_id"] = df["household_id"].astype(str)
        df["square_footage"] = pd.to_numeric(
            df["square_footage"], errors="coerce"
        ).astype(float)
        df["num_occupants"] = pd.to_numeric(
            df["num_occupants"], errors="coerce"
        ).fillna(0).astype(int)
        df["annual_electricity_kwh"] = pd.to_numeric(
            df["annual_electricity_kwh"], errors="coerce"
        ).astype(float)
        df["sample_weight"] = pd.to_numeric(
            df["sample_weight"], errors="coerce"
        ).astype(float)

        # Handle negative RECS sentinel values
        df.loc[
            df["annual_electricity_kwh"] < 0, "annual_electricity_kwh"
        ] = float("nan")

        # Add RECS vintage year
        df["recs_year"] = RECS_VINTAGE_YEAR

        # Select output columns
        return df[list(self.recs_required_columns)].copy()

    # -- Completeness logging override -----------------------------------------

    def log_completeness(self, df: pd.DataFrame) -> None:
        """Log state-level completeness (overrides county-level base)."""
        if "state_abbr" in df.columns:
            n_states = df["state_abbr"].nunique()
            pct = (n_states / TOTAL_US_STATES) * 100
            logger.info(
                "%s completeness: %d rows, %d states (%.1f%% of %d)",
                self.source_name, len(df), n_states, pct, TOTAL_US_STATES,
            )
        elif "household_id" in df.columns:
            logger.info(
                "%s completeness (RECS): %d household records",
                self.source_name, df["household_id"].nunique(),
            )
        else:
            super().log_completeness(df)

    # -- RECS-specific caching -------------------------------------------------

    def _cache_recs(
        self,
        df: pd.DataFrame,
        label: str,
        data_vintage: str,
    ) -> Path:
        """Cache RECS microdata with its own metadata sidecar.

        RECS has different confidence (B) and source identifier (EIA_RECS)
        from the state aggregate table, so we write a custom metadata
        sidecar instead of using the base class method.

        Args:
            df: RECS microdata DataFrame.
            label: File stem for the parquet file.
            data_vintage: Human-readable description.

        Returns:
            Path to the saved parquet file.
        """
        parquet_path = self.cache_dir() / f"{label}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(
            "%s: saved %d RECS rows to %s",
            self.source_name, len(df), parquet_path,
        )

        metadata = {
            "source": "EIA_RECS",
            "confidence": "B",
            "attribution": "attributed",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "data_vintage": data_vintage,
            "recs_vintage": str(RECS_VINTAGE_YEAR),
        }
        meta_path = self.cache_dir() / f"{label}_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info(
            "%s: wrote RECS metadata to %s", self.source_name, meta_path,
        )

        return parquet_path

    # -- Main fetch ------------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch EIA state-level energy data and RECS microdata.

        Downloads residential electricity prices, electricity consumption,
        and natural gas prices from the EIA API, plus RECS household
        microdata. Caches state aggregate and RECS as SEPARATE files.

        Args:
            years: If provided, fetch only these years for state data.
                   If None, fetch trailing 15 years from current year.

        Returns:
            State aggregate DataFrame (primary output for validation).
            RECS microdata is cached separately.
        """
        self._require_api_key()

        if years is None:
            current_year = date.today().year
            start_year = current_year - EIA_MIN_HISTORY_YEARS
            end_year = current_year
        else:
            start_year = min(years)
            end_year = max(years)

        # --- Fetch state-level data ---
        failed_queries: list[str] = []

        try:
            elec_prices = self._fetch_electricity_prices(start_year, end_year)
        except Exception:
            logger.warning(
                "EIA_ENERGY: failed to fetch electricity prices",
                exc_info=True,
            )
            elec_prices = pd.DataFrame(
                columns=["state_abbr", "year", "electricity_price_cents_kwh"]
            )
            failed_queries.append("electricity_prices")

        try:
            elec_consumption = self._fetch_electricity_consumption(
                start_year, end_year,
            )
        except Exception:
            logger.warning(
                "EIA_ENERGY: failed to fetch electricity consumption",
                exc_info=True,
            )
            elec_consumption = pd.DataFrame(
                columns=["state_abbr", "year", "electricity_consumption_mwh"]
            )
            failed_queries.append("electricity_consumption")

        try:
            gas_prices = self._fetch_natural_gas_prices(start_year, end_year)
        except Exception:
            logger.warning(
                "EIA_ENERGY: failed to fetch natural gas prices",
                exc_info=True,
            )
            gas_prices = pd.DataFrame(
                columns=["state_abbr", "year", "natural_gas_price"]
            )
            failed_queries.append("natural_gas_prices")

        if failed_queries:
            logger.warning(
                "EIA_ENERGY: failed queries: %s. Proceeding with available data.",
                failed_queries,
            )

        # Merge electricity prices and consumption
        state_df = pd.merge(
            elec_prices, elec_consumption,
            on=["state_abbr", "year"], how="outer",
        )

        # Merge in natural gas prices (left join — gas data incomplete)
        state_df = pd.merge(
            state_df, gas_prices,
            on=["state_abbr", "year"], how="left",
        )

        # Add state FIPS codes
        state_df["state_fips"] = state_df["state_abbr"].map(STATE_ABBR_TO_FIPS)
        state_df = state_df.dropna(subset=["state_fips"]).copy()

        # Filter to requested years
        if years is not None:
            state_df = state_df[state_df["year"].isin(years)].copy()

        # Enforce dtypes
        state_df["state_fips"] = state_df["state_fips"].astype(str)
        state_df["state_abbr"] = state_df["state_abbr"].astype(str)
        state_df["year"] = state_df["year"].astype(int)
        state_df["electricity_price_cents_kwh"] = pd.to_numeric(
            state_df["electricity_price_cents_kwh"], errors="coerce",
        ).astype(float)
        state_df["electricity_consumption_mwh"] = pd.to_numeric(
            state_df["electricity_consumption_mwh"], errors="coerce",
        ).astype(float)
        state_df["natural_gas_price"] = pd.to_numeric(
            state_df["natural_gas_price"], errors="coerce",
        ).astype(float)

        # Select output columns
        state_df = state_df[list(self.required_columns)].copy()

        # Cache state aggregate
        vintage = f"EIA residential energy data {start_year} to {end_year}"
        self.cache_raw(
            state_df, label="eia_state_aggregate", data_vintage=vintage,
        )

        self.log_completeness(state_df)

        # --- Fetch RECS microdata (separate) ---
        try:
            recs_bytes = self._download_recs_csv()
            recs_df = self._parse_recs_csv(recs_bytes)
            self._cache_recs(
                recs_df,
                label="eia_recs_microdata",
                data_vintage=f"RECS {RECS_VINTAGE_YEAR} public use microdata",
            )
            self.log_completeness(recs_df)
            logger.info(
                "EIA_ENERGY: RECS %d — %d households, %d census divisions",
                RECS_VINTAGE_YEAR, len(recs_df),
                recs_df["census_division"].nunique(),
            )
        except Exception:
            logger.warning(
                "EIA_ENERGY: failed to fetch/parse RECS microdata",
                exc_info=True,
            )

        return state_df

    def run(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch, validate state aggregate, log completeness, return.

        Overrides base run() because we have two output tables with
        different schemas. Validates state aggregate against
        required_columns. RECS is validated separately during fetch().
        """
        logger.info("%s: starting ingestion", self.source_name)
        df = self.fetch(years=years)
        self.validate_output(df)
        return df
