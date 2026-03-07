"""Tests for the EIA Energy ingester (ingest/eia_energy.py).

All HTTP calls are mocked — no real requests to EIA API or RECS downloads.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from ingest.eia_energy import (
    CENSUS_DIVISION_NAMES,
    DWELLING_TYPE_LABELS,
    EIA_API_BASE_URL,
    EIA_ELECTRICITY_ROUTE,
    EIA_MIN_HISTORY_YEARS,
    EIA_NATURAL_GAS_ROUTE,
    EIAEnergyIngester,
    HEATING_FUEL_LABELS,
    RECS_VINTAGE_YEAR,
    STATE_ABBR_TO_FIPS,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _build_eia_api_response(records: list[dict]) -> dict:
    """Build a synthetic EIA API v2 JSON response."""
    return {
        "response": {
            "total": len(records),
            "data": records,
        }
    }


def _build_electricity_price_records(
    states: list[str] | None = None,
    years: list[int] | None = None,
) -> list[dict]:
    """Build synthetic EIA electricity price records."""
    if states is None:
        states = ["CA", "TX", "AL"]
    if years is None:
        years = [2020, 2021, 2022]
    records = []
    base_prices = {"CA": 22.5, "TX": 12.3, "AL": 13.8}
    for state in states:
        for year in years:
            records.append({
                "stateid": state,
                "period": year,
                "price": base_prices.get(state, 15.0) + (year - 2020) * 0.5,
                "sectorid": "RES",
            })
    return records


def _build_electricity_consumption_records(
    states: list[str] | None = None,
    years: list[int] | None = None,
) -> list[dict]:
    """Build synthetic EIA electricity consumption records."""
    if states is None:
        states = ["CA", "TX", "AL"]
    if years is None:
        years = [2020, 2021, 2022]
    records = []
    base_sales = {"CA": 90000.0, "TX": 140000.0, "AL": 30000.0}
    for state in states:
        for year in years:
            records.append({
                "stateid": state,
                "period": year,
                "sales": base_sales.get(state, 50000.0),
                "sectorid": "RES",
            })
    return records


def _build_gas_price_records(
    states: list[str] | None = None,
    years: list[int] | None = None,
) -> list[dict]:
    """Build synthetic EIA natural gas price records."""
    if states is None:
        states = ["CA", "TX"]  # AL has no gas data
    if years is None:
        years = [2020, 2021, 2022]
    records = []
    base_prices = {"CA": 14.5, "TX": 10.2}
    for state in states:
        for year in years:
            records.append({
                "duoarea": f"S{state}",
                "period": year,
                "value": base_prices.get(state, 12.0),
            })
    return records


def _build_recs_csv_bytes(
    n_households: int = 5,
) -> bytes:
    """Build a synthetic RECS microdata CSV file."""
    rows = []
    for i in range(1, n_households + 1):
        rows.append({
            "DOEID": 10000 + i,
            "DIVISION": (i % 9) + 1,
            "TYPEHUQ": (i % 5) + 1,
            "TOTSQFT_EN": 1200 + i * 200,
            "FUELHEAT": 5 if i % 2 == 0 else 1,  # electricity or gas
            "NHSLDMEM": (i % 4) + 1,
            "KWH": 8000 + i * 500,
            "CUFEETNG": 5000 + i * 100 if i % 2 != 0 else -2,
            "NWEIGHT": 15000.0 + i * 100,
        })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _make_json_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response with JSON content."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _make_bytes_response(content: bytes, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response with binary content."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# Pre-built fixtures
SAMPLE_RECS_CSV = _build_recs_csv_bytes(n_households=5)
SAMPLE_ELEC_PRICE_RECORDS = _build_electricity_price_records()
SAMPLE_ELEC_CONSUMPTION_RECORDS = _build_electricity_consumption_records()
SAMPLE_GAS_PRICE_RECORDS = _build_gas_price_records()


@pytest.fixture
def ingester(monkeypatch):
    """Return an EIAEnergyIngester with a fake API key."""
    monkeypatch.setenv("EIA_API_KEY", "test_key_12345")
    return EIAEnergyIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


def _mock_api_get(ingester_instance, recs_csv: bytes = SAMPLE_RECS_CSV):
    """Set up mocked api_get to return appropriate responses for each query.

    Routes:
    - electricity/retail-sales with price → electricity price data
    - electricity/retail-sales with sales → electricity consumption data
    - natural-gas/pri/sum → natural gas price data
    - RECS CSV URL → RECS microdata
    """
    elec_price_resp = _make_json_response(
        _build_eia_api_response(SAMPLE_ELEC_PRICE_RECORDS)
    )
    elec_consumption_resp = _make_json_response(
        _build_eia_api_response(SAMPLE_ELEC_CONSUMPTION_RECORDS)
    )
    gas_price_resp = _make_json_response(
        _build_eia_api_response(SAMPLE_GAS_PRICE_RECORDS)
    )
    recs_resp = _make_bytes_response(recs_csv)

    call_count = {"api": 0}

    def side_effect(url, params=None, headers=None):
        call_count["api"] += 1
        if params is None:
            params = {}

        url_str = str(url)

        # RECS CSV download
        if "recs2020" in url_str or "consumption/residential" in url_str:
            return recs_resp

        # EIA API queries
        data_field = params.get("data[0]", "")
        if EIA_ELECTRICITY_ROUTE in url_str:
            if data_field == "price":
                return elec_price_resp
            if data_field == "sales":
                return elec_consumption_resp
        if EIA_NATURAL_GAS_ROUTE in url_str:
            return gas_price_resp

        # Fallback
        return elec_price_resp

    return side_effect


# ---------------------------------------------------------------------------
# Test: Output schema — state aggregate
# ---------------------------------------------------------------------------

class TestStateAggregateSchema:
    """Verify state aggregate DataFrame has all expected columns/dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required columns."""
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        expected = {
            "state_fips", "state_abbr", "year",
            "electricity_price_cents_kwh", "electricity_consumption_mwh",
            "natural_gas_price",
        }
        assert set(df.columns) == expected

    def test_state_fips_is_string(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        assert pd.api.types.is_string_dtype(df["state_fips"])

    def test_state_abbr_is_string(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        assert pd.api.types.is_string_dtype(df["state_abbr"])

    def test_year_is_int(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        assert pd.api.types.is_integer_dtype(df["year"])

    def test_electricity_price_is_float(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        assert pd.api.types.is_float_dtype(df["electricity_price_cents_kwh"])

    def test_electricity_consumption_is_float(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        assert pd.api.types.is_float_dtype(df["electricity_consumption_mwh"])

    def test_natural_gas_price_is_float(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        assert pd.api.types.is_float_dtype(df["natural_gas_price"])


# ---------------------------------------------------------------------------
# Test: State FIPS mapping
# ---------------------------------------------------------------------------

class TestStateFIPSMapping:
    """Verify state abbreviations map to correct FIPS codes."""

    def test_california_fips(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        ca = df[df["state_abbr"] == "CA"]
        assert not ca.empty
        assert ca.iloc[0]["state_fips"] == "06"

    def test_texas_fips(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        tx = df[df["state_abbr"] == "TX"]
        assert not tx.empty
        assert tx.iloc[0]["state_fips"] == "48"

    def test_alabama_fips(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        al = df[df["state_abbr"] == "AL"]
        assert not al.empty
        assert al.iloc[0]["state_fips"] == "01"


# ---------------------------------------------------------------------------
# Test: Price parsing
# ---------------------------------------------------------------------------

class TestPriceParsing:
    """Verify electricity prices are correctly parsed as cents/kWh."""

    def test_known_price_value(self, ingester, tmp_raw_dir):
        """California 2020 price should be 22.5 cents/kWh."""
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020])
        ca_2020 = df[(df["state_abbr"] == "CA") & (df["year"] == 2020)]
        assert not ca_2020.empty
        assert abs(ca_2020.iloc[0]["electricity_price_cents_kwh"] - 22.5) < 0.01


# ---------------------------------------------------------------------------
# Test: Consumption parsing
# ---------------------------------------------------------------------------

class TestConsumptionParsing:
    """Verify electricity consumption is correctly parsed as MWh."""

    def test_known_consumption_value(self, ingester, tmp_raw_dir):
        """Texas 2020 consumption should be 140000 MWh."""
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020])
        tx_2020 = df[(df["state_abbr"] == "TX") & (df["year"] == 2020)]
        assert not tx_2020.empty
        assert abs(tx_2020.iloc[0]["electricity_consumption_mwh"] - 140000.0) < 0.01


# ---------------------------------------------------------------------------
# Test: Natural gas — present and missing
# ---------------------------------------------------------------------------

class TestNaturalGas:
    """Verify natural gas prices are captured or NaN as appropriate."""

    def test_gas_price_present(self, ingester, tmp_raw_dir):
        """States with gas data have numeric natural_gas_price."""
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020])
        ca_2020 = df[(df["state_abbr"] == "CA") & (df["year"] == 2020)]
        assert not ca_2020.empty
        assert pd.notna(ca_2020.iloc[0]["natural_gas_price"])

    def test_gas_price_missing_is_nan(self, ingester, tmp_raw_dir):
        """States without gas data have NaN for natural_gas_price (not 0)."""
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020])
        # AL is not in our synthetic gas data
        al_2020 = df[(df["state_abbr"] == "AL") & (df["year"] == 2020)]
        assert not al_2020.empty
        assert pd.isna(al_2020.iloc[0]["natural_gas_price"])


# ---------------------------------------------------------------------------
# Test: Multi-year data
# ---------------------------------------------------------------------------

class TestMultiYearData:
    """Verify the ingester fetches data across the full year range."""

    def test_three_years_present(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020, 2021, 2022])
        assert set(df["year"].unique()) == {2020, 2021, 2022}


# ---------------------------------------------------------------------------
# Test: RECS schema
# ---------------------------------------------------------------------------

class TestRECSSchema:
    """Verify RECS DataFrame has all expected columns with correct dtypes."""

    def test_recs_columns_present(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        expected = set(ingester.recs_required_columns.keys())
        assert set(recs.columns) == expected

    def test_recs_household_id_is_string(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        assert pd.api.types.is_string_dtype(recs["household_id"])


# ---------------------------------------------------------------------------
# Test: RECS dwelling type parsing
# ---------------------------------------------------------------------------

class TestRECSDwellingType:
    """Verify dwelling type codes are mapped to human-readable labels."""

    def test_dwelling_type_labels_present(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        # All labels should be from our known mapping
        valid_labels = set(DWELLING_TYPE_LABELS.values()) | {"Unknown"}
        assert set(recs["dwelling_type"].unique()).issubset(valid_labels)

    def test_dwelling_type_code_preserved(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        assert "dwelling_type_code" in recs.columns
        # Should have valid numeric codes
        assert recs["dwelling_type_code"].notna().all()


# ---------------------------------------------------------------------------
# Test: RECS heating fuel parsing
# ---------------------------------------------------------------------------

class TestRECSHeatingFuel:
    """Verify heating fuel codes are mapped to labels."""

    def test_heating_fuel_labels_present(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        valid_labels = set(HEATING_FUEL_LABELS.values()) | {"Unknown"}
        assert set(recs["heating_fuel"].unique()).issubset(valid_labels)

    def test_heating_fuel_code_preserved(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        assert "heating_fuel_code" in recs.columns


# ---------------------------------------------------------------------------
# Test: RECS census division parsing
# ---------------------------------------------------------------------------

class TestRECSCensusDivision:
    """Verify census division codes are mapped to names."""

    def test_census_division_names_present(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        valid_names = set(CENSUS_DIVISION_NAMES.values()) | {"Unknown"}
        assert set(recs["census_division_name"].unique()).issubset(valid_names)


# ---------------------------------------------------------------------------
# Test: RECS sample weight preservation
# ---------------------------------------------------------------------------

class TestRECSSampleWeight:
    """Verify RECS sample weights are preserved."""

    def test_sample_weight_present_and_positive(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        assert "sample_weight" in recs.columns
        assert (recs["sample_weight"] > 0).all()
        # Verify the weights match our synthetic fixture
        assert abs(recs["sample_weight"].min() - 15100.0) < 0.01


# ---------------------------------------------------------------------------
# Test: RECS raw codes preserved
# ---------------------------------------------------------------------------

class TestRECSRawCodes:
    """Verify raw numeric codes are preserved alongside labels."""

    def test_both_code_and_label_columns_exist(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        assert "dwelling_type" in recs.columns
        assert "dwelling_type_code" in recs.columns
        assert "heating_fuel" in recs.columns
        assert "heating_fuel_code" in recs.columns


# ---------------------------------------------------------------------------
# Test: RECS vintage documented
# ---------------------------------------------------------------------------

class TestRECSVintage:
    """Verify the RECS vintage is documented in metadata."""

    def test_recs_year_column(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        recs = pd.read_parquet(
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        )
        assert (recs["recs_year"] == RECS_VINTAGE_YEAR).all()

    def test_recs_vintage_in_metadata(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        meta_path = (
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata_metadata.json"
        )
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "recs_vintage" in meta
        assert meta["recs_vintage"] == str(RECS_VINTAGE_YEAR)


# ---------------------------------------------------------------------------
# Test: Separation of outputs
# ---------------------------------------------------------------------------

class TestOutputSeparation:
    """Verify state aggregate and RECS are cached as separate files."""

    def test_two_separate_parquet_files(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        state_file = tmp_raw_dir / "eia_energy" / "eia_state_aggregate.parquet"
        recs_file = tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        assert state_file.exists()
        assert recs_file.exists()

    def test_state_aggregate_location(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        assert (
            tmp_raw_dir / "eia_energy" / "eia_state_aggregate.parquet"
        ).exists()

    def test_recs_location(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        assert (
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata.parquet"
        ).exists()

    def test_separate_metadata_sidecars(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        state_meta = (
            tmp_raw_dir / "eia_energy" / "eia_state_aggregate_metadata.json"
        )
        recs_meta = (
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata_metadata.json"
        )
        assert state_meta.exists()
        assert recs_meta.exists()


# ---------------------------------------------------------------------------
# Test: API key handling
# ---------------------------------------------------------------------------

class TestAPIKeyHandling:
    """Verify clear error if EIA_API_KEY is missing."""

    def test_missing_api_key_raises(self, monkeypatch, tmp_raw_dir):
        monkeypatch.delenv("EIA_API_KEY", raising=False)
        ing = EIAEnergyIngester()
        with pytest.raises(RuntimeError, match="EIA_API_KEY"):
            ing.fetch(years=[2020])

    def test_api_key_in_request(self, ingester, tmp_raw_dir):
        """Verify the API key is included in request params."""
        captured_params = {}

        def capture_api_get(url, params=None, headers=None):
            if params and "api_key" in params:
                captured_params.update(params)
            return _make_json_response(
                _build_eia_api_response(SAMPLE_ELEC_PRICE_RECORDS)
            )

        # Mock RECS download too
        with patch.object(ingester, "api_get", side_effect=capture_api_get):
            with patch.object(ingester, "_download_recs_csv", return_value=SAMPLE_RECS_CSV):
                with patch.object(ingester, "_parse_recs_csv", return_value=pd.DataFrame(
                    columns=list(ingester.recs_required_columns)
                )):
                    ingester.fetch(years=[2020])

        assert "api_key" in captured_params
        assert captured_params["api_key"] == "test_key_12345"


# ---------------------------------------------------------------------------
# Test: Ingest purity check
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed in output."""

    def test_no_derived_columns_state(self, ingester, tmp_raw_dir):
        """State aggregate must not contain CCI-derived columns."""
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020])
        forbidden = {
            "climate_attributed_cost", "consumption_per_household",
            "consumption_per_baseline_dwelling", "hdd_regression_coef",
            "price_yoy_change", "rate_case_break", "county_fips",
            "hdd_anomaly", "cdd_anomaly", "degree_day", "score",
            "percentile", "cci_dollar",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set_state(self, ingester, tmp_raw_dir):
        """State aggregate columns are EXACTLY the required set."""
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            df = ingester.fetch(years=[2020])
        expected = set(ingester.required_columns.keys())
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "state_fips": ["06"],
            "state_abbr": ["CA"],
            "year": [2020],
            "electricity_price_cents_kwh": [22.5],
            "electricity_consumption_mwh": [90000.0],
            "natural_gas_price": [14.5],
            "climate_attributed_cost": [500.0],  # FORBIDDEN
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Metadata sidecar — state
# ---------------------------------------------------------------------------

class TestStateMetadata:
    """Verify state aggregate metadata sidecar content."""

    def test_metadata_content(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        meta_path = (
            tmp_raw_dir / "eia_energy" / "eia_state_aggregate_metadata.json"
        )
        meta = json.loads(meta_path.read_text())
        assert meta["source"] == "EIA_ENERGY"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "attributed"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta


# ---------------------------------------------------------------------------
# Test: Metadata sidecar — RECS
# ---------------------------------------------------------------------------

class TestRECSMetadata:
    """Verify RECS metadata sidecar content."""

    def test_metadata_content(self, ingester, tmp_raw_dir):
        with patch.object(ingester, "api_get", side_effect=_mock_api_get(ingester)):
            ingester.fetch(years=[2020])
        meta_path = (
            tmp_raw_dir / "eia_energy" / "eia_recs_microdata_metadata.json"
        )
        meta = json.loads(meta_path.read_text())
        assert meta["source"] == "EIA_RECS"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "attributed"
        assert meta["recs_vintage"] == str(RECS_VINTAGE_YEAR)


# ---------------------------------------------------------------------------
# Test: Partial failure — states
# ---------------------------------------------------------------------------

class TestPartialFailure:
    """Verify partial API failures don't abort entire run."""

    def test_gas_failure_still_returns_elec_data(self, ingester, tmp_raw_dir):
        """If natural gas query fails, electricity data is still cached."""
        call_count = {"n": 0}

        def side_effect(url, params=None, headers=None):
            call_count["n"] += 1
            if params is None:
                params = {}

            url_str = str(url)

            # RECS download
            if "recs2020" in url_str or "consumption/residential" in url_str:
                return _make_bytes_response(SAMPLE_RECS_CSV)

            data_field = params.get("data[0]", "")

            # Electricity queries succeed
            if EIA_ELECTRICITY_ROUTE in url_str:
                if data_field == "price":
                    return _make_json_response(
                        _build_eia_api_response(SAMPLE_ELEC_PRICE_RECORDS)
                    )
                if data_field == "sales":
                    return _make_json_response(
                        _build_eia_api_response(SAMPLE_ELEC_CONSUMPTION_RECORDS)
                    )

            # Gas query fails
            if EIA_NATURAL_GAS_ROUTE in url_str:
                raise httpx.HTTPStatusError(
                    "500 Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )

            return _make_json_response(_build_eia_api_response([]))

        with patch.object(ingester, "api_get", side_effect=side_effect):
            df = ingester.fetch(years=[2020])

        # Should still have electricity data
        assert not df.empty
        assert "electricity_price_cents_kwh" in df.columns
        assert df["electricity_price_cents_kwh"].notna().any()


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic triggers on HTTP 500/503."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = _make_json_response(
            _build_eia_api_response(SAMPLE_ELEC_PRICE_RECORDS)
        )

        call_count = {"n": 0}

        def mock_get(url, params=None, headers=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return fail_resp
            return ok_resp

        with patch.object(ingester, "_client", MagicMock()):
            ingester._client.get = mock_get
            ingester._last_call_time = 0.0
            resp = ingester.api_get(
                f"{EIA_API_BASE_URL}/{EIA_ELECTRICITY_ROUTE}"
            )

        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Test: Completeness logging
# ---------------------------------------------------------------------------

class TestCompletenessLogging:
    """Verify log_completeness reports state count."""

    def test_completeness_reports_states(self, ingester, tmp_raw_dir, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            with patch.object(
                ingester, "api_get", side_effect=_mock_api_get(ingester)
            ):
                ingester.fetch(years=[2020])

        # Should mention states in completeness log
        assert any(
            "states" in msg.lower() or "completeness" in msg.lower()
            for msg in caplog.messages
        )


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "eia_energy"

    def test_confidence(self, ingester):
        assert ingester.confidence == "A"

    def test_attribution(self, ingester):
        assert ingester.attribution == "attributed"

    def test_calls_per_second(self, ingester):
        assert ingester.calls_per_second <= 2.0
