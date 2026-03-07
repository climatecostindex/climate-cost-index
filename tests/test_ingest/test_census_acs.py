"""Tests for the Census ACS ingester (ingest/census_acs.py).

All HTTP calls are mocked — no real requests to the Census API.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

from ingest.census_acs import (
    CENSUS_ACS_BASE_URL,
    CENSUS_SUPPRESSION_VALUE,
    CensusACSIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Matches the real Census API response format: first row = headers, rest = data
SAMPLE_RESPONSE = [
    ["B25001_001E", "B25003_002E", "B01003_001E", "B19013_001E", "state", "county"],
    ["24457", "16832", "58761", "68315", "01", "001"],
    ["125113", "70708", "233420", "71039", "01", "003"],
    ["11673", "5858", "24877", "39712", "01", "005"],
]

# Response with a suppressed median income value
SAMPLE_WITH_SUPPRESSION = [
    ["B25001_001E", "B25003_002E", "B01003_001E", "B19013_001E", "state", "county"],
    ["24457", "16832", "58761", "68315", "01", "001"],
    ["500", "200", "1000", "-666666666", "02", "270"],  # suppressed income
    ["300", "100", "800", None, "15", "005"],  # null income
]

# Response with short state/county codes needing zero-pad
SAMPLE_SHORT_CODES = [
    ["B25001_001E", "B25003_002E", "B01003_001E", "B19013_001E", "state", "county"],
    ["24457", "16832", "58761", "68315", "1", "1"],     # state=1, county=1 → "01001"
    ["500000", "300000", "10000000", "75000", "6", "37"],  # state=6, county=37 → "06037"
]


def _make_response(data: list) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def ingester():
    """Return a fresh CensusACSIngester instance."""
    return CensusACSIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Test: Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has all expected columns with correct dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required columns."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        expected = {"fips", "year", "total_housing_units", "owner_occupied_units",
                    "population", "median_household_income"}
        assert set(df.columns) == expected

    def test_fips_is_string(self, ingester, tmp_raw_dir):
        """FIPS column is string dtype."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        assert pd.api.types.is_string_dtype(df["fips"])

    def test_year_is_int(self, ingester, tmp_raw_dir):
        """Year column is integer dtype."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        assert pd.api.types.is_integer_dtype(df["year"])

    def test_numeric_cols_are_float(self, ingester, tmp_raw_dir):
        """Housing, population, and income columns are float dtype."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        for col in ["total_housing_units", "owner_occupied_units",
                     "population", "median_household_income"]:
            assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"


# ---------------------------------------------------------------------------
# Test: FIPS construction
# ---------------------------------------------------------------------------

class TestFIPSConstruction:
    """Verify 5-digit FIPS is correctly built from state + county codes."""

    def test_standard_codes(self, ingester, tmp_raw_dir):
        """State '01' + county '001' → '01001'."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        assert "01001" in df["fips"].values
        assert "01003" in df["fips"].values
        assert "01005" in df["fips"].values

    def test_short_codes_zero_padded(self, ingester, tmp_raw_dir):
        """State '6' + county '37' → '06037', state '1' + county '1' → '01001'."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_SHORT_CODES)):
            df = ingester.fetch(years=[2022])

        assert "01001" in df["fips"].values
        assert "06037" in df["fips"].values

    def test_all_fips_5_digit(self, ingester, tmp_raw_dir):
        """Every FIPS in output matches the 5-digit pattern."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        assert df["fips"].str.match(r"^\d{5}$").all()


# ---------------------------------------------------------------------------
# Test: Column renaming
# ---------------------------------------------------------------------------

class TestColumnRenaming:
    """Verify ACS variable codes are renamed to human-readable names."""

    def test_no_acs_codes_in_output(self, ingester, tmp_raw_dir):
        """Output must not contain raw ACS variable code column names."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        acs_codes = {"B25001_001E", "B25003_002E", "B01003_001E", "B19013_001E"}
        assert acs_codes.isdisjoint(set(df.columns))

    def test_human_readable_names_present(self, ingester, tmp_raw_dir):
        """Output uses the renamed column names."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        for col in ["total_housing_units", "owner_occupied_units",
                     "population", "median_household_income"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Test: Null handling
# ---------------------------------------------------------------------------

class TestNullHandling:
    """Verify Census suppression values become NaN, not dropped rows."""

    def test_suppression_becomes_nan(self, ingester, tmp_raw_dir):
        """The -666666666 suppression sentinel is converted to NaN."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_WITH_SUPPRESSION)):
            df = ingester.fetch(years=[2022])

        suppressed_row = df[df["fips"] == "02270"]
        assert len(suppressed_row) == 1
        assert np.isnan(suppressed_row.iloc[0]["median_household_income"])

    def test_null_becomes_nan(self, ingester, tmp_raw_dir):
        """JSON null values are converted to NaN."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_WITH_SUPPRESSION)):
            df = ingester.fetch(years=[2022])

        null_row = df[df["fips"] == "15005"]
        assert len(null_row) == 1
        assert np.isnan(null_row.iloc[0]["median_household_income"])

    def test_rows_not_dropped(self, ingester, tmp_raw_dir):
        """Rows with suppressed/null values are preserved, not dropped."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_WITH_SUPPRESSION)):
            df = ingester.fetch(years=[2022])

        assert len(df) == 3  # All 3 data rows preserved


# ---------------------------------------------------------------------------
# Test: Ingest purity (no derived metrics)
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed."""

    def test_no_extra_columns(self, ingester, tmp_raw_dir):
        """Output must not contain derived columns."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        forbidden = {"per_capita_income", "housing_density", "yoy_change",
                     "income_rank", "percentile", "score", "owner_pct"}
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        expected = {"fips", "year", "total_housing_units", "owner_occupied_units",
                    "population", "median_household_income"}
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "fips": ["01001"],
            "year": [2022],
            "total_housing_units": [24457.0],
            "owner_occupied_units": [16832.0],
            "population": [58761.0],
            "median_household_income": [68315.0],
            "per_capita_income": [999.0],
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir):
        """A _metadata.json file is written next to each parquet file."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            ingester.fetch(years=[2022])

        assert (tmp_raw_dir / "census_acs" / "census_acs_2022_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution values."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            ingester.fetch(years=[2022])

        meta_path = tmp_raw_dir / "census_acs" / "census_acs_2022_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "CENSUS_ACS"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "none"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_combined_metadata_written(self, ingester, tmp_raw_dir):
        """The combined census_acs_all file gets a metadata sidecar."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            ingester.fetch(years=[2022])

        assert (tmp_raw_dir / "census_acs" / "census_acs_all.parquet").exists()
        assert (tmp_raw_dir / "census_acs" / "census_acs_all_metadata.json").exists()


# ---------------------------------------------------------------------------
# Test: Completeness logging
# ---------------------------------------------------------------------------

class TestCompletenessLogging:
    """Verify log_completeness reports county coverage."""

    def test_completeness_logged(self, ingester, tmp_raw_dir, caplog):
        """run() logs county count and coverage percentage."""
        import logging
        with caplog.at_level(logging.INFO):
            with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
                ingester.run(years=[2022])

        assert any("counties" in msg.lower() for msg in caplog.messages)

    def test_county_count(self, ingester, tmp_raw_dir):
        """Sample data has 3 unique counties."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2022])

        assert df["fips"].nunique() == 3


# ---------------------------------------------------------------------------
# Test: API key handling
# ---------------------------------------------------------------------------

class TestAPIKeyHandling:
    """Verify the request includes/excludes the API key appropriately."""

    def test_key_included_when_set(self, ingester, tmp_raw_dir, monkeypatch):
        """Request params include 'key' when CENSUS_API_KEY is set."""
        monkeypatch.setenv("CENSUS_API_KEY", "test_key_12345")

        captured_params = {}

        def mock_api_get(url, params=None, headers=None):
            captured_params.update(params or {})
            return _make_response(SAMPLE_RESPONSE)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            ingester.fetch(years=[2022])

        assert "key" in captured_params
        assert captured_params["key"] == "test_key_12345"

    def test_key_omitted_when_unset(self, ingester, tmp_raw_dir, monkeypatch):
        """Request params omit 'key' when CENSUS_API_KEY is empty."""
        monkeypatch.setenv("CENSUS_API_KEY", "")

        captured_params = {}

        def mock_api_get(url, params=None, headers=None):
            captured_params.update(params or {})
            return _make_response(SAMPLE_RESPONSE)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            ingester.fetch(years=[2022])

        assert "key" not in captured_params


# ---------------------------------------------------------------------------
# Test: Multi-year fetch & partial failure
# ---------------------------------------------------------------------------

class TestMultiYearFetch:
    """Verify multi-year fetching and partial failure handling."""

    def test_multiple_years_fetched(self, ingester, tmp_raw_dir):
        """Data from multiple years is concatenated."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            df = ingester.fetch(years=[2021, 2022])

        assert set(df["year"].unique()) == {2021, 2022}
        # 3 counties × 2 years
        assert len(df) == 6

    def test_per_year_caching(self, ingester, tmp_raw_dir):
        """Each year gets its own parquet + metadata file."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RESPONSE)):
            ingester.fetch(years=[2021, 2022])

        assert (tmp_raw_dir / "census_acs" / "census_acs_2021.parquet").exists()
        assert (tmp_raw_dir / "census_acs" / "census_acs_2022.parquet").exists()
        assert (tmp_raw_dir / "census_acs" / "census_acs_2021_metadata.json").exists()
        assert (tmp_raw_dir / "census_acs" / "census_acs_2022_metadata.json").exists()

    def test_partial_year_failure(self, ingester, tmp_raw_dir):
        """Data from successful years returned when one year fails."""
        def mock_api_get(url, params=None, headers=None):
            if "2021" in url:
                raise httpx.TransportError("Connection refused")
            return _make_response(SAMPLE_RESPONSE)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch(years=[2021, 2022])

        assert not df.empty
        assert set(df["year"].unique()) == {2022}
        assert (tmp_raw_dir / "census_acs" / "census_acs_2022.parquet").exists()

    def test_all_years_fail_returns_empty(self, ingester, tmp_raw_dir):
        """Returns empty DataFrame when every year fails."""
        with patch.object(ingester, "api_get", side_effect=httpx.TransportError("fail")):
            df = ingester.fetch(years=[2022])

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic on transient HTTP failures."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = _make_response(SAMPLE_RESPONSE)

        call_count = 0

        def mock_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fail_resp
            return ok_resp

        with patch.object(ingester, "_client", MagicMock()):
            ingester._client.get = mock_get
            ingester._last_call_time = 0.0
            df = ingester._fetch_year(2022)

        assert not df.empty
        assert call_count == 2

    def test_retries_on_503(self, ingester, tmp_raw_dir):
        """Retries after HTTP 503."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        ok_resp = _make_response(SAMPLE_RESPONSE)

        call_count = 0

        def mock_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return fail_resp
            return ok_resp

        with patch.object(ingester, "_client", MagicMock()):
            ingester._client.get = mock_get
            ingester._last_call_time = 0.0
            df = ingester._fetch_year(2022)

        assert not df.empty
        assert call_count == 3
