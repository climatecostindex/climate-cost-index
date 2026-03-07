"""Tests for the CDC EPHT ingester (ingest/cdc_epht.py).

All HTTP calls are mocked — no real requests to the CDC EPHT API.

The CDC EPHT API uses:
  - GET /contentareas/json → list content areas
  - GET /indicators/{contentAreaId} → list indicators
  - GET /measures/{indicatorId} → list measures
  - GET /geographicTypes/{measureId} → available geo types
  - GET /geographicItems/{measureId}/{geoTypeId}/0 → available states/counties
  - GET /temporalItems/{measureId}/{geoTypeId}/ALL/ALL → available years
  - GET /stratificationlevel/{measureId}/{geoTypeId}/{isSmoothed} → strat levels
  - POST /getCoreHolder/{measureId}/{stratLevelId}/{smoothing}/0 → data
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

from ingest.cdc_epht import (
    CDCEPHTIngester,
    CONTENT_AREAS_URL,
    INDICATORS_URL,
    MEASURES_URL,
    GEO_TYPES_URL,
    GEO_ITEMS_URL,
    TEMPORAL_ITEMS_URL,
    STRAT_LEVELS_URL,
    DATA_URL,
    GEO_TYPE_COUNTY,
    GEO_TYPE_STATE,
    IS_SMOOTHED,
    normalize_county_fips,
    normalize_state_fips,
)


# ---------------------------------------------------------------------------
# Fixtures & synthetic API responses
# ---------------------------------------------------------------------------

SAMPLE_CONTENT_AREAS = [
    {"id": "11", "name": "Air Quality"},
    {"id": "35", "name": "Heat & Heat-related Illness (HRI)"},
    {"id": "9", "name": "Cancer"},
]

SAMPLE_INDICATORS = [
    {"id": "89", "name": "Emergency Department Visits for HRI"},
    {"id": "225", "name": "Heat & Health Index"},
]

SAMPLE_MEASURES = [
    {"id": "438", "name": "Annual Number of Emergency Department Visits for HRI"},
    {"id": "440", "name": "Age-adjusted Rate of ED Visits for HRI per 100,000"},
    {"id": "439", "name": "Crude Rate of ED Visits for HRI per 100,000"},
]

SAMPLE_GEO_TYPES_STATE_ONLY = [
    {"id": 960, "geographicTypeId": 1, "geographicType": "State",
     "smoothingLevelId": 1, "smoothingLevel": "No Smoothing Available"},
]

SAMPLE_GEO_TYPES_BOTH = [
    {"id": 960, "geographicTypeId": 1, "geographicType": "State",
     "smoothingLevelId": 1},
    {"id": 961, "geographicTypeId": 2, "geographicType": "County",
     "smoothingLevelId": 1},
]

SAMPLE_GEO_ITEMS_STATES = [
    {"parentGeographicId": 1, "parentName": "Alabama", "id": 1},
    {"parentGeographicId": 6, "parentName": "California", "id": 6},
    {"parentGeographicId": 12, "parentName": "Florida", "id": 12},
]

SAMPLE_GEO_ITEMS_COUNTIES = [
    {"parentGeographicId": 1001, "parentName": "Autauga County", "id": 1001},
    {"parentGeographicId": 1003, "parentName": "Baldwin County", "id": 1003},
    {"parentGeographicId": 6037, "parentName": "Los Angeles County", "id": 6037},
]

SAMPLE_TEMPORAL_ITEMS = [
    {"temporalId": 2020, "temporal": "2020", "temporalTypeId": 1},
    {"temporalId": 2021, "temporal": "2021", "temporalTypeId": 1},
    {"temporalId": 2022, "temporal": "2022", "temporalTypeId": 1},
]

SAMPLE_STRAT_LEVELS = [
    {"id": 1, "name": "State", "abbreviation": "ST",
     "geographicTypeId": 1, "stratificationType": []},
]

# State-level tableResult records
SAMPLE_STATE_TABLE_RESULT = [
    {"geoId": "01", "geo": "Alabama", "temporal": "2020",
     "temporalId": 2020, "dataValue": "152", "suppressionFlag": "0",
     "geographicTypeId": 1},
    {"geoId": "01", "geo": "Alabama", "temporal": "2021",
     "temporalId": 2021, "dataValue": "187", "suppressionFlag": "0",
     "geographicTypeId": 1},
    {"geoId": "01", "geo": "Alabama", "temporal": "2022",
     "temporalId": 2022, "dataValue": "95", "suppressionFlag": "1",
     "geographicTypeId": 1},  # suppressed
    {"geoId": "06", "geo": "California", "temporal": "2020",
     "temporalId": 2020, "dataValue": "5079", "suppressionFlag": "0",
     "geographicTypeId": 1},
    {"geoId": "6", "geo": "California", "temporal": "2021",
     "temporalId": 2021, "dataValue": "", "suppressionFlag": "0",
     "geographicTypeId": 1},  # empty → suppressed via _parse_count
    {"geoId": "12", "geo": "Florida", "temporal": "2020",
     "temporalId": 2020, "dataValue": None, "suppressionFlag": "0",
     "geographicTypeId": 1},  # null value
    {"geoId": "12", "geo": "Florida", "temporal": "2021",
     "temporalId": 2021, "dataValue": "4500", "suppressionFlag": "0",
     "geographicTypeId": 1},
]

# County-level tableResult records
SAMPLE_COUNTY_TABLE_RESULT = [
    {"geoId": "01001", "geo": "Autauga County", "temporal": "2020",
     "temporalId": 2020, "dataValue": "12", "suppressionFlag": "0",
     "geographicTypeId": 2},
    {"geoId": "01003", "geo": "Baldwin County", "temporal": "2020",
     "temporalId": 2020, "dataValue": "25", "suppressionFlag": "0",
     "geographicTypeId": 2},
    {"geoId": "01003", "geo": "Baldwin County", "temporal": "2021",
     "temporalId": 2021, "dataValue": "Suppressed", "suppressionFlag": "1",
     "geographicTypeId": 2},  # suppressed
    {"geoId": "6037", "geo": "Los Angeles County", "temporal": "2020",
     "temporalId": 2020, "dataValue": "580", "suppressionFlag": "0",
     "geographicTypeId": 2},
]


def _make_response(data: list | dict) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _setup_full_mock(
    ingester: CDCEPHTIngester,
    geo_types: list | None = None,
    state_table_result: list | None = None,
    county_table_result: list | None = None,
    geo_items_states: list | None = None,
    geo_items_counties: list | None = None,
    temporal_items: list | None = None,
) -> None:
    """Patch api_get and client.post to return appropriate mock responses.

    This patches:
    - api_get for all GET endpoints (discovery, geo items, temporals, strat levels)
    - client.post for the getCoreHolder data endpoint
    """
    if geo_types is None:
        geo_types = SAMPLE_GEO_TYPES_STATE_ONLY
    if state_table_result is None:
        state_table_result = SAMPLE_STATE_TABLE_RESULT
    if county_table_result is None:
        county_table_result = SAMPLE_COUNTY_TABLE_RESULT
    if geo_items_states is None:
        geo_items_states = SAMPLE_GEO_ITEMS_STATES
    if geo_items_counties is None:
        geo_items_counties = SAMPLE_GEO_ITEMS_COUNTIES
    if temporal_items is None:
        temporal_items = SAMPLE_TEMPORAL_ITEMS

    def mock_api_get(url: str, params=None, headers=None) -> httpx.Response:
        if url == CONTENT_AREAS_URL:
            return _make_response(SAMPLE_CONTENT_AREAS)
        if url.startswith(INDICATORS_URL + "/"):
            return _make_response(SAMPLE_INDICATORS)
        if url.startswith(MEASURES_URL + "/"):
            return _make_response(SAMPLE_MEASURES)
        if url.startswith(GEO_TYPES_URL + "/"):
            return _make_response(geo_types)
        if url.startswith(GEO_ITEMS_URL + "/"):
            # Determine geo type from URL
            if f"/{GEO_TYPE_COUNTY}/" in url:
                return _make_response(geo_items_counties)
            return _make_response(geo_items_states)
        if url.startswith(TEMPORAL_ITEMS_URL + "/"):
            return _make_response(temporal_items)
        if url.startswith(STRAT_LEVELS_URL + "/"):
            return _make_response(SAMPLE_STRAT_LEVELS)
        return _make_response([])

    # Mock the POST for getCoreHolder
    def mock_post(url: str, json=None, timeout=None) -> httpx.Response:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        # Determine which result to return based on geographicTypeIdFilter
        if json and json.get("geographicTypeIdFilter") == GEO_TYPE_COUNTY:
            resp.json.return_value = {"tableResult": county_table_result}
        else:
            resp.json.return_value = {"tableResult": state_table_result}
        return resp

    ingester.api_get = MagicMock(side_effect=mock_api_get)
    # Mock the client property to intercept POST calls
    mock_client = MagicMock()
    mock_client.post = MagicMock(side_effect=mock_post)
    ingester._client = mock_client


@pytest.fixture
def ingester() -> CDCEPHTIngester:
    """Return a fresh CDCEPHTIngester instance."""
    return CDCEPHTIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Test: Geographic resolution tagging
# ---------------------------------------------------------------------------

class TestGeoResolutionTagging:
    """Verify every record is tagged with its geographic resolution."""

    def test_county_records_tagged_county(self, ingester, tmp_raw_dir):
        """County-level records have geo_resolution = 'county'."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020, 2021, 2022])

        county_rows = df[df["geo_resolution"] == "county"]
        assert len(county_rows) > 0
        assert (county_rows["geo_resolution"] == "county").all()

    def test_county_records_have_5digit_fips(self, ingester, tmp_raw_dir):
        """County-level records have 5-digit FIPS in the fips column."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020, 2021])

        county_rows = df[df["geo_resolution"] == "county"]
        assert county_rows["fips"].str.match(r"^\d{5}$").all()

    def test_state_records_tagged_state(self, ingester, tmp_raw_dir):
        """State-level records have geo_resolution = 'state'."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021, 2022])

        state_rows = df[df["geo_resolution"] == "state"]
        assert len(state_rows) > 0
        assert (state_rows["geo_resolution"] == "state").all()

    def test_state_records_have_2digit_fips(self, ingester, tmp_raw_dir):
        """State-level records have 2-digit FIPS in the fips column."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021, 2022])

        state_rows = df[df["geo_resolution"] == "state"]
        assert state_rows["fips"].str.match(r"^\d{2}$").all()

    def test_mixed_resolution_output(self, ingester, tmp_raw_dir):
        """Output contains both county and state records."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020, 2021, 2022])

        assert "county" in df["geo_resolution"].values
        assert "state" in df["geo_resolution"].values

    def test_state_fips_always_present(self, ingester, tmp_raw_dir):
        """state_fips is populated for all records as 2-digit zero-padded."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020, 2021, 2022])

        assert df["state_fips"].notna().all()
        assert df["state_fips"].str.match(r"^\d{2}$").all()


# ---------------------------------------------------------------------------
# Test: Suppressed and missing data
# ---------------------------------------------------------------------------

class TestSuppressedData:
    """Verify suppressed values become NaN, not 0, not dropped."""

    def test_suppression_flag_to_nan(self, ingester, tmp_raw_dir):
        """Records with suppressionFlag != '0' have NaN heat_ed_visits."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021, 2022])

        state_rows = df[df["geo_resolution"] == "state"]
        # Alabama 2022 has suppressionFlag=1
        row_2022 = state_rows[
            (state_rows["state_fips"] == "01") & (state_rows["year"] == 2022)
        ]
        assert len(row_2022) == 1
        assert np.isnan(row_2022.iloc[0]["heat_ed_visits"])

    def test_empty_string_to_nan(self, ingester, tmp_raw_dir):
        """Empty string dataValue becomes NaN."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021])

        state_rows = df[df["geo_resolution"] == "state"]
        ca_2021 = state_rows[
            (state_rows["state_fips"] == "06") & (state_rows["year"] == 2021)
        ]
        assert len(ca_2021) == 1
        assert np.isnan(ca_2021.iloc[0]["heat_ed_visits"])

    def test_null_value_to_nan(self, ingester, tmp_raw_dir):
        """None/null dataValue becomes NaN."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021])

        state_rows = df[df["geo_resolution"] == "state"]
        fl_2020 = state_rows[
            (state_rows["state_fips"] == "12") & (state_rows["year"] == 2020)
        ]
        assert len(fl_2020) == 1
        assert np.isnan(fl_2020.iloc[0]["heat_ed_visits"])

    def test_suppressed_county_to_nan(self, ingester, tmp_raw_dir):
        """County record with suppressionFlag=1 has NaN heat_ed_visits."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020, 2021])

        county_rows = df[df["geo_resolution"] == "county"]
        baldwin_2021 = county_rows[
            (county_rows["fips"] == "01003") & (county_rows["year"] == 2021)
        ]
        assert len(baldwin_2021) == 1
        assert np.isnan(baldwin_2021.iloc[0]["heat_ed_visits"])

    def test_suppressed_rows_preserved(self, ingester, tmp_raw_dir):
        """Rows with suppressed visit counts are retained, not filtered out."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021, 2022])

        state_rows = df[df["geo_resolution"] == "state"]
        al_rows = state_rows[state_rows["state_fips"] == "01"]
        # All 3 Alabama records (including suppressed 2022) should be present
        assert len(al_rows) == 3

    def test_population_is_nan(self, ingester, tmp_raw_dir):
        """Population is NaN since this measure doesn't include it."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020])

        assert df["population"].isna().all()


# ---------------------------------------------------------------------------
# Test: FIPS normalization
# ---------------------------------------------------------------------------

class TestFIPSNormalization:
    """Verify FIPS codes are properly zero-padded."""

    def test_county_fips_zero_padded(self, ingester, tmp_raw_dir):
        """County geoId '6037' becomes '06037'."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020])

        county_rows = df[df["geo_resolution"] == "county"]
        assert "06037" in county_rows["fips"].values

    def test_county_fips_already_padded(self, ingester, tmp_raw_dir):
        """County geoId '01001' stays '01001'."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020])

        county_rows = df[df["geo_resolution"] == "county"]
        assert "01001" in county_rows["fips"].values

    def test_state_fips_zero_padded(self, ingester, tmp_raw_dir):
        """State geoId '6' becomes '06'."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021])

        state_rows = df[df["geo_resolution"] == "state"]
        assert "06" in state_rows["fips"].values

    def test_normalize_county_fips_function(self):
        """Direct test of normalize_county_fips helper."""
        assert normalize_county_fips(1001) == "01001"
        assert normalize_county_fips("1001") == "01001"
        assert normalize_county_fips(6037) == "06037"
        assert normalize_county_fips("06037") == "06037"
        assert normalize_county_fips(6037.0) == "06037"

    def test_normalize_state_fips_function(self):
        """Direct test of normalize_state_fips helper."""
        assert normalize_state_fips(1) == "01"
        assert normalize_state_fips("1") == "01"
        assert normalize_state_fips(6) == "06"
        assert normalize_state_fips("06") == "06"
        assert normalize_state_fips(6.0) == "06"


# ---------------------------------------------------------------------------
# Test: Measure discovery
# ---------------------------------------------------------------------------

class TestMeasureDiscovery:
    """Verify the ingester discovers measure IDs programmatically."""

    def test_discovers_measure_id(self, ingester, tmp_raw_dir):
        """Ingester traverses content areas → indicators → measures."""
        _setup_full_mock(ingester)
        measure_id = ingester.discover_measure()

        assert measure_id == "438"
        assert "Annual Number" in ingester._measure_name

    def test_measure_id_logged(self, ingester, tmp_raw_dir, caplog):
        """The discovered measure ID and name are logged."""
        import logging
        _setup_full_mock(ingester)
        with caplog.at_level(logging.INFO):
            ingester.discover_measure()

        assert any("438" in msg for msg in caplog.messages)

    def test_raises_if_no_heat_content_area(self, ingester):
        """RuntimeError raised if no heat-related content area found."""
        non_heat = [{"id": "11", "name": "Asthma"}, {"id": "22", "name": "Cancer"}]

        def mock_api_get(url, params=None, headers=None):
            return _make_response(non_heat)

        ingester.api_get = MagicMock(side_effect=mock_api_get)
        with pytest.raises(RuntimeError, match="no heat-related content area"):
            ingester._discover_content_area_id()

    def test_raises_if_no_ed_indicator(self, ingester):
        """RuntimeError raised if no ED visit indicator found."""
        non_ed = [{"id": "999", "name": "Extreme Cold Index"}]

        def mock_api_get(url, params=None, headers=None):
            return _make_response(non_ed)

        ingester.api_get = MagicMock(side_effect=mock_api_get)
        with pytest.raises(RuntimeError, match="no ED visit indicator"):
            ingester._discover_indicator_id("35")

    def test_raises_if_no_count_measure(self, ingester):
        """RuntimeError raised if no count measure found (only rates)."""
        only_rates = [
            {"id": "440", "name": "Age-adjusted Rate of ED Visits per 100,000"},
            {"id": "439", "name": "Crude Rate of ED Visits per 100,000"},
        ]

        def mock_api_get(url, params=None, headers=None):
            return _make_response(only_rates)

        ingester.api_get = MagicMock(side_effect=mock_api_get)
        with pytest.raises(RuntimeError, match="no count measure"):
            ingester._discover_measure_id("89")


# ---------------------------------------------------------------------------
# Test: Coverage documentation
# ---------------------------------------------------------------------------

class TestCoverageDocumentation:
    """Verify metadata documents reporting state coverage."""

    def test_reporting_states_in_metadata(self, ingester, tmp_raw_dir):
        """Metadata includes reporting states by year."""
        _setup_full_mock(ingester)
        ingester.fetch(years=[2020, 2021, 2022])

        meta_path = tmp_raw_dir / "cdc_epht" / "cdc_epht_all_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())

        assert "coverage" in meta
        assert "reporting_states_by_year" in meta["coverage"]
        assert "2020" in meta["coverage"]["reporting_states_by_year"]

    def test_coverage_counts(self, ingester, tmp_raw_dir):
        """Metadata documents total reporting states."""
        _setup_full_mock(ingester)
        ingester.fetch(years=[2020, 2021, 2022])

        meta_path = tmp_raw_dir / "cdc_epht" / "cdc_epht_all_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["coverage"]["total_reporting_states"] > 0

    def test_county_coverage_tracked(self, ingester, tmp_raw_dir):
        """Metadata tracks county-level states separately."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        ingester.fetch(years=[2020, 2021])

        meta_path = tmp_raw_dir / "cdc_epht" / "cdc_epht_all_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert "county_level_states_by_year" in meta["coverage"]
        assert meta["coverage"]["total_states_with_county_data"] > 0


# ---------------------------------------------------------------------------
# Test: Output and metadata
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame schema and metadata sidecar."""

    def test_output_columns(self, ingester, tmp_raw_dir):
        """Output has exactly the required columns."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021, 2022])

        expected = {"fips", "state_fips", "year", "heat_ed_visits",
                    "population", "geo_resolution"}
        assert set(df.columns) == expected

    def test_column_dtypes(self, ingester, tmp_raw_dir):
        """Columns have correct dtypes."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020, 2021, 2022])

        assert pd.api.types.is_string_dtype(df["fips"])
        assert pd.api.types.is_string_dtype(df["state_fips"])
        assert pd.api.types.is_integer_dtype(df["year"])
        assert pd.api.types.is_float_dtype(df["heat_ed_visits"])
        assert pd.api.types.is_float_dtype(df["population"])
        assert pd.api.types.is_string_dtype(df["geo_resolution"])

    def test_metadata_sidecar_written(self, ingester, tmp_raw_dir):
        """Metadata JSON is written alongside parquet file."""
        _setup_full_mock(ingester)
        ingester.fetch(years=[2020])

        meta_path = tmp_raw_dir / "cdc_epht" / "cdc_epht_all_metadata.json"
        assert meta_path.exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata has correct source/confidence/attribution values."""
        _setup_full_mock(ingester)
        ingester.fetch(years=[2020])

        meta_path = tmp_raw_dir / "cdc_epht" / "cdc_epht_all_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "CDC_EPHT"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta


# ---------------------------------------------------------------------------
# Test: Ingest purity (no derived metrics)
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed."""

    def test_no_derived_columns(self, ingester, tmp_raw_dir):
        """Output must not contain derived columns."""
        _setup_full_mock(ingester, geo_types=SAMPLE_GEO_TYPES_BOTH)
        df = ingester.fetch(years=[2020, 2021, 2022])

        forbidden = {
            "heat_ed_rate_per_100k", "health_burden_index", "state_avg_rate",
            "county_estimated_visits", "annual_trend", "per_capita_rate",
            "normalized_score", "percentile", "score", "burden_index",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020])

        expected = {"fips", "state_fips", "year", "heat_ed_visits",
                    "population", "geo_resolution"}
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "fips": ["01"], "state_fips": ["01"], "year": [2020],
            "heat_ed_visits": [152.0], "population": [np.nan],
            "geo_resolution": ["state"],
            "heat_ed_rate_per_100k": [3.1],
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Multi-year download & empty response
# ---------------------------------------------------------------------------

class TestMultiYearAndFailure:
    """Verify multi-year fetching and empty response handling."""

    def test_multiple_years_fetched(self, ingester, tmp_raw_dir):
        """Data from multiple years is present in output."""
        _setup_full_mock(ingester)
        df = ingester.fetch(years=[2020, 2021, 2022])

        assert {2020, 2021, 2022}.issubset(set(df["year"].unique()))

    def test_empty_response_returns_empty_df(self, ingester, tmp_raw_dir):
        """Returns empty DataFrame when API returns no data."""
        _setup_full_mock(
            ingester,
            state_table_result=[],
            geo_types=SAMPLE_GEO_TYPES_STATE_ONLY,
        )
        df = ingester.fetch(years=[2020])

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)

    def test_no_matching_years_returns_empty(self, ingester, tmp_raw_dir):
        """Returns empty when requested years don't match available years."""
        _setup_full_mock(ingester, temporal_items=[])
        df = ingester.fetch(years=[1990])

        assert df.empty


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic triggers on HTTP 500/503."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = _make_response(SAMPLE_CONTENT_AREAS)

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
            resp = ingester.api_get("https://example.com/test")

        assert call_count == 2
        assert resp.status_code == 200

    def test_retries_on_503(self, ingester, tmp_raw_dir):
        """Retries after HTTP 503."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        ok_resp = _make_response(SAMPLE_CONTENT_AREAS)

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
            resp = ingester.api_get("https://example.com/test")

        assert call_count == 3
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test: Completeness logging
# ---------------------------------------------------------------------------

class TestCompletenessLogging:
    """Verify log_completeness reports state-level coverage."""

    def test_completeness_reports_states(self, ingester, tmp_raw_dir, caplog):
        """run() logs reporting state count."""
        import logging
        _setup_full_mock(ingester)
        with caplog.at_level(logging.INFO):
            ingester.run(years=[2020, 2021, 2022])

        assert any("reporting states" in msg for msg in caplog.messages)
