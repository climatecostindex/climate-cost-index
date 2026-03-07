"""Tests for the Census block-group ingester (ingest/census_blocks.py).

All HTTP calls are mocked — no real requests to the Census API or CB file downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

from ingest.census_blocks import (
    ACS_VARIABLE,
    CENSUS_ACS_BASE_URL,
    CENSUS_SUPPRESSION_VALUE,
    CensusBlocksIngester,
    STATE_FIPS_CODES,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Census ACS API response format: first row = headers, remaining = data
# State 06 (California), county 037 (Los Angeles), two block groups
SAMPLE_ACS_STATE_06 = [
    [ACS_VARIABLE, "state", "county", "tract", "block group"],
    ["1500", "06", "037", "264202", "1"],
    ["2300", "06", "037", "264202", "2"],
    ["800", "06", "075", "010100", "3"],
]

# State 01 (Alabama) — tests leading zero preservation
SAMPLE_ACS_STATE_01 = [
    [ACS_VARIABLE, "state", "county", "tract", "block group"],
    ["450", "01", "001", "020100", "1"],
    ["620", "01", "003", "010200", "2"],
]

# State 02 (Alaska) — includes suppressed and null values
SAMPLE_ACS_STATE_02 = [
    [ACS_VARIABLE, "state", "county", "tract", "block group"],
    ["100", "02", "020", "000100", "1"],
    ["-666666666", "02", "020", "000100", "2"],  # suppressed
    [None, "02", "020", "000100", "3"],           # null
]

# Synthetic centroid data (simulates output of _fetch_centroids)
# Matches some block groups from the ACS responses above, but not all
SAMPLE_CENTROIDS = pd.DataFrame({
    "block_group_fips": [
        "060372642021", "060372642022", "060750101003",
        "010010201001", "010030102002",
        "020200001001", "020200001002", "020200001003",
        "999990001001",  # CB-only (no ACS match)
    ],
    "lat": [34.0522, 34.0530, 37.7749, 32.5378, 30.7277,
            63.8711, 63.8720, 63.8730, 40.0000],
    "lon": [-118.2437, -118.2440, -122.4194, -86.6441, -87.7929,
            -145.7822, -145.7830, -145.7840, -100.0000],
})

# Minimal centroids (only one entry)
MINIMAL_CENTROIDS = pd.DataFrame({
    "block_group_fips": ["010010201001"],
    "lat": [32.5378],
    "lon": [-86.6441],
})


def _make_response(data: list) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def ingester():
    """Return a fresh CensusBlocksIngester instance."""
    return CensusBlocksIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


def _mock_api_get_for_states(state_responses: dict[str, list]) -> callable:
    """Return a mock api_get that dispatches based on state FIPS in params."""
    def mock_api_get(url, params=None, headers=None):
        in_param = params.get("in", "")
        for state_fips, response_data in state_responses.items():
            if f"state:{state_fips}" in in_param:
                return _make_response(response_data)
        # Return empty for any other state
        return _make_response([[ACS_VARIABLE, "state", "county", "tract", "block group"]])

    return mock_api_get


def _run_fetch_with_mocks(
    ingester: CensusBlocksIngester,
    tmp_raw_dir: Path,
    state_responses: dict[str, list],
    centroids_df: pd.DataFrame = SAMPLE_CENTROIDS,
    year: int = 2022,
) -> pd.DataFrame:
    """Helper: run fetch with mocked ACS API and centroids."""
    mock_api = _mock_api_get_for_states(state_responses)

    with patch.object(ingester, "api_get", side_effect=mock_api):
        with patch.object(ingester, "_fetch_centroids", return_value=centroids_df.copy()):
            with patch("ingest.census_blocks.time.sleep"):
                with patch("ingest.census_blocks.STATE_FIPS_CODES", list(state_responses.keys())):
                    return ingester.fetch(years=[year])


# ---------------------------------------------------------------------------
# Test: FIPS construction
# ---------------------------------------------------------------------------

class TestFIPSConstruction:
    """Verify block-group, county, and state FIPS are correctly constructed."""

    def test_block_group_fips_construction(self, ingester, tmp_raw_dir):
        """State '06' + county '037' + tract '264202' + block group '1' → '060372642021'."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert "060372642021" in df["block_group_fips"].values

    def test_county_fips_extraction(self, ingester, tmp_raw_dir):
        """Block group '060372642021' → county_fips '06037'."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        row = df[df["block_group_fips"] == "060372642021"]
        assert len(row) == 1
        assert row.iloc[0]["county_fips"] == "06037"

    def test_state_fips_extraction(self, ingester, tmp_raw_dir):
        """Block group '060372642021' → state_fips '06'."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        row = df[df["block_group_fips"] == "060372642021"]
        assert len(row) == 1
        assert row.iloc[0]["state_fips"] == "06"

    def test_leading_zero_preservation(self, ingester, tmp_raw_dir):
        """State '01' + county '001' + tract '020100' + block group '1' → '010010201001'."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"01": SAMPLE_ACS_STATE_01},
        )
        assert "010010201001" in df["block_group_fips"].values
        # Verify it's a string, not numeric (which would lose the leading zero)
        fips_val = df[df["block_group_fips"] == "010010201001"]["block_group_fips"].iloc[0]
        assert isinstance(fips_val, str)
        assert fips_val.startswith("01")

    def test_all_fips_12_digit(self, ingester, tmp_raw_dir):
        """Every block_group_fips matches the 12-digit pattern."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01},
        )
        assert df["block_group_fips"].str.match(r"^\d{12}$").all()

    def test_all_county_fips_5_digit(self, ingester, tmp_raw_dir):
        """Every county_fips matches the 5-digit pattern."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01},
        )
        # Filter to only ACS-sourced rows (county_fips derived from API data)
        acs_rows = df[df["block_group_fips"].isin([
            "060372642021", "060372642022", "060750101003",
            "010010201001", "010030102002",
        ])]
        assert acs_rows["county_fips"].str.match(r"^\d{5}$").all()

    def test_all_state_fips_2_digit(self, ingester, tmp_raw_dir):
        """Every state_fips matches the 2-digit pattern."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01},
        )
        acs_rows = df[df["block_group_fips"].isin([
            "060372642021", "060372642022", "060750101003",
            "010010201001", "010030102002",
        ])]
        assert acs_rows["state_fips"].str.match(r"^\d{2}$").all()


# ---------------------------------------------------------------------------
# Test: Housing unit data
# ---------------------------------------------------------------------------

class TestHousingUnitData:
    """Verify housing unit parsing and suppression handling."""

    def test_acs_variable_parsing(self, ingester, tmp_raw_dir):
        """B25001_001E is correctly parsed and stored as housing_units."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        row = df[df["block_group_fips"] == "060372642021"]
        assert len(row) == 1
        assert row.iloc[0]["housing_units"] == 1500.0

    def test_suppressed_value_becomes_nan(self, ingester, tmp_raw_dir):
        """Census suppression value -666666666 is converted to NaN."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"02": SAMPLE_ACS_STATE_02},
        )
        suppressed = df[df["block_group_fips"] == "020200001002"]
        assert len(suppressed) == 1
        assert np.isnan(suppressed.iloc[0]["housing_units"])

    def test_null_value_becomes_nan(self, ingester, tmp_raw_dir):
        """JSON null values are converted to NaN."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"02": SAMPLE_ACS_STATE_02},
        )
        null_row = df[df["block_group_fips"] == "020200001003"]
        assert len(null_row) == 1
        assert np.isnan(null_row.iloc[0]["housing_units"])

    def test_suppressed_rows_preserved(self, ingester, tmp_raw_dir):
        """Rows with suppressed/null housing units are retained, not dropped."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"02": SAMPLE_ACS_STATE_02},
        )
        acs_bgs = ["020200001001", "020200001002", "020200001003"]
        for bg in acs_bgs:
            assert bg in df["block_group_fips"].values

    def test_suppressed_not_stored_as_negative(self, ingester, tmp_raw_dir):
        """Suppressed values are NaN, not the raw negative sentinel."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"02": SAMPLE_ACS_STATE_02},
        )
        # No housing_units value should be -666666666
        valid = df["housing_units"].dropna()
        assert (valid != CENSUS_SUPPRESSION_VALUE).all()


# ---------------------------------------------------------------------------
# Test: Centroid data
# ---------------------------------------------------------------------------

class TestCentroids:
    """Verify centroid coordinate parsing and joining."""

    def test_coordinate_parsing(self, ingester, tmp_raw_dir):
        """Latitude and longitude are correctly parsed from CB file."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        row = df[df["block_group_fips"] == "060372642021"]
        assert len(row) == 1
        assert abs(row.iloc[0]["lat"] - 34.0522) < 0.001
        assert abs(row.iloc[0]["lon"] - (-118.2437)) < 0.001

    def test_coordinate_types(self, ingester, tmp_raw_dir):
        """lat and lon are float dtype."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert pd.api.types.is_float_dtype(df["lat"])
        assert pd.api.types.is_float_dtype(df["lon"])

    def test_coordinate_ranges(self, ingester, tmp_raw_dir):
        """Coordinates fall within U.S. bounds (lat: 17–72, lon: -180 to -65)."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01, "02": SAMPLE_ACS_STATE_02},
        )
        valid_lat = df["lat"].dropna()
        valid_lon = df["lon"].dropna()
        assert (valid_lat >= 17).all() and (valid_lat <= 72).all()
        assert (valid_lon >= -180).all() and (valid_lon <= -65).all()

    def test_centroid_join(self, ingester, tmp_raw_dir):
        """ACS housing data is correctly joined with CB centroid coordinates."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        # Block group present in both ACS and CB should have all fields
        row = df[df["block_group_fips"] == "060372642021"]
        assert len(row) == 1
        assert row.iloc[0]["housing_units"] == 1500.0
        assert not np.isnan(row.iloc[0]["lat"])
        assert not np.isnan(row.iloc[0]["lon"])


# ---------------------------------------------------------------------------
# Test: Join behavior and missing data
# ---------------------------------------------------------------------------

class TestJoinBehavior:
    """Verify outer-join handling of ACS-only and CB-only block groups."""

    def test_missing_cb_entry(self, ingester, tmp_raw_dir):
        """Block group in ACS but not CB file gets NaN for lat/lon."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
            centroids_df=MINIMAL_CENTROIDS,
        )
        row = df[df["block_group_fips"] == "060372642021"]
        assert len(row) == 1
        assert np.isnan(row.iloc[0]["lat"])
        assert np.isnan(row.iloc[0]["lon"])
        # Housing units should still be present
        assert row.iloc[0]["housing_units"] == 1500.0

    def test_missing_acs_entry(self, ingester, tmp_raw_dir):
        """Block group in CB file but not ACS gets NaN for housing_units."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        # "999990001001" is in CB centroids but not in any ACS response
        row = df[df["block_group_fips"] == "999990001001"]
        assert len(row) == 1
        assert np.isnan(row.iloc[0]["housing_units"])
        # Coordinates should still be present
        assert abs(row.iloc[0]["lat"] - 40.0) < 0.001
        assert abs(row.iloc[0]["lon"] - (-100.0)) < 0.001

    def test_join_completeness(self, ingester, tmp_raw_dir):
        """Most block groups from ACS match with CB centroid entries."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01},
        )
        # All 5 ACS block groups from states 06 and 01 should be present
        acs_bgs = [
            "060372642021", "060372642022", "060750101003",
            "010010201001", "010030102002",
        ]
        for bg in acs_bgs:
            assert bg in df["block_group_fips"].values

        # Count how many ACS block groups have coordinates
        matched = df[df["block_group_fips"].isin(acs_bgs)]
        has_coords = matched["lat"].notna().sum()
        assert has_coords == len(acs_bgs)  # All should match in this mock data


# ---------------------------------------------------------------------------
# Test: State-by-state processing
# ---------------------------------------------------------------------------

class TestStateProcessing:
    """Verify state-by-state ACS querying and partial failure handling."""

    def test_multi_state_query(self, ingester, tmp_raw_dir):
        """Ingester queries ACS separately per state; all results combined."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01, "02": SAMPLE_ACS_STATE_02},
        )
        # Should have block groups from all three states
        states_in_output = df[df["state_fips"].isin(["01", "02", "06"])]["state_fips"].unique()
        assert set(states_in_output) == {"01", "02", "06"}

    def test_partial_state_failure(self, ingester, tmp_raw_dir):
        """If one state's query fails, other states' data is still cached."""
        def mock_api_get(url, params=None, headers=None):
            in_param = params.get("in", "")
            if "state:06" in in_param:
                raise httpx.TransportError("Connection refused")
            if "state:01" in in_param:
                return _make_response(SAMPLE_ACS_STATE_01)
            return _make_response([[ACS_VARIABLE, "state", "county", "tract", "block group"]])

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            with patch.object(ingester, "_fetch_centroids", return_value=SAMPLE_CENTROIDS.copy()):
                with patch("ingest.census_blocks.time.sleep"):
                    with patch("ingest.census_blocks.STATE_FIPS_CODES", ["06", "01"]):
                        df = ingester.fetch(years=[2022])

        # State 01 data should be present despite state 06 failure
        assert not df.empty
        assert "010010201001" in df["block_group_fips"].values

    def test_api_key_included_when_set(self, ingester, tmp_raw_dir, monkeypatch):
        """Request params include 'key' when CENSUS_API_KEY is set."""
        monkeypatch.setenv("CENSUS_API_KEY", "test_key_abc123")

        captured_params: list[dict] = []

        def mock_api_get(url, params=None, headers=None):
            captured_params.append(dict(params or {}))
            return _make_response(SAMPLE_ACS_STATE_06)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            with patch.object(ingester, "_fetch_centroids", return_value=SAMPLE_CENTROIDS.copy()):
                with patch("ingest.census_blocks.time.sleep"):
                    with patch("ingest.census_blocks.STATE_FIPS_CODES", ["06"]):
                        ingester.fetch(years=[2022])

        assert len(captured_params) > 0
        assert "key" in captured_params[0]
        assert captured_params[0]["key"] == "test_key_abc123"

    def test_api_key_omitted_when_unset(self, ingester, tmp_raw_dir, monkeypatch):
        """Request params omit 'key' when CENSUS_API_KEY is empty."""
        monkeypatch.setenv("CENSUS_API_KEY", "")

        captured_params: list[dict] = []

        def mock_api_get(url, params=None, headers=None):
            captured_params.append(dict(params or {}))
            return _make_response(SAMPLE_ACS_STATE_06)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            with patch.object(ingester, "_fetch_centroids", return_value=SAMPLE_CENTROIDS.copy()):
                with patch("ingest.census_blocks.time.sleep"):
                    with patch("ingest.census_blocks.STATE_FIPS_CODES", ["06"]):
                        ingester.fetch(years=[2022])

        assert len(captured_params) > 0
        assert "key" not in captured_params[0]


# ---------------------------------------------------------------------------
# Test: Output schema and metadata
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has all expected columns with correct dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required columns."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        expected = {"block_group_fips", "county_fips", "state_fips",
                    "housing_units", "lat", "lon"}
        assert set(df.columns) == expected

    def test_block_group_fips_is_string(self, ingester, tmp_raw_dir):
        """block_group_fips is string dtype."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert pd.api.types.is_string_dtype(df["block_group_fips"])

    def test_county_fips_is_string(self, ingester, tmp_raw_dir):
        """county_fips is string dtype."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert pd.api.types.is_string_dtype(df["county_fips"])

    def test_state_fips_is_string(self, ingester, tmp_raw_dir):
        """state_fips is string dtype."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert pd.api.types.is_string_dtype(df["state_fips"])

    def test_housing_units_is_float(self, ingester, tmp_raw_dir):
        """housing_units is float dtype."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert pd.api.types.is_float_dtype(df["housing_units"])

    def test_lat_is_float(self, ingester, tmp_raw_dir):
        """lat is float dtype."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert pd.api.types.is_float_dtype(df["lat"])

    def test_lon_is_float(self, ingester, tmp_raw_dir):
        """lon is float dtype."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert pd.api.types.is_float_dtype(df["lon"])


class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir):
        """A _metadata.json file is written next to the parquet file."""
        _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert (tmp_raw_dir / "census_blocks" / "census_blocks_2022_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source/confidence/attribution values."""
        _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        meta_path = tmp_raw_dir / "census_blocks" / "census_blocks_2022_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "CENSUS_BLOCKS"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "none"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_parquet_file_created(self, ingester, tmp_raw_dir):
        """A parquet file is created in the cache directory."""
        _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        assert (tmp_raw_dir / "census_blocks" / "census_blocks_2022.parquet").exists()


# ---------------------------------------------------------------------------
# Test: Row count and multi-state coverage
# ---------------------------------------------------------------------------

class TestRowCount:
    """Verify output contains records from all queried states."""

    def test_records_from_all_states(self, ingester, tmp_raw_dir):
        """Output contains block groups from every queried state."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01},
        )
        acs_states = df[df["state_fips"].isin(["01", "06"])]["state_fips"].unique()
        assert "01" in acs_states
        assert "06" in acs_states


# ---------------------------------------------------------------------------
# Test: Ingest purity (no derived metrics)
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed."""

    def test_no_flood_zone_columns(self, ingester, tmp_raw_dir):
        """Output must not contain flood zone overlay columns."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        forbidden = {
            "in_flood_zone", "pct_hu_high_risk", "flood_zone_overlay",
            "county_housing_total", "county_aggregation", "flood_zone",
            "pct_area_high_risk", "pct_housing_units_high_risk",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        expected = {"block_group_fips", "county_fips", "state_fips",
                    "housing_units", "lat", "lon"}
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "block_group_fips": ["060372642021"],
            "county_fips": ["06037"],
            "state_fips": ["06"],
            "housing_units": [1500.0],
            "lat": [34.0522],
            "lon": [-118.2437],
            "in_flood_zone": [True],  # forbidden extra column
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)

    def test_no_county_aggregation(self, ingester, tmp_raw_dir):
        """Output is at block-group grain (12-digit FIPS), NOT county level."""
        df = _run_fetch_with_mocks(
            ingester, tmp_raw_dir,
            {"06": SAMPLE_ACS_STATE_06},
        )
        # All block_group_fips should be 12 digits
        assert df["block_group_fips"].str.match(r"^\d{12}$").all()
        # Multiple block groups per county (county 06037 has 2 block groups)
        county_037 = df[df["county_fips"] == "06037"]
        assert len(county_037) >= 2


# ---------------------------------------------------------------------------
# Test: Completeness logging
# ---------------------------------------------------------------------------

class TestCompletenessLogging:
    """Verify log_completeness reports block groups and states."""

    def test_completeness_logged(self, ingester, tmp_raw_dir, caplog):
        """run() logs block group count and state coverage."""
        import logging
        with caplog.at_level(logging.INFO):
            df = _run_fetch_with_mocks(
                ingester, tmp_raw_dir,
                {"06": SAMPLE_ACS_STATE_06, "01": SAMPLE_ACS_STATE_01},
            )
            ingester.log_completeness(df)

        assert any("block groups" in msg.lower() for msg in caplog.messages)
        assert any("states" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic triggers on HTTP 500/503."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = _make_response(SAMPLE_ACS_STATE_06)

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
            df = ingester._fetch_state_block_groups(2022, "06")

        assert not df.empty
        assert call_count == 2

    def test_retries_on_503(self, ingester, tmp_raw_dir):
        """Retries after HTTP 503."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        ok_resp = _make_response(SAMPLE_ACS_STATE_06)

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
            df = ingester._fetch_state_block_groups(2022, "06")

        assert not df.empty
        assert call_count == 3
