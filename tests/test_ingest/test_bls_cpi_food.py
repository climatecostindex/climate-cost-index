"""Tests for the BLS CPI Food at Home ingester (ingest/bls_cpi_food.py).

All HTTP calls are mocked — no real requests to BLS.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

from ingest.bls_cpi_food import (
    BLS_API_URL,
    BLS_AREA_CODE_URL,
    BLSCPIFoodIngester,
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    NATIONAL_AREA_CODE,
    REGION_AREA_CODES,
    SERIES_PREFIX,
    SERIES_SUFFIX,
    V1_MAX_SERIES_PER_BATCH,
    V1_MAX_YEARS_PER_QUERY,
    V2_MAX_SERIES_PER_BATCH,
    V2_MAX_YEARS_PER_QUERY,
    _build_series_id,
    _classify_geo_type,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Synthetic BLS area code file content (tab-separated, current BLS format)
SYNTHETIC_AREA_FILE = (
    "area_code\tarea_name\tdisplay_level\tselectable\tsort_sequence\n"
    "0000\tU.S. city average\t0\tT\t1\n"
    "0100\tNortheast\t0\tT\t2\n"
    "0200\tMidwest\t0\tT\t3\n"
    "0300\tSouth\t0\tT\t4\n"
    "0400\tWest\t0\tT\t5\n"
    "0110\tNew England\t1\tT\t10\n"
    "S000\tSize Class A\t0\tT\t6\n"
    "S100\tNortheast - Size Class A\t1\tT\t7\n"
    "N000\tSize Class B/C\t0\tT\t8\n"
    "S12A\tMiami-Fort Lauderdale-West Palm Beach, FL\t1\tT\t10\n"
    "S23A\tMilwaukee-Racine, WI\t1\tT\t11\n"
    "S35B\tNew York-Newark-Jersey City, NY-NJ-PA\t1\tT\t12\n"
    "A104\tPittsburgh, PA\t1\tT\t9\n"
)

# Synthetic BLS area code file with old format (has area_type column)
SYNTHETIC_AREA_FILE_OLD_FORMAT = (
    "area_type\tarea_code\tarea_text\tdisplay_level\tselectable\tsort_sequence\n"
    "A\t0000\tU.S. city average\t0\tT\t1\n"
    "A\t0100\tNortheast\t0\tT\t2\n"
    "A\t0200\tMidwest\t0\tT\t3\n"
    "A\t0300\tSouth\t0\tT\t4\n"
    "A\t0400\tWest\t0\tT\t5\n"
    "A\tS12A\tMiami-Fort Lauderdale-West Palm Beach, FL\t1\tT\t10\n"
    "A\tS23A\tMilwaukee-Racine, WI\t1\tT\t11\n"
    "A\tS35B\tNew York-Newark-Jersey City, NY-NJ-PA\t1\tT\t12\n"
    "B\tXXXX\tNon-CPI area\t2\tF\t99\n"
)


def _make_bls_api_response(
    series_data: list[dict],
    status: str = "REQUEST_SUCCEEDED",
) -> dict:
    """Build a synthetic BLS API JSON response.

    Args:
        series_data: List of series dicts with "seriesID" and "data" keys.
        status: BLS response status string.

    Returns:
        Dict matching BLS API response structure.
    """
    return {
        "status": status,
        "responseTime": 100,
        "message": [],
        "Results": {
            "series": series_data,
        },
    }


def _make_series(
    area_code: str,
    year_data: dict[int, float],
    include_m13: bool = True,
) -> dict:
    """Build a single BLS series dict with synthetic observations.

    Args:
        area_code: BLS area code.
        year_data: Mapping of year → annual CPI index value.
        include_m13: If True, include an M13 (annual average) observation.
                     If False, include only 12 monthly values.

    Returns:
        Series dict matching BLS API structure.
    """
    series_id = _build_series_id(area_code)
    data = []

    for year, value in year_data.items():
        if include_m13:
            data.append({
                "year": str(year),
                "period": "M13",
                "periodName": "Annual",
                "value": str(value),
                "footnotes": [{}],
            })
        # Always include monthly values too
        for month in range(1, 13):
            # Vary monthly values slightly around the annual average
            monthly_val = value + (month - 6.5) * 0.1
            data.append({
                "year": str(year),
                "period": f"M{month:02d}",
                "periodName": f"Month {month}",
                "value": str(round(monthly_val, 1)),
                "footnotes": [{}],
            })

    return {"seriesID": series_id, "data": data}


def _make_area_response() -> MagicMock:
    """Create a mock httpx.Response for the area code file."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.text = SYNTHETIC_AREA_FILE
    resp.raise_for_status = MagicMock()
    return resp


def _default_series_data() -> list[dict]:
    """Build default series data for national + 4 regions + 4 metros."""
    return [
        _make_series("0000", {2022: 300.0, 2023: 312.0, 2024: 318.0}),
        _make_series("0100", {2022: 305.0, 2023: 316.0, 2024: 322.0}),
        _make_series("0200", {2022: 290.0, 2023: 301.0, 2024: 307.0}),
        _make_series("0300", {2022: 295.0, 2023: 308.0, 2024: 314.0}),
        _make_series("0400", {2022: 310.0, 2023: 322.0, 2024: 328.0}),
        _make_series("S12A", {2022: 302.0, 2023: 315.0, 2024: 321.0}),
        _make_series("S23A", {2022: 288.0, 2023: 299.0, 2024: 305.0}),
        _make_series("S35B", {2022: 320.0, 2023: 334.0, 2024: 340.0}),
        _make_series("A104", {2022: 298.0, 2023: 310.0, 2024: 316.0}),
    ]


@pytest.fixture
def ingester():
    """Return a fresh BLSCPIFoodIngester instance."""
    return BLSCPIFoodIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def mock_env_no_key(monkeypatch):
    """Ensure no BLS_API_KEY is set."""
    monkeypatch.delenv("BLS_API_KEY", raising=False)


@pytest.fixture
def mock_env_with_key(monkeypatch):
    """Set a test BLS_API_KEY."""
    monkeypatch.setenv("BLS_API_KEY", "test-key-12345")


def _patch_ingester_for_fetch(
    ingester: BLSCPIFoodIngester,
    series_data: list[dict] | None = None,
    area_text: str = SYNTHETIC_AREA_FILE,
    api_status: str = "REQUEST_SUCCEEDED",
):
    """Return context manager patches for area codes and API POST.

    Args:
        ingester: The ingester instance.
        series_data: Series data for the API response. Defaults to _default_series_data().
        area_text: Raw text for area code file.
        api_status: BLS API response status.

    Returns:
        Tuple of (area_patch, api_patch) context managers.
    """
    if series_data is None:
        series_data = _default_series_data()

    area_resp = MagicMock(spec=httpx.Response)
    area_resp.status_code = 200
    area_resp.text = area_text
    area_resp.raise_for_status = MagicMock()

    api_response = _make_bls_api_response(series_data, status=api_status)

    def mock_api_get(url, params=None, headers=None):
        return area_resp

    def mock_api_post(url, json_body):
        return api_response

    return (
        patch.object(ingester, "api_get", side_effect=mock_api_get),
        patch.object(ingester, "_api_post", side_effect=mock_api_post),
    )


# ---------------------------------------------------------------------------
# Test: Series ID construction and area mapping
# ---------------------------------------------------------------------------

class TestSeriesIDConstruction:
    """Verify series IDs follow BLS naming convention."""

    def test_series_id_format(self):
        """Series IDs follow CUUR{area_code}SAF11 format."""
        sid = _build_series_id("0000")
        assert sid == "CUUR0000SAF11"

    def test_regional_series_ids(self):
        """All 4 Census regions produce valid series IDs."""
        for code in ["0100", "0200", "0300", "0400"]:
            sid = _build_series_id(code)
            assert sid == f"CUUR{code}SAF11"

    def test_national_series_included(self, ingester, tmp_raw_dir, mock_env_no_key):
        """National series CUUR0000SAF11 is included in queries."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch as mock_post:
            df = ingester.fetch(years=[2023])

        # Check that national area_code appears in results
        assert "0000" in df["area_code"].values

    def test_regional_series_coverage(self, ingester, tmp_raw_dir, mock_env_no_key):
        """All 4 Census regions are queried and appear in output."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        for code in ["0100", "0200", "0300", "0400"]:
            assert code in df["area_code"].values, f"Region {code} missing"

    def test_metro_series_discovery(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Metro area series are discovered and queried."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        metro_codes = set(df[df["geo_type"] == "metro"]["area_code"].unique())
        assert len(metro_codes) >= 4
        assert "S12A" in metro_codes
        assert "A104" in metro_codes

    def test_area_name_mapping(self, ingester, tmp_raw_dir, mock_env_no_key):
        """BLS area codes are mapped to human-readable names."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        # Check a metro area name
        miami = df[df["area_code"] == "S12A"]
        assert len(miami) > 0
        assert "Miami" in miami.iloc[0]["area_name"]

    def test_metro_series_id_format(self):
        """Metro area series IDs follow the CUUR{code}SAF11 pattern."""
        sid = _build_series_id("S12A")
        assert sid == "CUURS12ASAF11"


# ---------------------------------------------------------------------------
# Test: Geographic type tagging
# ---------------------------------------------------------------------------

class TestGeoTypeTagging:
    """Verify geo_type classification."""

    def test_geo_type_national(self):
        """National area code classifies as 'national'."""
        assert _classify_geo_type("0000") == "national"

    def test_geo_type_region(self):
        """Region area codes classify as 'region'."""
        for code in ["0100", "0200", "0300", "0400"]:
            assert _classify_geo_type(code) == "region"

    def test_geo_type_metro(self):
        """Metro area codes classify as 'metro'."""
        assert _classify_geo_type("S12A") == "metro"
        assert _classify_geo_type("S35B") == "metro"

    def test_geo_type_in_output(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Output records have correct geo_type values."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        # National
        nat = df[df["area_code"] == "0000"]
        assert (nat["geo_type"] == "national").all()

        # Region
        reg = df[df["area_code"] == "0100"]
        assert (reg["geo_type"] == "region").all()

        # Metro
        met = df[df["area_code"] == "S12A"]
        assert (met["geo_type"] == "metro").all()


# ---------------------------------------------------------------------------
# Test: Annual average computation
# ---------------------------------------------------------------------------

class TestAnnualAverageComputation:
    """Verify M13 preference and computed average fallback."""

    def test_published_annual_average_m13(self, ingester, tmp_raw_dir, mock_env_no_key):
        """When M13 is present, it's used and period_type='M13'."""
        series = [_make_series("0000", {2023: 312.0}, include_m13=True)]
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        nat = df[df["area_code"] == "0000"]
        assert len(nat) == 1
        assert nat.iloc[0]["food_cpi_index"] == 312.0
        assert nat.iloc[0]["period_type"] == "M13"

    def test_computed_annual_average(self, ingester, tmp_raw_dir, mock_env_no_key):
        """When M13 is absent, average is computed from monthly and period_type='computed'."""
        series = [_make_series("0000", {2023: 312.0}, include_m13=False)]
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        nat = df[df["area_code"] == "0000"]
        assert len(nat) == 1
        assert nat.iloc[0]["period_type"] == "computed"
        # Computed average should be close to 312.0 (monthly values vary slightly)
        assert abs(nat.iloc[0]["food_cpi_index"] - 312.0) < 1.0

    def test_partial_year_handling(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Partial year with <12 months uses available months for average."""
        # Build a series with only 6 months of data
        series_id = _build_series_id("0000")
        data = []
        for month in range(1, 7):  # Jan-Jun only
            data.append({
                "year": "2025",
                "period": f"M{month:02d}",
                "periodName": f"Month {month}",
                "value": str(300.0 + month),
                "footnotes": [{}],
            })

        series = [{"seriesID": series_id, "data": data}]
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2025])

        nat = df[df["area_code"] == "0000"]
        assert len(nat) == 1
        assert nat.iloc[0]["period_type"] == "computed"
        # Average of 301, 302, 303, 304, 305, 306 = 303.5
        assert abs(nat.iloc[0]["food_cpi_index"] - 303.5) < 0.01


# ---------------------------------------------------------------------------
# Test: Year-over-year change
# ---------------------------------------------------------------------------

class TestYoYChange:
    """Verify year-over-year percentage change computation."""

    def test_yoy_computation(self, ingester, tmp_raw_dir, mock_env_no_key):
        """YoY change = (current - prior) / prior × 100."""
        # 250 → 260 = 4.0% change
        series = [_make_series("0000", {2022: 250.0, 2023: 260.0})]
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2022, 2023])

        row_2023 = df[(df["area_code"] == "0000") & (df["year"] == 2023)]
        assert len(row_2023) == 1
        assert abs(row_2023.iloc[0]["food_cpi_yoy_change"] - 4.0) < 0.01

    def test_first_year_nan(self, ingester, tmp_raw_dir, mock_env_no_key):
        """First year in series has NaN for YoY change."""
        series = [_make_series("0000", {2022: 250.0, 2023: 260.0})]
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2022, 2023])

        row_2022 = df[(df["area_code"] == "0000") & (df["year"] == 2022)]
        assert len(row_2022) == 1
        assert pd.isna(row_2022.iloc[0]["food_cpi_yoy_change"])


# ---------------------------------------------------------------------------
# Test: API key handling
# ---------------------------------------------------------------------------

class TestAPIKeyHandling:
    """Verify v1/v2 API behavior based on key presence."""

    def test_v2_api_with_key(self, ingester, mock_env_with_key):
        """Registration key is included in request when BLS_API_KEY is set."""
        assert ingester._get_api_key() == "test-key-12345"

    def test_v1_api_without_key(self, ingester, mock_env_no_key):
        """No registration key when BLS_API_KEY is not set."""
        assert ingester._get_api_key() is None

    def test_batch_size_v2(self, ingester, mock_env_with_key):
        """With API key, batch size is 50 series, 20 years."""
        max_series, max_years = ingester._api_limits()
        assert max_series == V2_MAX_SERIES_PER_BATCH
        assert max_years == V2_MAX_YEARS_PER_QUERY

    def test_batch_size_v1(self, ingester, mock_env_no_key):
        """Without API key, batch size is 25 series, 10 years."""
        max_series, max_years = ingester._api_limits()
        assert max_series == V1_MAX_SERIES_PER_BATCH
        assert max_years == V1_MAX_YEARS_PER_QUERY

    def test_registration_key_in_request(self, ingester, tmp_raw_dir, mock_env_with_key):
        """v2 API request body includes registrationkey."""
        series_data = _default_series_data()

        # Track what body was sent to _api_post
        captured_bodies = []
        original_api_post = ingester._api_post

        area_resp = MagicMock(spec=httpx.Response)
        area_resp.status_code = 200
        area_resp.text = SYNTHETIC_AREA_FILE
        area_resp.raise_for_status = MagicMock()

        def mock_query(series_ids, start_year, end_year):
            body = {
                "seriesid": series_ids,
                "startyear": str(start_year),
                "endyear": str(end_year),
                "registrationkey": ingester._get_api_key(),
            }
            captured_bodies.append(body)
            return _make_bls_api_response(series_data)

        with (
            patch.object(ingester, "api_get", return_value=area_resp),
            patch.object(ingester, "_query_series_batch", side_effect=mock_query),
        ):
            ingester.fetch(years=[2023])

        assert len(captured_bodies) > 0
        assert captured_bodies[0]["registrationkey"] == "test-key-12345"

    def test_no_registration_key_without_env(self, ingester, tmp_raw_dir, mock_env_no_key):
        """v1 API request body omits registrationkey."""
        series_data = _default_series_data()

        captured_bodies = []

        area_resp = MagicMock(spec=httpx.Response)
        area_resp.status_code = 200
        area_resp.text = SYNTHETIC_AREA_FILE
        area_resp.raise_for_status = MagicMock()

        def mock_query(series_ids, start_year, end_year):
            body = {
                "seriesid": series_ids,
                "startyear": str(start_year),
                "endyear": str(end_year),
            }
            key = ingester._get_api_key()
            if key:
                body["registrationkey"] = key
            captured_bodies.append(body)
            return _make_bls_api_response(series_data)

        with (
            patch.object(ingester, "api_get", return_value=area_resp),
            patch.object(ingester, "_query_series_batch", side_effect=mock_query),
        ):
            ingester.fetch(years=[2023])

        assert len(captured_bodies) > 0
        assert "registrationkey" not in captured_bodies[0]


# ---------------------------------------------------------------------------
# Test: Output and metadata
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has all expected columns with correct dtypes."""

    def test_output_columns(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Output contains exactly the required columns."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        expected = {"area_code", "area_name", "geo_type", "year",
                    "food_cpi_index", "food_cpi_yoy_change", "period_type"}
        assert set(df.columns) == expected

    def test_area_code_is_string(self, ingester, tmp_raw_dir, mock_env_no_key):
        """area_code column is string dtype."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        assert pd.api.types.is_string_dtype(df["area_code"])

    def test_area_name_is_string(self, ingester, tmp_raw_dir, mock_env_no_key):
        """area_name column is string dtype."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        assert pd.api.types.is_string_dtype(df["area_name"])

    def test_geo_type_is_string(self, ingester, tmp_raw_dir, mock_env_no_key):
        """geo_type column is string dtype."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        assert pd.api.types.is_string_dtype(df["geo_type"])

    def test_year_is_int(self, ingester, tmp_raw_dir, mock_env_no_key):
        """year column is integer dtype."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        assert pd.api.types.is_integer_dtype(df["year"])

    def test_food_cpi_index_is_float(self, ingester, tmp_raw_dir, mock_env_no_key):
        """food_cpi_index column is float dtype."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        assert pd.api.types.is_float_dtype(df["food_cpi_index"])

    def test_food_cpi_yoy_change_is_float(self, ingester, tmp_raw_dir, mock_env_no_key):
        """food_cpi_yoy_change column is float dtype."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        assert pd.api.types.is_float_dtype(df["food_cpi_yoy_change"])

    def test_period_type_is_string(self, ingester, tmp_raw_dir, mock_env_no_key):
        """period_type column is string dtype."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        assert pd.api.types.is_string_dtype(df["period_type"])


class TestMetadataSidecar:
    """Verify metadata JSON is written with correct values."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Metadata JSON file is written alongside parquet."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            ingester.fetch(years=[2023])

        meta_path = tmp_raw_dir / "bls_cpi_food" / "bls_cpi_food_all_metadata.json"
        assert meta_path.exists()

    def test_metadata_content(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Metadata has correct source/confidence/attribution."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            ingester.fetch(years=[2023])

        meta_path = tmp_raw_dir / "bls_cpi_food" / "bls_cpi_food_all_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "BLS_CPI_FOOD"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "none"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta


# ---------------------------------------------------------------------------
# Test: Ingest purity check
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO county-level or CCI-derived metrics in output."""

    def test_no_county_columns(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Output must not contain county-level columns."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        forbidden = {
            "county_fips", "fips", "county_food_cost", "climate_attribution",
            "rolling_avg", "trend_slope", "inflation_adjusted", "score",
            "percentile", "county_name",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Output columns are EXACTLY the required set — no extras."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        expected = {"area_code", "area_name", "geo_type", "year",
                    "food_cpi_index", "food_cpi_yoy_change", "period_type"}
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "area_code": ["0000"],
            "area_name": ["U.S. city average"],
            "geo_type": ["national"],
            "year": [2023],
            "food_cpi_index": [312.0],
            "food_cpi_yoy_change": [4.0],
            "period_type": ["M13"],
            "county_fips": ["01001"],  # FORBIDDEN
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Area code preservation
# ---------------------------------------------------------------------------

class TestAreaCodePreservation:
    """Verify BLS area codes are preserved exactly."""

    def test_area_codes_unchanged(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Area codes in output match exactly what BLS provides."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2023])

        codes = set(df["area_code"].unique())
        expected_codes = {"0000", "0100", "0200", "0300", "0400", "S12A", "S23A", "S35B", "A104"}
        assert codes == expected_codes


# ---------------------------------------------------------------------------
# Test: Multi-year download
# ---------------------------------------------------------------------------

class TestMultiYearDownload:
    """Verify the ingester fetches multiple years."""

    def test_multi_year_data(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Multiple years of data are returned."""
        series_data = _default_series_data()
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2022, 2023, 2024])

        years = set(df["year"].unique())
        assert years == {2022, 2023, 2024}


# ---------------------------------------------------------------------------
# Test: Partial failure
# ---------------------------------------------------------------------------

class TestPartialFailure:
    """Verify partial batch failures don't abort the entire run."""

    def test_partial_batch_failure(self, ingester, tmp_raw_dir, mock_env_no_key):
        """If one batch fails, data from other batches is still cached."""
        series_data = _default_series_data()

        area_resp = MagicMock(spec=httpx.Response)
        area_resp.status_code = 200
        area_resp.text = SYNTHETIC_AREA_FILE
        area_resp.raise_for_status = MagicMock()

        call_count = 0
        api_response = _make_bls_api_response(series_data)

        def mock_api_post(url, json_body):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "500 Internal Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            return api_response

        with (
            patch.object(ingester, "api_get", return_value=area_resp),
            patch.object(ingester, "_api_post", side_effect=mock_api_post),
        ):
            # Force v1 limits to create multiple batches
            with patch.object(ingester, "_api_limits", return_value=(3, 20)):
                df = ingester.fetch(years=[2023])

        # Should still have some data despite first batch failure
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic triggers on HTTP 500/503."""

    def test_retries_on_500(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        series_data = _default_series_data()
        api_response = _make_bls_api_response(series_data)

        area_resp = MagicMock(spec=httpx.Response)
        area_resp.status_code = 200
        area_resp.text = SYNTHETIC_AREA_FILE
        area_resp.raise_for_status = MagicMock()

        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.json.return_value = api_response
        ok_resp.raise_for_status = MagicMock()

        call_count = 0

        def mock_post(url, json=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fail_resp
            return ok_resp

        with (
            patch.object(ingester, "api_get", return_value=area_resp),
            patch.object(ingester, "_client", MagicMock()),
        ):
            ingester._client.post = mock_post
            ingester._last_call_time = 0.0
            result = ingester._api_post(BLS_API_URL, {"seriesid": ["test"]})

        assert call_count == 2

    def test_retries_on_503(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Retries after HTTP 503."""
        api_response = _make_bls_api_response(_default_series_data())

        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.json.return_value = api_response
        ok_resp.raise_for_status = MagicMock()

        call_count = 0

        def mock_post(url, json=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return fail_resp
            return ok_resp

        with patch.object(ingester, "_client", MagicMock()):
            ingester._client.post = mock_post
            ingester._last_call_time = 0.0
            result = ingester._api_post(BLS_API_URL, {"seriesid": ["test"]})

        assert call_count == 3


# ---------------------------------------------------------------------------
# Test: Missing metro coverage
# ---------------------------------------------------------------------------

class TestMissingMetroCoverage:
    """Verify missing metro-year combinations are not imputed."""

    def test_no_imputation_for_missing_years(self, ingester, tmp_raw_dir, mock_env_no_key):
        """Metro area with data for only a subset of years has no imputed rows."""
        # S12A has data for 2022-2024, S23A has data for 2023 only
        series_data = [
            _make_series("0000", {2022: 300.0, 2023: 312.0, 2024: 318.0}),
            _make_series("S12A", {2022: 302.0, 2023: 315.0, 2024: 321.0}),
            _make_series("S23A", {2023: 299.0}),
        ]
        area_patch, api_patch = _patch_ingester_for_fetch(ingester, series_data)

        with area_patch, api_patch:
            df = ingester.fetch(years=[2022, 2023, 2024])

        # S23A should only have 2023
        s23a = df[df["area_code"] == "S23A"]
        assert set(s23a["year"]) == {2023}

        # S12A should have all 3 years
        s12a = df[df["area_code"] == "S12A"]
        assert set(s12a["year"]) == {2022, 2023, 2024}


# ---------------------------------------------------------------------------
# Test: Fallback area codes
# ---------------------------------------------------------------------------

class TestFallbackAreaCodes:
    """Verify fallback when area code reference is unavailable."""

    def test_fallback_to_regions(self, ingester, tmp_raw_dir, mock_env_no_key):
        """When area code file download fails, falls back to regions only."""
        area_map = ingester._fallback_area_codes()

        assert NATIONAL_AREA_CODE in area_map
        for code in REGION_AREA_CODES:
            assert code in area_map
        # No metro areas
        assert len(area_map) == 5


# ---------------------------------------------------------------------------
# Test: Area code parsing
# ---------------------------------------------------------------------------

class TestAreaCodeParsing:
    """Verify area code reference file parsing."""

    def test_parse_area_codes(self, ingester):
        """Parses BLS cu.area file correctly (current format)."""
        areas = ingester._parse_area_codes(SYNTHETIC_AREA_FILE)

        # National + 4 regions + 4 metros = 9
        assert len(areas) == 9
        assert areas["0000"] == "U.S. city average"
        assert areas["0100"] == "Northeast"
        assert "Miami" in areas["S12A"]
        assert "Pittsburgh" in areas["A104"]

    def test_parse_old_format(self, ingester):
        """Parses BLS cu.area file with area_type column (old format)."""
        areas = ingester._parse_area_codes(SYNTHETIC_AREA_FILE_OLD_FORMAT)

        assert areas["0000"] == "U.S. city average"
        assert "S12A" in areas
        assert "XXXX" not in areas

    def test_size_class_codes_excluded(self, ingester):
        """Size-class codes (S000, N000, D000, etc.) are excluded."""
        areas = ingester._parse_area_codes(SYNTHETIC_AREA_FILE)
        assert "S000" not in areas
        assert "S100" not in areas
        assert "N000" not in areas

    def test_sub_region_codes_excluded(self, ingester):
        """Sub-region codes (0110, 0230, etc.) are excluded."""
        areas = ingester._parse_area_codes(SYNTHETIC_AREA_FILE)
        assert "0110" not in areas


# ---------------------------------------------------------------------------
# Test: Series ID extraction
# ---------------------------------------------------------------------------

class TestSeriesIDExtraction:
    """Verify area code extraction from series IDs."""

    def test_extract_national(self, ingester):
        """Extracts national area code from series ID."""
        assert ingester._extract_area_code("CUUR0000SAF11") == "0000"

    def test_extract_region(self, ingester):
        """Extracts region area code from series ID."""
        assert ingester._extract_area_code("CUUR0100SAF11") == "0100"

    def test_extract_metro(self, ingester):
        """Extracts metro area code from series ID."""
        assert ingester._extract_area_code("CUURS12ASAF11") == "S12A"

    def test_invalid_prefix_returns_none(self, ingester):
        """Invalid prefix returns None."""
        assert ingester._extract_area_code("XXXX0000SAF11") is None

    def test_invalid_suffix_returns_none(self, ingester):
        """Invalid suffix returns None."""
        assert ingester._extract_area_code("CUUR0000XXXXX") is None


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "bls_cpi_food"

    def test_confidence(self, ingester):
        assert ingester.confidence == "B"

    def test_attribution(self, ingester):
        assert ingester.attribution == "none"
