"""Tests for the USDM Drought ingester (ingest/usdm_drought.py).

All HTTP calls are mocked — no real requests to the USDM API.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from ingest.usdm_drought import (
    USDM_API_BASE,
    USDMDroughtIngester,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_usdm_record(
    fips: str = "01001",
    map_date: str = "2024-01-02T00:00:00",
    none_pct: float = 50.0,
    d0: float = 20.0,
    d1: float = 15.0,
    d2: float = 10.0,
    d3: float = 5.0,
    d4: float = 0.0,
) -> dict:
    """Build a single USDM API response record (matches real API camelCase keys)."""
    return {
        "mapDate": map_date,
        "fips": fips,
        "county": "Test County",
        "state": "AL",
        "none": none_pct,
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "validStart": "2023-12-26T00:00:00",
        "validEnd": map_date,
        "statisticFormatID": 2,
    }


def _make_response(records: list[dict], status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = records
    resp.raise_for_status = MagicMock()
    return resp


SAMPLE_RECORDS = [
    _make_usdm_record(fips="01001", map_date="2024-01-02T00:00:00"),
    _make_usdm_record(fips="01003", map_date="2024-01-02T00:00:00", none_pct=30.0, d0=30.0, d1=20.0, d2=10.0, d3=7.0, d4=3.0),
    _make_usdm_record(fips="01005", map_date="2024-01-09T00:00:00", none_pct=100.0, d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0),
]


@pytest.fixture
def ingester():
    """Return a fresh USDMDroughtIngester instance."""
    return USDMDroughtIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory so caching writes to disk safely."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Test: Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has all expected columns with correct dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required columns."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch.object(type(ingester), "_default_years", return_value=[2024]):
                with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                    df = ingester.fetch(years=[2024])

        assert set(df.columns) == {"fips", "date", "d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "none_pct"}

    def test_fips_is_string(self, ingester, tmp_raw_dir):
        """FIPS column is string dtype."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        assert pd.api.types.is_string_dtype(df["fips"])

    def test_date_is_date_object(self, ingester, tmp_raw_dir):
        """Date column contains Python date objects."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        assert all(isinstance(d, date) for d in df["date"])

    def test_pct_cols_are_float(self, ingester, tmp_raw_dir):
        """Drought percentage columns are float dtype."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        for col in ["d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "none_pct"]:
            assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"


# ---------------------------------------------------------------------------
# Test: FIPS normalization
# ---------------------------------------------------------------------------

class TestFIPSNormalization:
    """Verify FIPS codes are 5-digit zero-padded strings."""

    @pytest.mark.parametrize("raw_fips,expected", [
        ("01001", "01001"),
        ("1001", "01001"),
        (1001, "01001"),
        (1001.0, "01001"),
    ])
    def test_fips_formats(self, ingester, tmp_raw_dir, raw_fips, expected):
        """Various raw FIPS formats are normalized to 5-digit strings."""
        record = _make_usdm_record(fips=raw_fips)
        with patch.object(ingester, "api_get", return_value=_make_response([record])):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        assert df.iloc[0]["fips"] == expected

    def test_all_fips_are_5_digit(self, ingester, tmp_raw_dir):
        """Every FIPS in output matches the 5-digit pattern."""
        records = [
            _make_usdm_record(fips="1001"),
            _make_usdm_record(fips="06075"),
            _make_usdm_record(fips=48201),
        ]
        with patch.object(ingester, "api_get", return_value=_make_response(records)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        assert df["fips"].str.match(r"^\d{5}$").all()


# ---------------------------------------------------------------------------
# Test: Percentage validation
# ---------------------------------------------------------------------------

class TestPercentageValidation:
    """Verify warnings when percentages don't sum to ~100%."""

    def test_valid_sums_no_warning(self, ingester, tmp_raw_dir, caplog):
        """No warning logged when percentages sum to 100%."""
        # Default records sum to 100%
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                ingester.fetch(years=[2024])

        assert "percentage sums outside" not in caplog.text

    def test_bad_sum_triggers_warning(self, ingester, tmp_raw_dir, caplog):
        """Warning logged when percentages sum far from 100%."""
        bad_record = _make_usdm_record(
            none_pct=10.0, d0=10.0, d1=10.0, d2=10.0, d3=10.0, d4=10.0,
        )  # Sums to 60%
        import logging
        with caplog.at_level(logging.WARNING):
            with patch.object(ingester, "api_get", return_value=_make_response([bad_record])):
                with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                    ingester.fetch(years=[2024])

        assert "percentage sums outside" in caplog.text

    def test_within_tolerance_no_warning(self, ingester, tmp_raw_dir, caplog):
        """No warning when sums are within ±2% of 100%."""
        # 99.5% total — within tolerance
        record = _make_usdm_record(
            none_pct=49.5, d0=20.0, d1=15.0, d2=10.0, d3=5.0, d4=0.0,
        )
        with patch.object(ingester, "api_get", return_value=_make_response([record])):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                ingester.fetch(years=[2024])

        assert "percentage sums outside" not in caplog.text


# ---------------------------------------------------------------------------
# Test: Ingest purity (no derived metrics)
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed — output is raw only."""

    def test_no_extra_columns(self, ingester, tmp_raw_dir):
        """Output must not contain derived columns like max_severity, drought_score, etc."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        forbidden = {"max_severity", "drought_score", "weeks_in_drought", "severity_area_integral",
                     "annual_avg", "year", "score", "rank", "percentile"}
        assert forbidden.isdisjoint(set(df.columns)), (
            f"Output contains derived columns: {forbidden & set(df.columns)}"
        )

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set, no more, no less."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        expected = {"fips", "date", "d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "none_pct"}
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "fips": ["01001"],
            "date": [date(2024, 1, 2)],
            "d0_pct": [20.0],
            "d1_pct": [15.0],
            "d2_pct": [10.0],
            "d3_pct": [5.0],
            "d4_pct": [0.0],
            "none_pct": [50.0],
            "drought_score": [999.0],  # ← extra derived column
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
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                ingester.fetch(years=[2024])

        meta_path = tmp_raw_dir / "usdm" / "usdm_2024_metadata.json"
        assert meta_path.exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution values."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                ingester.fetch(years=[2024])

        meta_path = tmp_raw_dir / "usdm" / "usdm_2024_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "USDM"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_combined_metadata_also_written(self, ingester, tmp_raw_dir):
        """The combined usdm_all file also gets a metadata sidecar."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                ingester.fetch(years=[2024])

        assert (tmp_raw_dir / "usdm" / "usdm_all_metadata.json").exists()
        assert (tmp_raw_dir / "usdm" / "usdm_all.parquet").exists()


# ---------------------------------------------------------------------------
# Test: Completeness logging
# ---------------------------------------------------------------------------

class TestCompletenessLogging:
    """Verify log_completeness reports county coverage percentage."""

    def test_completeness_logged(self, ingester, tmp_raw_dir, caplog):
        """log_completeness produces an INFO log with county count and percentage."""
        import logging
        with caplog.at_level(logging.INFO):
            with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
                with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                    df = ingester.run(years=[2024])

        # Should see a log line mentioning counties and percentage
        assert any("counties" in msg.lower() for msg in caplog.messages)

    def test_completeness_values_reasonable(self, ingester, tmp_raw_dir):
        """With 3 sample records, we get 3 unique FIPS codes."""
        with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        assert df["fips"].nunique() == 3


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic on transient HTTP failures."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = _make_response(SAMPLE_RECORDS)

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
            # Fetch one state, one year — should retry and succeed
            df = ingester._fetch_state_year("AL", 2024)

        assert not df.empty
        assert call_count == 2

    def test_retries_on_503(self, ingester, tmp_raw_dir):
        """Retries after HTTP 503."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        ok_resp = _make_response(SAMPLE_RECORDS)

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
            df = ingester._fetch_state_year("AL", 2024)

        assert not df.empty
        assert call_count == 3

    def test_exhausts_retries(self, ingester, tmp_raw_dir):
        """Raises after exhausting max retries on persistent 500."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500
        fail_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=fail_resp,
        )

        with patch.object(ingester, "_client", MagicMock()):
            ingester._client.get = MagicMock(return_value=fail_resp)
            ingester._last_call_time = 0.0
            with pytest.raises(httpx.HTTPStatusError):
                ingester._fetch_state_year("AL", 2024)


# ---------------------------------------------------------------------------
# Test: Partial failure
# ---------------------------------------------------------------------------

class TestPartialFailure:
    """Verify that if one year fails, other years' data is still cached."""

    def test_partial_year_failure(self, ingester, tmp_raw_dir):
        """Data from successful years is returned even when one year fails."""
        good_resp = _make_response(SAMPLE_RECORDS)

        def mock_api_get(url, params=None, headers=None):
            # Fail for year 2023, succeed for 2024
            if params and "1/1/2023" in str(params.get("startdate", "")):
                raise httpx.TransportError("Connection refused")
            return good_resp

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2023, 2024])

        # Should have data from 2024 even though 2023 failed
        assert not df.empty
        # Check that 2024 parquet was cached
        assert (tmp_raw_dir / "usdm" / "usdm_2024.parquet").exists()

    def test_partial_state_failure(self, ingester, tmp_raw_dir):
        """Data from successful states is returned when one state fails."""
        al_records = [_make_usdm_record(fips="01001")]
        ca_records = [_make_usdm_record(fips="06001")]

        def mock_api_get(url, params=None, headers=None):
            aoi = params.get("aoi", "") if params else ""
            if aoi == "AK":
                raise httpx.TransportError("Timeout")
            if aoi == "CA":
                return _make_response(ca_records)
            return _make_response(al_records)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL", "AK", "CA"]):
                df = ingester.fetch(years=[2024])

        # Should have data from AL and CA but not AK
        fips_set = set(df["fips"])
        assert "01001" in fips_set
        assert "06001" in fips_set

    def test_all_years_fail_returns_empty(self, ingester, tmp_raw_dir):
        """Returns empty DataFrame when every request fails."""
        with patch.object(ingester, "api_get", side_effect=httpx.TransportError("fail")):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)


# ---------------------------------------------------------------------------
# Test: Base class integration
# ---------------------------------------------------------------------------

class TestBaseClassIntegration:
    """Verify the ingester works correctly with base class run() method."""

    def test_run_validates_and_caches(self, ingester, tmp_raw_dir, caplog):
        """run() calls validate_output and log_completeness."""
        import logging
        with caplog.at_level(logging.INFO):
            with patch.object(ingester, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
                with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                    df = ingester.run(years=[2024])

        assert not df.empty
        assert any("completeness" in msg.lower() for msg in caplog.messages)

    def test_context_manager(self, tmp_raw_dir):
        """Ingester works as a context manager and closes client."""
        with USDMDroughtIngester() as ing:
            with patch.object(ing, "api_get", return_value=_make_response(SAMPLE_RECORDS)):
                with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                    df = ing.fetch(years=[2024])
                    assert not df.empty
        # After exit, client should be closed
        assert ing._client is None

    def test_empty_api_response(self, ingester, tmp_raw_dir):
        """Handles empty JSON response gracefully."""
        with patch.object(ingester, "api_get", return_value=_make_response([])):
            with patch("ingest.usdm_drought.US_STATE_CODES", ["AL"]):
                df = ingester.fetch(years=[2024])

        assert df.empty
