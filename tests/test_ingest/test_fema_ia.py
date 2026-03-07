"""Tests for the FEMA Individual Assistance ingester (ingest/fema_ia.py).

All HTTP calls are mocked — no real requests to the OpenFEMA API.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

from ingest.fema_ia import (
    FEMA_DECLARATIONS_URL,
    FEMA_IA_FILTER,
    FEMA_PAGE_SIZE,
    FEMAIAIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Matches the real OpenFEMA DisasterDeclarationsSummaries response format
def _make_records(count: int, start: int = 0) -> list[dict]:
    """Generate mock FEMA disaster declaration records.

    Args:
        count: Number of records to generate.
        start: Starting index (offsets disaster number and FIPS).
    """
    records = []
    for i in range(start, start + count):
        state = f"{(i % 50) + 1:02d}"
        county = f"{(i % 200) + 1:03d}"
        records.append({
            "disasterNumber": 4000 + i,
            "fipsStateCode": state,
            "fipsCountyCode": county,
            "declarationDate": f"2020-{(i % 12) + 1:02d}-15T00:00:00.000Z",
            "incidentType": ["Hurricane", "Flood", "Fire", "Tornado"][i % 4],
            "state": "XX",
            "designatedArea": f"County {county}",
        })
    return records


def _make_api_response(
    records: list[dict],
    total_count: int | None = None,
) -> dict:
    """Build an OpenFEMA-style JSON response body."""
    if total_count is None:
        total_count = len(records)
    return {
        "metadata": {"count": total_count},
        "DisasterDeclarationsSummaries": records,
    }


def _make_httpx_response(body: dict) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


# A small set of realistic records for basic tests
SAMPLE_RECORDS = [
    {
        "disasterNumber": 4337,
        "fipsStateCode": "01",
        "fipsCountyCode": "001",
        "declarationDate": "2017-09-14T00:00:00.000Z",
        "incidentType": "Hurricane",
        "state": "AL",
        "designatedArea": "Baldwin County",
    },
    {
        "disasterNumber": 4337,
        "fipsStateCode": "01",
        "fipsCountyCode": "003",
        "declarationDate": "2017-09-14T00:00:00.000Z",
        "incidentType": "Hurricane",
        "state": "AL",
        "designatedArea": "Mobile County",
    },
    {
        "disasterNumber": 4611,
        "fipsStateCode": "06",
        "fipsCountyCode": "037",
        "declarationDate": "2021-07-22T00:00:00.000Z",
        "incidentType": "Fire",
        "state": "CA",
        "designatedArea": "Los Angeles County",
    },
]

# Records with malformed FIPS (county code "000" or missing values)
MALFORMED_FIPS_RECORDS = [
    {
        "disasterNumber": 4500,
        "fipsStateCode": "01",
        "fipsCountyCode": "001",
        "declarationDate": "2020-06-15T00:00:00.000Z",
        "incidentType": "Flood",
        "state": "AL",
        "designatedArea": "Baldwin County",
    },
    {
        "disasterNumber": 4501,
        "fipsStateCode": "12",
        "fipsCountyCode": "000",  # Statewide designation — no specific county
        "declarationDate": "2020-08-10T00:00:00.000Z",
        "incidentType": "Hurricane",
        "state": "FL",
        "designatedArea": "Statewide",
    },
    {
        "disasterNumber": 4502,
        "fipsStateCode": None,
        "fipsCountyCode": None,
        "declarationDate": "2020-09-20T00:00:00.000Z",
        "incidentType": "Tornado",
        "state": "XX",
        "designatedArea": "Unknown",
    },
]


@pytest.fixture
def ingester():
    """Return a fresh FEMAIAIngester instance."""
    return FEMAIAIngester()


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
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        expected = {
            "disaster_number", "fips", "year", "ia_amount",
            "registrant_count", "disaster_type", "declaration_date",
        }
        assert set(df.columns) == expected

    def test_disaster_number_is_string(self, ingester, tmp_raw_dir):
        """disaster_number column is string dtype."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert pd.api.types.is_string_dtype(df["disaster_number"])

    def test_fips_is_string(self, ingester, tmp_raw_dir):
        """FIPS column is string dtype (non-NaN values)."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        valid_fips = df["fips"].dropna()
        assert all(isinstance(v, str) for v in valid_fips)

    def test_year_is_int(self, ingester, tmp_raw_dir):
        """Year column is integer dtype."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert pd.api.types.is_integer_dtype(df["year"])

    def test_ia_amount_is_float(self, ingester, tmp_raw_dir):
        """ia_amount column is float dtype (NaN-compatible)."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert pd.api.types.is_float_dtype(df["ia_amount"])

    def test_registrant_count_is_float(self, ingester, tmp_raw_dir):
        """registrant_count column is float dtype (NaN-compatible)."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert pd.api.types.is_float_dtype(df["registrant_count"])

    def test_disaster_type_is_string(self, ingester, tmp_raw_dir):
        """disaster_type column is string dtype."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert pd.api.types.is_string_dtype(df["disaster_type"])

    def test_declaration_date_is_date(self, ingester, tmp_raw_dir):
        """declaration_date values are Python date objects."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert all(isinstance(d, date) for d in df["declaration_date"])


# ---------------------------------------------------------------------------
# Test: FIPS normalization
# ---------------------------------------------------------------------------

class TestFIPSNormalization:
    """Verify FIPS codes are 5-digit zero-padded strings."""

    def test_standard_fips(self, ingester, tmp_raw_dir):
        """State '01' + county '001' → '01001'."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert "01001" in df["fips"].values
        assert "01003" in df["fips"].values
        assert "06037" in df["fips"].values

    def test_all_valid_fips_5_digits(self, ingester, tmp_raw_dir):
        """All non-NaN FIPS codes are exactly 5 digits."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        valid_fips = df["fips"].dropna()
        assert valid_fips.str.match(r"^\d{5}$").all()

    def test_short_codes_zero_padded(self, ingester, tmp_raw_dir):
        """Shorter state/county codes are zero-padded correctly."""
        records = [{
            "disasterNumber": 5000,
            "fipsStateCode": "6",
            "fipsCountyCode": "37",
            "declarationDate": "2022-01-01T00:00:00.000Z",
            "incidentType": "Fire",
            "state": "CA",
            "designatedArea": "Los Angeles",
        }]
        body = _make_api_response(records)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert "06037" in df["fips"].values


# ---------------------------------------------------------------------------
# Test: Pagination
# ---------------------------------------------------------------------------

class TestPagination:
    """Verify the ingester correctly paginates through multiple pages."""

    def test_three_pages_all_captured(self, ingester, tmp_raw_dir):
        """Mock 3 pages of results — verify all records are captured."""
        page1_records = _make_records(FEMA_PAGE_SIZE, start=0)
        page2_records = _make_records(FEMA_PAGE_SIZE, start=FEMA_PAGE_SIZE)
        page3_records = _make_records(500, start=2 * FEMA_PAGE_SIZE)  # partial last page
        total = len(page1_records) + len(page2_records) + len(page3_records)

        call_count = 0

        def mock_api_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            skip = int(params.get("$skip", "0"))

            if skip == 0:
                return _make_httpx_response(_make_api_response(page1_records, total))
            elif skip == FEMA_PAGE_SIZE:
                return _make_httpx_response(_make_api_response(page2_records, total))
            elif skip == 2 * FEMA_PAGE_SIZE:
                return _make_httpx_response(_make_api_response(page3_records, total))
            else:
                return _make_httpx_response(_make_api_response([], total))

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert len(df) == total
        assert call_count >= 3

    def test_pagination_stops_on_empty_page(self, ingester, tmp_raw_dir):
        """Pagination stops when the API returns an empty records array."""
        records = _make_records(5)

        call_count = 0

        def mock_api_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            skip = int(params.get("$skip", "0"))

            if skip == 0:
                return _make_httpx_response(_make_api_response(records, 5))
            else:
                return _make_httpx_response(_make_api_response([], 5))

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert len(df) == 5

    def test_pagination_stops_on_partial_page(self, ingester, tmp_raw_dir):
        """Pagination stops when a page returns fewer records than $top."""
        # A single partial page (< FEMA_PAGE_SIZE records)
        records = _make_records(50)
        body = _make_api_response(records, total_count=50)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert len(df) == 50


# ---------------------------------------------------------------------------
# Test: Pagination completeness
# ---------------------------------------------------------------------------

class TestPaginationCompleteness:
    """Verify total fetched records matches expected count from API metadata."""

    def test_total_matches_metadata_count(self, ingester, tmp_raw_dir):
        """Total fetched matches the metadata count reported by the API."""
        total_expected = 2500 + 300  # 2500 full pages + 300 partial
        page1 = _make_records(FEMA_PAGE_SIZE, start=0)
        page2 = _make_records(FEMA_PAGE_SIZE, start=FEMA_PAGE_SIZE)
        page3 = _make_records(300, start=2 * FEMA_PAGE_SIZE)  # last partial page

        # Only works when FEMA_PAGE_SIZE >= 1000 and total is 2800
        # Adjust for actual page size
        if FEMA_PAGE_SIZE == 1000:
            pages_data = [page1, page2, page3]
            total_expected = sum(len(p) for p in pages_data)
        else:
            # Fallback: single page
            pages_data = [_make_records(50)]
            total_expected = 50

        page_idx = 0

        def mock_api_get(url, params=None, headers=None):
            nonlocal page_idx
            skip = int(params.get("$skip", "0"))
            idx = skip // FEMA_PAGE_SIZE

            if idx < len(pages_data):
                return _make_httpx_response(
                    _make_api_response(pages_data[idx], total_expected)
                )
            return _make_httpx_response(_make_api_response([], total_expected))

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert len(df) == total_expected

    def test_progress_logging(self, ingester, tmp_raw_dir, caplog):
        """Pagination progress is logged."""
        import logging

        records = _make_records(5)
        body = _make_api_response(records, total_count=5)
        resp = _make_httpx_response(body)

        with caplog.at_level(logging.INFO):
            with patch.object(ingester, "api_get", return_value=resp):
                ingester.fetch()

        assert any("fetched" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Test: Ingest purity (no derived metrics)
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed."""

    def test_no_extra_columns(self, ingester, tmp_raw_dir):
        """Output must not contain derived columns."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        forbidden = {
            "ia_per_household", "annual_total", "severity_ratio",
            "per_capita", "score", "percentile", "county_total",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        expected = {
            "disaster_number", "fips", "year", "ia_amount",
            "registrant_count", "disaster_type", "declaration_date",
        }
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "disaster_number": ["4337"],
            "fips": ["01001"],
            "year": [2017],
            "ia_amount": [np.nan],
            "registrant_count": [np.nan],
            "disaster_type": ["Hurricane"],
            "declaration_date": [date(2017, 9, 14)],
            "ia_per_household": [999.0],  # FORBIDDEN derived column
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir):
        """A _metadata.json file is written next to each year's parquet file."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            ingester.fetch()

        # SAMPLE_RECORDS have years 2017 and 2021
        assert (tmp_raw_dir / "fema_ia" / "fema_ia_2017_metadata.json").exists()
        assert (tmp_raw_dir / "fema_ia" / "fema_ia_2021_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution values."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            ingester.fetch()

        meta_path = tmp_raw_dir / "fema_ia" / "fema_ia_2017_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "FEMA_IA"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_combined_metadata_written(self, ingester, tmp_raw_dir):
        """The combined fema_ia_all file gets a metadata sidecar."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            ingester.fetch()

        assert (tmp_raw_dir / "fema_ia" / "fema_ia_all.parquet").exists()
        assert (tmp_raw_dir / "fema_ia" / "fema_ia_all_metadata.json").exists()


# ---------------------------------------------------------------------------
# Test: ia_amount and registrant_count handling
# ---------------------------------------------------------------------------

class TestIAAmountHandling:
    """Verify ia_amount and registrant_count are NaN (not available from this endpoint)."""

    def test_ia_amount_is_nan(self, ingester, tmp_raw_dir):
        """ia_amount is NaN for all records (not available from declarations endpoint)."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert df["ia_amount"].isna().all()

    def test_registrant_count_is_nan(self, ingester, tmp_raw_dir):
        """registrant_count is NaN for all records (same reason)."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert df["registrant_count"].isna().all()

    def test_nan_records_not_dropped(self, ingester, tmp_raw_dir):
        """All records are preserved despite NaN ia_amount/registrant_count."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert len(df) == len(SAMPLE_RECORDS)


# ---------------------------------------------------------------------------
# Test: Malformed FIPS
# ---------------------------------------------------------------------------

class TestMalformedFIPS:
    """Verify records with missing/malformed FIPS are retained with NaN FIPS."""

    def test_county_code_000_becomes_nan(self, ingester, tmp_raw_dir):
        """Records with county code '000' (statewide) get NaN FIPS."""
        body = _make_api_response(MALFORMED_FIPS_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        # Record with fipsCountyCode "000" should have NaN fips
        statewide_row = df[df["disaster_number"] == "4501"]
        assert len(statewide_row) == 1
        assert pd.isna(statewide_row.iloc[0]["fips"])

    def test_null_state_county_becomes_nan(self, ingester, tmp_raw_dir):
        """Records with None state/county codes get NaN FIPS."""
        body = _make_api_response(MALFORMED_FIPS_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        null_row = df[df["disaster_number"] == "4502"]
        assert len(null_row) == 1
        assert pd.isna(null_row.iloc[0]["fips"])

    def test_malformed_rows_not_dropped(self, ingester, tmp_raw_dir):
        """All rows are preserved, including those with malformed FIPS."""
        body = _make_api_response(MALFORMED_FIPS_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert len(df) == len(MALFORMED_FIPS_RECORDS)

    def test_malformed_fips_logged(self, ingester, tmp_raw_dir, caplog):
        """A warning is logged when records have malformed FIPS."""
        import logging

        body = _make_api_response(MALFORMED_FIPS_RECORDS)
        resp = _make_httpx_response(body)

        with caplog.at_level(logging.WARNING):
            with patch.object(ingester, "api_get", return_value=resp):
                ingester.fetch()

        assert any("malformed fips" in msg.lower() for msg in caplog.messages)

    def test_valid_fips_still_correct(self, ingester, tmp_raw_dir):
        """Valid FIPS records are not affected by malformed ones."""
        body = _make_api_response(MALFORMED_FIPS_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        valid_row = df[df["disaster_number"] == "4500"]
        assert valid_row.iloc[0]["fips"] == "01001"


# ---------------------------------------------------------------------------
# Test: Declaration date parsing & year extraction
# ---------------------------------------------------------------------------

class TestDateParsing:
    """Verify declaration dates are parsed and years extracted correctly."""

    def test_date_parsed(self, ingester, tmp_raw_dir):
        """Declaration dates are parsed as Python date objects."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        first_date = df[df["fips"] == "01001"].iloc[0]["declaration_date"]
        assert first_date == date(2017, 9, 14)

    def test_year_extracted(self, ingester, tmp_raw_dir):
        """Year is extracted from declaration date."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert set(df["year"].unique()) == {2017, 2021}

    def test_disaster_type_preserved(self, ingester, tmp_raw_dir):
        """Disaster types from API are preserved."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert "Hurricane" in df["disaster_type"].values
        assert "Fire" in df["disaster_type"].values


# ---------------------------------------------------------------------------
# Test: Year filtering
# ---------------------------------------------------------------------------

class TestYearFiltering:
    """Verify that the years parameter filters results post-fetch."""

    def test_filter_to_single_year(self, ingester, tmp_raw_dir):
        """Passing years=[2017] returns only 2017 records."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch(years=[2017])

        assert set(df["year"].unique()) == {2017}
        assert len(df) == 2  # Two 2017 records in SAMPLE_RECORDS

    def test_filter_no_match_returns_empty(self, ingester, tmp_raw_dir):
        """Passing years with no matching records returns empty DataFrame."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch(years=[1999])

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)


# ---------------------------------------------------------------------------
# Test: Caching
# ---------------------------------------------------------------------------

class TestCaching:
    """Verify per-year and combined caching behavior."""

    def test_per_year_parquet_files(self, ingester, tmp_raw_dir):
        """Each year in the data gets its own parquet file."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            ingester.fetch()

        assert (tmp_raw_dir / "fema_ia" / "fema_ia_2017.parquet").exists()
        assert (tmp_raw_dir / "fema_ia" / "fema_ia_2021.parquet").exists()

    def test_combined_parquet_file(self, ingester, tmp_raw_dir):
        """A combined fema_ia_all.parquet is written."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            ingester.fetch()

        assert (tmp_raw_dir / "fema_ia" / "fema_ia_all.parquet").exists()


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic on transient HTTP failures."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        body = _make_api_response(SAMPLE_RECORDS)
        ok_resp = _make_httpx_response(body)

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
            resp = ingester.api_get(FEMA_DECLARATIONS_URL)

        assert call_count == 2

    def test_retries_on_503(self, ingester, tmp_raw_dir):
        """Retries after HTTP 503."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        body = _make_api_response(SAMPLE_RECORDS)
        ok_resp = _make_httpx_response(body)

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
            resp = ingester.api_get(FEMA_DECLARATIONS_URL)

        assert call_count == 3


# ---------------------------------------------------------------------------
# Test: Partial pagination failure
# ---------------------------------------------------------------------------

class TestPartialPaginationFailure:
    """Verify that data from successful pages is preserved when pagination fails."""

    def test_pages_before_failure_returned(self, ingester, tmp_raw_dir):
        """If pagination fails on page 3, pages 1-2 data is still returned."""
        # Each page must be full-size (FEMA_PAGE_SIZE) so pagination continues
        page1_records = _make_records(FEMA_PAGE_SIZE, start=0)
        page2_records = _make_records(FEMA_PAGE_SIZE, start=FEMA_PAGE_SIZE)
        total = FEMA_PAGE_SIZE * 3  # API says there are 3 pages total

        def mock_api_get(url, params=None, headers=None):
            skip = int(params.get("$skip", "0"))

            if skip == 0:
                return _make_httpx_response(_make_api_response(page1_records, total))
            elif skip == FEMA_PAGE_SIZE:
                return _make_httpx_response(_make_api_response(page2_records, total))
            else:
                raise httpx.TransportError("Connection reset")

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        # Should have records from pages 1 and 2
        assert len(df) == 2 * FEMA_PAGE_SIZE

    def test_partial_data_is_cached(self, ingester, tmp_raw_dir):
        """Data from successful pages is cached even when later pages fail."""
        page1_records = _make_records(5, start=0)
        # All records will be year 2020 based on _make_records

        def mock_api_get(url, params=None, headers=None):
            skip = int(params.get("$skip", "0"))
            if skip == 0:
                return _make_httpx_response(_make_api_response(page1_records, 100))
            raise httpx.TransportError("Timeout")

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert not df.empty
        # Check that at least one cache file was written
        cache_files = list((tmp_raw_dir / "fema_ia").glob("*.parquet"))
        assert len(cache_files) > 0

    def test_total_api_failure_returns_empty(self, ingester, tmp_raw_dir):
        """If the very first page fails, return empty DataFrame."""
        def mock_api_get(url, params=None, headers=None):
            raise httpx.TransportError("Connection refused")

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)


# ---------------------------------------------------------------------------
# Test: Completeness logging
# ---------------------------------------------------------------------------

class TestCompletenessLogging:
    """Verify log_completeness reports county coverage."""

    def test_completeness_logged(self, ingester, tmp_raw_dir, caplog):
        """run() logs county count and coverage percentage."""
        import logging

        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with caplog.at_level(logging.INFO):
            with patch.object(ingester, "api_get", return_value=resp):
                ingester.run()

        assert any("counties" in msg.lower() for msg in caplog.messages)

    def test_county_count(self, ingester, tmp_raw_dir):
        """Sample data has 3 unique counties."""
        body = _make_api_response(SAMPLE_RECORDS)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert df["fips"].dropna().nunique() == 3


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "fema_ia"

    def test_confidence(self, ingester):
        assert ingester.confidence == "B"

    def test_attribution(self, ingester):
        assert ingester.attribution == "proxy"

    def test_calls_per_second(self, ingester):
        """Rate limit is polite (<=1 req/sec for federal API)."""
        assert ingester.calls_per_second <= 1.0


# ---------------------------------------------------------------------------
# Test: Disaster type handling
# ---------------------------------------------------------------------------

class TestDisasterType:
    """Verify disaster type handling including missing values."""

    def test_missing_incident_type_becomes_unknown(self, ingester, tmp_raw_dir):
        """Records with null incidentType get 'Unknown' disaster_type."""
        records = [{
            "disasterNumber": 9999,
            "fipsStateCode": "01",
            "fipsCountyCode": "001",
            "declarationDate": "2022-01-01T00:00:00.000Z",
            "incidentType": None,
            "state": "AL",
            "designatedArea": "Test County",
        }]
        body = _make_api_response(records)
        resp = _make_httpx_response(body)

        with patch.object(ingester, "api_get", return_value=resp):
            df = ingester.fetch()

        assert df.iloc[0]["disaster_type"] == "Unknown"
