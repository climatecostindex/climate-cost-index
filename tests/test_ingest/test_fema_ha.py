"""Tests for the FEMA Housing Assistance ingester (ingest/fema_ha.py).

All HTTP calls are mocked — no real requests to the OpenFEMA API.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from ingest.fema_ha import (
    FEMA_HA_ENTITY_NAMES,
    FEMA_HA_OWNERS_URL,
    FEMA_HA_PAGE_SIZE,
    FEMA_HA_RENTERS_URL,
    FEMAHAIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_ha_records(
    count: int,
    disaster_number: int = 4337,
    state: str = "AL",
    county: str = "Barbour (County)",
    amount: float = 1000.0,
    registrations: int = 5,
    start_idx: int = 0,
) -> list[dict]:
    """Generate mock FEMA Housing Assistance records.

    Each record represents a city/zip-level payout within a county.
    """
    records = []
    for i in range(start_idx, start_idx + count):
        records.append({
            "disasterNumber": disaster_number,
            "state": state,
            "county": county,
            "totalApprovedIhpAmount": amount,
            "validRegistrations": registrations,
        })
    return records


def _make_api_response(
    records: list[dict],
    entity_name: str = "HousingAssistanceOwners",
    total_count: int | None = None,
) -> dict:
    """Build an OpenFEMA-style JSON response body."""
    if total_count is None:
        total_count = len(records)
    return {
        "metadata": {"count": total_count},
        entity_name: records,
    }


def _make_httpx_response(body: dict) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


# Sample records: same disaster + county, different cities (should aggregate)
OWNERS_RECORDS = [
    {
        "disasterNumber": 4337,
        "state": "AL",
        "county": "Barbour (County)",
        "totalApprovedIhpAmount": 5000.0,
        "validRegistrations": 10,
    },
    {
        "disasterNumber": 4337,
        "state": "AL",
        "county": "Barbour (County)",
        "totalApprovedIhpAmount": 3000.0,
        "validRegistrations": 6,
    },
    {
        "disasterNumber": 4337,
        "state": "AL",
        "county": "Jefferson (Parish)",
        "totalApprovedIhpAmount": 12000.0,
        "validRegistrations": 20,
    },
]

RENTERS_RECORDS = [
    {
        "disasterNumber": 4337,
        "state": "AL",
        "county": "Barbour (County)",
        "totalApprovedIhpAmount": 2000.0,
        "validRegistrations": 4,
    },
    {
        "disasterNumber": 4337,
        "state": "AL",
        "county": "Jefferson (Parish)",
        "totalApprovedIhpAmount": 8000.0,
        "validRegistrations": 15,
    },
]


@pytest.fixture
def ingester():
    """Return a fresh FEMAHAIngester instance."""
    return FEMAHAIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


def _mock_api_get_factory(owners_records, renters_records):
    """Create a mock api_get that returns different data per endpoint.

    Handles single-page responses. Both endpoints return all records on
    the first page and an empty array on subsequent pages.
    """
    def mock_api_get(url, params=None, headers=None):
        skip = int(params.get("$skip", "0")) if params else 0
        entity_name = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")

        if url == FEMA_HA_OWNERS_URL:
            records = owners_records if skip == 0 else []
            total = len(owners_records)
        elif url == FEMA_HA_RENTERS_URL:
            records = renters_records if skip == 0 else []
            total = len(renters_records)
        else:
            records = []
            total = 0

        body = _make_api_response(records, entity_name=entity_name, total_count=total)
        return _make_httpx_response(body)

    return mock_api_get


# ---------------------------------------------------------------------------
# Test: Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has all expected columns with correct dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains all expected columns."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        expected = {"disaster_number", "state", "county", "ia_amount", "registrant_count"}
        assert set(df.columns) == expected

    def test_column_types(self, ingester, tmp_raw_dir):
        """Verify column dtypes match spec."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        assert pd.api.types.is_string_dtype(df["disaster_number"])
        assert pd.api.types.is_string_dtype(df["state"])
        assert pd.api.types.is_string_dtype(df["county"])
        assert pd.api.types.is_float_dtype(df["ia_amount"])
        # registrant_count is float (to allow aggregation)
        assert pd.api.types.is_float_dtype(df["registrant_count"])

    def test_no_extra_columns(self, ingester, tmp_raw_dir):
        """Output contains ONLY the specified columns."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        expected = {"disaster_number", "state", "county", "ia_amount", "registrant_count"}
        assert set(df.columns) == expected


# ---------------------------------------------------------------------------
# Test: Data correctness
# ---------------------------------------------------------------------------

class TestDataCorrectness:
    """Verify aggregation and data transformations."""

    def test_owners_renters_combined(self, ingester, tmp_raw_dir):
        """ia_amount sums across both owners and renters for same county-disaster."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        barbour = df[
            (df["disaster_number"] == "4337")
            & (df["county"] == "Barbour (County)")
        ]
        assert len(barbour) == 1
        # Owners: 5000 + 3000 = 8000, Renters: 2000 → total 10000
        assert barbour.iloc[0]["ia_amount"] == 10000.0
        # Registrants: 10 + 6 + 4 = 20
        assert barbour.iloc[0]["registrant_count"] == 20.0

    def test_county_level_aggregation(self, ingester, tmp_raw_dir):
        """Multiple city-level records are aggregated to one per county-disaster."""
        # OWNERS_RECORDS has 2 rows for Barbour (County) under disaster 4337
        mock = _mock_api_get_factory(OWNERS_RECORDS, [])
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        barbour = df[
            (df["disaster_number"] == "4337")
            & (df["county"] == "Barbour (County)")
        ]
        assert len(barbour) == 1
        assert barbour.iloc[0]["ia_amount"] == 8000.0  # 5000 + 3000

    def test_disaster_number_is_string(self, ingester, tmp_raw_dir):
        """disaster_number is converted from int to string."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, [])
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        assert all(isinstance(v, str) for v in df["disaster_number"])
        assert "4337" in df["disaster_number"].values

    def test_county_name_preserved(self, ingester, tmp_raw_dir):
        """County names with suffixes are preserved exactly."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        assert "Barbour (County)" in df["county"].values
        assert "Jefferson (Parish)" in df["county"].values

    def test_zero_dollar_records_preserved(self, ingester, tmp_raw_dir):
        """Records with $0 ia_amount are preserved, not filtered out."""
        records = [
            {
                "disasterNumber": 5000,
                "state": "TX",
                "county": "Harris (County)",
                "totalApprovedIhpAmount": 0.0,
                "validRegistrations": 3,
            },
        ]
        mock = _mock_api_get_factory(records, [])
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        harris = df[df["county"] == "Harris (County)"]
        assert len(harris) == 1
        assert harris.iloc[0]["ia_amount"] == 0.0

    def test_null_amounts_treated_as_zero(self, ingester, tmp_raw_dir):
        """Records with null totalApprovedIhpAmount are treated as 0."""
        records = [
            {
                "disasterNumber": 5001,
                "state": "FL",
                "county": "Miami-Dade (County)",
                "totalApprovedIhpAmount": None,
                "validRegistrations": None,
            },
        ]
        mock = _mock_api_get_factory(records, [])
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        row = df[df["county"] == "Miami-Dade (County)"]
        assert len(row) == 1
        assert row.iloc[0]["ia_amount"] == 0.0
        assert row.iloc[0]["registrant_count"] == 0.0


# ---------------------------------------------------------------------------
# Test: Pagination
# ---------------------------------------------------------------------------

class TestPagination:
    """Verify the ingester correctly paginates through multiple pages."""

    def test_multi_page_fetch(self, ingester, tmp_raw_dir):
        """Mock 3+ pages for one endpoint — verify all records captured."""
        page1 = _make_ha_records(FEMA_HA_PAGE_SIZE, disaster_number=4337, amount=100.0, start_idx=0)
        page2 = _make_ha_records(FEMA_HA_PAGE_SIZE, disaster_number=4338, amount=200.0, start_idx=0)
        page3 = _make_ha_records(500, disaster_number=4339, amount=300.0, start_idx=0)
        total = len(page1) + len(page2) + len(page3)

        def mock_api_get(url, params=None, headers=None):
            skip = int(params.get("$skip", "0")) if params else 0
            entity_name = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")

            if url == FEMA_HA_OWNERS_URL:
                if skip == 0:
                    records = page1
                elif skip == FEMA_HA_PAGE_SIZE:
                    records = page2
                elif skip == 2 * FEMA_HA_PAGE_SIZE:
                    records = page3
                else:
                    records = []
                body = _make_api_response(records, entity_name=entity_name, total_count=total)
            else:
                # Renters endpoint: empty
                body = _make_api_response([], entity_name=entity_name, total_count=0)
            return _make_httpx_response(body)

        with patch.object(ingester, "api_get", side_effect=mock_api_get), \
             patch("ingest.fema_ha.time.sleep"):
            df = ingester.fetch()

        # 3 unique disasters, each aggregated to 1 row
        assert len(df) == 3

    def test_pagination_stops_on_partial_page(self, ingester, tmp_raw_dir):
        """Pagination stops when final page has fewer records than $top."""
        records = _make_ha_records(50, disaster_number=4337)

        def mock_api_get(url, params=None, headers=None):
            skip = int(params.get("$skip", "0")) if params else 0
            entity_name = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")

            if url == FEMA_HA_OWNERS_URL and skip == 0:
                body = _make_api_response(records, entity_name=entity_name, total_count=50)
            elif url == FEMA_HA_RENTERS_URL and skip == 0:
                body = _make_api_response([], entity_name=entity_name, total_count=0)
            else:
                entity = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")
                body = _make_api_response([], entity_name=entity, total_count=0)
            return _make_httpx_response(body)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        # All 50 records have same (disaster, state, county) → 1 aggregated row
        assert len(df) == 1

    def test_pagination_stops_on_empty_page(self, ingester, tmp_raw_dir):
        """Pagination stops when the API returns an empty records array."""
        records = _make_ha_records(5, disaster_number=4337)

        call_count = 0

        def mock_api_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            skip = int(params.get("$skip", "0")) if params else 0
            entity_name = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")

            if url == FEMA_HA_OWNERS_URL:
                if skip == 0:
                    recs = records
                else:
                    recs = []
                body = _make_api_response(recs, entity_name=entity_name, total_count=5)
            else:
                body = _make_api_response([], entity_name=entity_name, total_count=0)
            return _make_httpx_response(body)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert not df.empty

    def test_progress_logging(self, ingester, tmp_raw_dir, caplog):
        """Pagination progress is logged."""
        import logging

        records = _make_ha_records(5, disaster_number=4337)
        mock = _mock_api_get_factory(records, [])

        with caplog.at_level(logging.INFO):
            with patch.object(ingester, "api_get", side_effect=mock):
                ingester.fetch()

        assert any("fetched" in msg.lower() or "finished" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Test: Pagination completeness
# ---------------------------------------------------------------------------

class TestPaginationCompleteness:
    """Verify total fetched records matches expected count from API metadata."""

    def test_total_matches_metadata_count(self, ingester, tmp_raw_dir, caplog):
        """Total fetched matches the metadata count."""
        import logging

        records = _make_ha_records(50, disaster_number=4337)

        def mock_api_get(url, params=None, headers=None):
            skip = int(params.get("$skip", "0")) if params else 0
            entity_name = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")

            if url == FEMA_HA_OWNERS_URL and skip == 0:
                body = _make_api_response(records, entity_name=entity_name, total_count=50)
            elif url == FEMA_HA_RENTERS_URL and skip == 0:
                body = _make_api_response([], entity_name=entity_name, total_count=0)
            else:
                body = _make_api_response([], entity_name=entity_name, total_count=0)
            return _make_httpx_response(body)

        with caplog.at_level(logging.INFO):
            with patch.object(ingester, "api_get", side_effect=mock_api_get):
                ingester.fetch()

        # The log should show "50 records fetched (expected 50)"
        assert any("50" in msg and "fetched" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Test: Resilience
# ---------------------------------------------------------------------------

class TestResilience:
    """Verify graceful handling of partial and total failures."""

    def test_partial_pagination_failure(self, ingester, tmp_raw_dir):
        """If pagination fails on page 2, page 1 data is still returned."""
        page1 = _make_ha_records(FEMA_HA_PAGE_SIZE, disaster_number=4337, amount=100.0)

        def mock_api_get(url, params=None, headers=None):
            skip = int(params.get("$skip", "0")) if params else 0
            entity_name = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")

            if url == FEMA_HA_OWNERS_URL:
                if skip == 0:
                    body = _make_api_response(
                        page1, entity_name=entity_name, total_count=2000
                    )
                    return _make_httpx_response(body)
                else:
                    raise httpx.TransportError("Connection reset")
            else:
                body = _make_api_response([], entity_name=entity_name, total_count=0)
                return _make_httpx_response(body)

        with patch.object(ingester, "api_get", side_effect=mock_api_get), \
             patch("ingest.fema_ha.time.sleep"):
            df = ingester.fetch()

        # Page 1 data should still be present (aggregated to 1 row)
        assert not df.empty
        assert "4337" in df["disaster_number"].values

    def test_one_endpoint_fails_other_succeeds(self, ingester, tmp_raw_dir):
        """If renters endpoint fails entirely, owners data is still returned."""
        def mock_api_get(url, params=None, headers=None):
            skip = int(params.get("$skip", "0")) if params else 0
            entity_name = FEMA_HA_ENTITY_NAMES.get(url, "HousingAssistanceOwners")

            if url == FEMA_HA_OWNERS_URL:
                if skip == 0:
                    body = _make_api_response(
                        OWNERS_RECORDS, entity_name=entity_name
                    )
                    return _make_httpx_response(body)
                else:
                    body = _make_api_response([], entity_name=entity_name, total_count=len(OWNERS_RECORDS))
                    return _make_httpx_response(body)
            else:
                raise httpx.TransportError("Connection refused")

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert not df.empty
        assert "Barbour (County)" in df["county"].values

    def test_both_endpoints_fail(self, ingester, tmp_raw_dir):
        """If both endpoints fail, return empty DataFrame with correct schema."""
        def mock_api_get(url, params=None, headers=None):
            raise httpx.TransportError("Connection refused")

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            df = ingester.fetch()

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)

    def test_retry_on_500_503(self, ingester, tmp_raw_dir):
        """Retry logic triggers on HTTP 500/503 (inherited from base class)."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        body = _make_api_response(
            OWNERS_RECORDS, entity_name="HousingAssistanceOwners"
        )
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
            resp = ingester.api_get(FEMA_HA_OWNERS_URL)

        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: Ingest purity
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed."""

    def test_no_derived_metrics(self, ingester, tmp_raw_dir):
        """Output contains NO derived columns."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        forbidden = {
            "ia_per_household", "fema_ncei_ratio", "severity_score",
            "fips", "percentile", "annual_total", "per_capita",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_no_fips_column(self, ingester, tmp_raw_dir):
        """Output does NOT contain a fips column."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        assert "fips" not in df.columns

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the 5 specified."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            df = ingester.fetch()

        expected = {"disaster_number", "state", "county", "ia_amount", "registrant_count"}
        assert set(df.columns) == expected


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir):
        """fema_ha_all_metadata.json is written alongside the parquet file."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            ingester.fetch()

        assert (tmp_raw_dir / "fema_ha" / "fema_ha_all_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            ingester.fetch()

        meta_path = tmp_raw_dir / "fema_ha" / "fema_ha_all_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "FEMA_HA"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_parquet_file_created(self, ingester, tmp_raw_dir):
        """fema_ha_all.parquet is written."""
        mock = _mock_api_get_factory(OWNERS_RECORDS, RENTERS_RECORDS)
        with patch.object(ingester, "api_get", side_effect=mock):
            ingester.fetch()

        assert (tmp_raw_dir / "fema_ha" / "fema_ha_all.parquet").exists()


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "fema_ha"

    def test_confidence(self, ingester):
        assert ingester.confidence == "B"

    def test_attribution(self, ingester):
        assert ingester.attribution == "proxy"
