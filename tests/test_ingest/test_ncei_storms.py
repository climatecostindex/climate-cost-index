"""Tests for the NCEI Storm Events ingester (ingest/ncei_storms.py).

All HTTP calls are mocked — no real requests to ncei.noaa.gov.
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

from ingest.ncei_storms import (
    NCEI_DETAILS_PATTERN,
    NCEI_LOCATIONS_PATTERN,
    NCEI_STORMS_BASE_URL,
    NCEIStormsIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Synthetic directory listing HTML matching the real NCEI format
LISTING_HTML = """
<html><body><pre>
<a href="StormEvents_details-ftp_v1.0_d2022_c20240101.csv.gz">StormEvents_details-ftp_v1.0_d2022_c20240101.csv.gz</a>
<a href="StormEvents_details-ftp_v1.0_d2023_c20240201.csv.gz">StormEvents_details-ftp_v1.0_d2023_c20240201.csv.gz</a>
<a href="StormEvents_locations-ftp_v1.0_d2022_c20240101.csv.gz">StormEvents_locations-ftp_v1.0_d2022_c20240101.csv.gz</a>
<a href="StormEvents_locations-ftp_v1.0_d2023_c20240201.csv.gz">StormEvents_locations-ftp_v1.0_d2023_c20240201.csv.gz</a>
</pre></body></html>
"""


def _build_details_csv(rows: list[dict] | None = None) -> bytes:
    """Build a synthetic StormEvents_details gzipped CSV.

    Args:
        rows: List of dicts with column overrides. Missing columns get defaults.

    Returns:
        Gzipped CSV bytes.
    """
    defaults = {
        "EVENT_ID": "10001",
        "STATE_FIPS": "1",
        "CZ_TYPE": "C",
        "CZ_FIPS": "1",
        "BEGIN_YEARMONTH": "202201",
        "BEGIN_DAY": "15",
        "EVENT_TYPE": "Thunderstorm Wind",
        "DAMAGE_PROPERTY": "25K",
        "DAMAGE_CROPS": "0",
        "INJURIES_DIRECT": "0",
        "DEATHS_DIRECT": "0",
        "MAGNITUDE": "60",
    }

    if rows is None:
        rows = [
            {
                "EVENT_ID": "10001",
                "STATE_FIPS": "1",
                "CZ_TYPE": "C",
                "CZ_FIPS": "1",
                "BEGIN_YEARMONTH": "202201",
                "BEGIN_DAY": "15",
                "EVENT_TYPE": "Thunderstorm Wind",
                "DAMAGE_PROPERTY": "25K",
                "DAMAGE_CROPS": "5K",
                "INJURIES_DIRECT": "2",
                "DEATHS_DIRECT": "0",
                "MAGNITUDE": "60",
            },
            {
                "EVENT_ID": "10002",
                "STATE_FIPS": "48",
                "CZ_TYPE": "C",
                "CZ_FIPS": "201",
                "BEGIN_YEARMONTH": "202203",
                "BEGIN_DAY": "20",
                "EVENT_TYPE": "Tornado",
                "DAMAGE_PROPERTY": "1.5M",
                "DAMAGE_CROPS": "0.5M",
                "INJURIES_DIRECT": "5",
                "DEATHS_DIRECT": "1",
                "MAGNITUDE": "2",
            },
            {
                "EVENT_ID": "10003",
                "STATE_FIPS": "12",
                "CZ_TYPE": "C",
                "CZ_FIPS": "86",
                "BEGIN_YEARMONTH": "202206",
                "BEGIN_DAY": "1",
                "EVENT_TYPE": "Hurricane/Typhoon",
                "DAMAGE_PROPERTY": "0.5B",
                "DAMAGE_CROPS": "100M",
                "INJURIES_DIRECT": "50",
                "DEATHS_DIRECT": "3",
                "MAGNITUDE": "",
            },
            {
                "EVENT_ID": "10004",
                "STATE_FIPS": "36",
                "CZ_TYPE": "Z",
                "CZ_FIPS": "72",
                "BEGIN_YEARMONTH": "202201",
                "BEGIN_DAY": "5",
                "EVENT_TYPE": "Winter Storm",
                "DAMAGE_PROPERTY": "0",
                "DAMAGE_CROPS": "",
                "INJURIES_DIRECT": "0",
                "DEATHS_DIRECT": "0",
                "MAGNITUDE": "",
            },
            {
                "EVENT_ID": "10005",
                "STATE_FIPS": "0",
                "CZ_TYPE": "M",
                "CZ_FIPS": "350",
                "BEGIN_YEARMONTH": "202207",
                "BEGIN_DAY": "10",
                "EVENT_TYPE": "Marine Thunderstorm Wind",
                "DAMAGE_PROPERTY": "",
                "DAMAGE_CROPS": "",
                "INJURIES_DIRECT": "0",
                "DEATHS_DIRECT": "0",
                "MAGNITUDE": "40",
            },
            {
                "EVENT_ID": "10006",
                "STATE_FIPS": "6",
                "CZ_TYPE": "C",
                "CZ_FIPS": "37",
                "BEGIN_YEARMONTH": "202209",
                "BEGIN_DAY": "12",
                "EVENT_TYPE": "Flash Flood",
                "DAMAGE_PROPERTY": "0",
                "DAMAGE_CROPS": "0",
                "INJURIES_DIRECT": "0",
                "DEATHS_DIRECT": "0",
                "MAGNITUDE": "",
            },
        ]

    data_rows = []
    for row in rows:
        merged = {**defaults, **row}
        data_rows.append(merged)

    df = pd.DataFrame(data_rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return gzip.compress(csv_bytes)


def _build_locations_csv(rows: list[dict] | None = None) -> bytes:
    """Build a synthetic StormEvents_locations gzipped CSV.

    Args:
        rows: List of dicts with EVENT_ID, LATITUDE, LONGITUDE.

    Returns:
        Gzipped CSV bytes.
    """
    if rows is None:
        rows = [
            {"EVENT_ID": "10001", "LATITUDE": "32.35", "LONGITUDE": "-86.28"},
            {"EVENT_ID": "10002", "LATITUDE": "29.76", "LONGITUDE": "-95.37"},
            {"EVENT_ID": "10003", "LATITUDE": "26.12", "LONGITUDE": "-80.14"},
            # 10004 (winter storm zone) — no location record
            {"EVENT_ID": "10005", "LATITUDE": "25.50", "LONGITUDE": "-90.00"},
            {"EVENT_ID": "10006", "LATITUDE": "37.77", "LONGITUDE": "-122.42"},
        ]

    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return gzip.compress(csv_bytes)


# Prebuilt fixture data
SAMPLE_DETAILS = _build_details_csv()
SAMPLE_LOCATIONS = _build_locations_csv()


def _mock_download_for_year(
    details_bytes: bytes,
    locations_bytes: bytes | None,
):
    """Return a side_effect function for _download_csv.

    Routes calls to the correct fixture based on filename pattern.
    """
    def _side_effect(filename: str) -> bytes:
        if "details" in filename:
            return details_bytes
        if "locations" in filename:
            if locations_bytes is None:
                raise httpx.HTTPStatusError(
                    "404 Not Found",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                )
            return locations_bytes
        raise httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

    return _side_effect


@pytest.fixture
def ingester():
    """Return a fresh NCEIStormsIngester instance."""
    return NCEIStormsIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


def _fetch_with_mocks(ingester, details_bytes=SAMPLE_DETAILS,
                       locations_bytes=SAMPLE_LOCATIONS, years=None):
    """Run fetch() with mocked directory listing and CSV downloads."""
    if years is None:
        years = [2022]
    with (
        patch.object(ingester, "_get_file_listing", return_value=LISTING_HTML),
        patch.object(
            ingester, "_download_csv",
            side_effect=_mock_download_for_year(details_bytes, locations_bytes),
        ),
    ):
        return ingester.fetch(years=years)


# ---------------------------------------------------------------------------
# Test: Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has all expected columns with correct dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required columns."""
        df = _fetch_with_mocks(ingester)
        expected = {
            "event_id", "fips", "date", "event_type",
            "property_damage", "crop_damage",
            "injuries_direct", "deaths_direct",
            "magnitude", "begin_lat", "begin_lon",
        }
        assert set(df.columns) == expected

    def test_event_id_is_string(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_string_dtype(df["event_id"])

    def test_fips_is_string(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_string_dtype(df["fips"])

    def test_event_type_is_string(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_string_dtype(df["event_type"])

    def test_property_damage_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_float_dtype(df["property_damage"])

    def test_crop_damage_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_float_dtype(df["crop_damage"])

    def test_injuries_direct_is_int(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_integer_dtype(df["injuries_direct"])

    def test_deaths_direct_is_int(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_integer_dtype(df["deaths_direct"])

    def test_magnitude_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_float_dtype(df["magnitude"])

    def test_begin_lat_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_float_dtype(df["begin_lat"])

    def test_begin_lon_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_with_mocks(ingester)
        assert pd.api.types.is_float_dtype(df["begin_lon"])


# ---------------------------------------------------------------------------
# Test: Event type preservation
# ---------------------------------------------------------------------------

class TestEventTypePreservation:
    """Verify raw event_type strings are passed through EXACTLY as provided."""

    def test_mixed_case_preserved(self, ingester, tmp_raw_dir):
        """Event types with mixed case are NOT lowercased or normalized."""
        df = _fetch_with_mocks(ingester)
        event_types = set(df["event_type"].values)
        assert "Thunderstorm Wind" in event_types
        assert "Tornado" in event_types

    def test_slash_in_event_type_preserved(self, ingester, tmp_raw_dir):
        """Event types with slashes like 'Hurricane/Typhoon' are preserved."""
        df = _fetch_with_mocks(ingester)
        assert "Hurricane/Typhoon" in df["event_type"].values

    def test_flash_flood_preserved(self, ingester, tmp_raw_dir):
        """Multi-word event type 'Flash Flood' is preserved."""
        df = _fetch_with_mocks(ingester)
        assert "Flash Flood" in df["event_type"].values

    def test_not_lowercased(self, ingester, tmp_raw_dir):
        """Event types are NOT lowercased."""
        df = _fetch_with_mocks(ingester)
        for et in df["event_type"]:
            assert et != et.lower() or et == et, "Should not be forcibly lowercased"
            # More directly: no known multi-word type should be all lowercase
        assert "thunderstorm wind" not in df["event_type"].values
        assert "tornado" not in df["event_type"].values

    def test_not_snake_cased(self, ingester, tmp_raw_dir):
        """Event types are NOT converted to snake_case."""
        df = _fetch_with_mocks(ingester)
        for et in df["event_type"]:
            assert "_" not in et or " " in et, f"Possible snake_case: {et}"


# ---------------------------------------------------------------------------
# Test: FIPS normalization
# ---------------------------------------------------------------------------

class TestFIPSNormalization:
    """Verify FIPS code handling for county, zone, and marine types."""

    def test_county_type_fips(self, ingester, tmp_raw_dir):
        """CZ_TYPE=C produces correct 5-digit zero-padded FIPS."""
        df = _fetch_with_mocks(ingester)
        # EVENT_ID 10001: state 1, county 1 → "01001"
        row = df[df["event_id"] == "10001"].iloc[0]
        assert row["fips"] == "01001"

    def test_county_type_fips_padding(self, ingester, tmp_raw_dir):
        """State and county FIPS are zero-padded correctly."""
        df = _fetch_with_mocks(ingester)
        # EVENT_ID 10002: state 48, county 201 → "48201"
        row = df[df["event_id"] == "10002"].iloc[0]
        assert row["fips"] == "48201"

    def test_county_type_fips_three_digit_county(self, ingester, tmp_raw_dir):
        """County FIPS < 100 is zero-padded to 3 digits."""
        df = _fetch_with_mocks(ingester)
        # EVENT_ID 10003: state 12, county 86 → "12086"
        row = df[df["event_id"] == "10003"].iloc[0]
        assert row["fips"] == "12086"

    def test_zone_type_fips_is_nan(self, ingester, tmp_raw_dir):
        """CZ_TYPE=Z records have NaN FIPS (zone-to-county mapping is transform work)."""
        df = _fetch_with_mocks(ingester)
        row = df[df["event_id"] == "10004"].iloc[0]
        assert pd.isna(row["fips"])

    def test_marine_type_fips_is_nan(self, ingester, tmp_raw_dir):
        """CZ_TYPE=M records have NaN FIPS (no county mapping for marine zones)."""
        df = _fetch_with_mocks(ingester)
        row = df[df["event_id"] == "10005"].iloc[0]
        assert pd.isna(row["fips"])

    def test_marine_records_not_dropped(self, ingester, tmp_raw_dir):
        """Marine zone records are retained, not filtered out."""
        df = _fetch_with_mocks(ingester)
        assert "10005" in df["event_id"].values


# ---------------------------------------------------------------------------
# Test: Damage string parsing
# ---------------------------------------------------------------------------

class TestDamageParsing:
    """Verify parsing of NCEI shorthand damage values."""

    def test_parse_25k(self, ingester):
        assert ingester._parse_damage("25K") == 25000.0

    def test_parse_1_5m(self, ingester):
        assert ingester._parse_damage("1.5M") == 1500000.0

    def test_parse_0_5b(self, ingester):
        assert ingester._parse_damage("0.5B") == 500000000.0

    def test_parse_zero(self, ingester):
        assert ingester._parse_damage("0") == 0.0

    def test_parse_empty_string(self, ingester):
        assert ingester._parse_damage("") == 0.0

    def test_parse_nan(self, ingester):
        assert ingester._parse_damage(float("nan")) == 0.0

    def test_parse_none(self, ingester):
        assert ingester._parse_damage(None) == 0.0

    def test_parse_lowercase_k(self, ingester):
        assert ingester._parse_damage("25k") == 25000.0

    def test_parse_lowercase_m(self, ingester):
        assert ingester._parse_damage("1.5m") == 1500000.0

    def test_parse_lowercase_b(self, ingester):
        assert ingester._parse_damage("0.5b") == 500000000.0

    def test_damage_values_in_output(self, ingester, tmp_raw_dir):
        """Verify damage parsing in full fetch context."""
        df = _fetch_with_mocks(ingester)
        # EVENT_ID 10001: "25K" property, "5K" crop
        row = df[df["event_id"] == "10001"].iloc[0]
        assert row["property_damage"] == 25000.0
        assert row["crop_damage"] == 5000.0

        # EVENT_ID 10002: "1.5M" property, "0.5M" crop
        row = df[df["event_id"] == "10002"].iloc[0]
        assert row["property_damage"] == 1500000.0
        assert row["crop_damage"] == 500000.0

        # EVENT_ID 10003: "0.5B" property, "100M" crop
        row = df[df["event_id"] == "10003"].iloc[0]
        assert row["property_damage"] == 500000000.0
        assert row["crop_damage"] == 100000000.0


# ---------------------------------------------------------------------------
# Test: Zero-damage preservation
# ---------------------------------------------------------------------------

class TestZeroDamagePreservation:
    """Verify records with $0 damage are preserved, not filtered out."""

    def test_zero_damage_records_retained(self, ingester, tmp_raw_dir):
        """Records with 0 or empty damage values are kept in output."""
        df = _fetch_with_mocks(ingester)
        # EVENT_ID 10006: "0" property, "0" crop — still in output
        assert "10006" in df["event_id"].values

    def test_zero_damage_value_is_zero(self, ingester, tmp_raw_dir):
        """Zero damage is stored as 0.0, not NaN."""
        df = _fetch_with_mocks(ingester)
        row = df[df["event_id"] == "10006"].iloc[0]
        assert row["property_damage"] == 0.0
        assert row["crop_damage"] == 0.0


# ---------------------------------------------------------------------------
# Test: Location join
# ---------------------------------------------------------------------------

class TestLocationJoin:
    """Verify lat/lon from locations file is correctly joined to details."""

    def test_coordinates_joined(self, ingester, tmp_raw_dir):
        """Events with matching location records have coordinates."""
        df = _fetch_with_mocks(ingester)
        row = df[df["event_id"] == "10001"].iloc[0]
        assert abs(row["begin_lat"] - 32.35) < 0.01
        assert abs(row["begin_lon"] - (-86.28)) < 0.01

    def test_missing_location_gives_nan(self, ingester, tmp_raw_dir):
        """Events without a matching location record get NaN coordinates."""
        df = _fetch_with_mocks(ingester)
        # EVENT_ID 10004 has no location record
        row = df[df["event_id"] == "10004"].iloc[0]
        assert pd.isna(row["begin_lat"])
        assert pd.isna(row["begin_lon"])

    def test_no_locations_file(self, ingester, tmp_raw_dir):
        """When locations file is unavailable, all coordinates are NaN."""
        df = _fetch_with_mocks(ingester, locations_bytes=None)
        assert df["begin_lat"].isna().all()
        assert df["begin_lon"].isna().all()


# ---------------------------------------------------------------------------
# Test: Ingest purity check
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics in output."""

    def test_no_derived_columns(self, ingester, tmp_raw_dir):
        """Output must not contain severity, tier, or aggregation columns."""
        df = _fetch_with_mocks(ingester)
        forbidden = {
            "severity_score", "tier", "county_event_count",
            "annual_damage_total", "severity_tier", "event_count",
            "damage_total", "score", "percentile",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        df = _fetch_with_mocks(ingester)
        expected = set(ingester.required_columns)
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "event_id": ["10001"],
            "fips": ["01001"],
            "date": [date(2022, 1, 15)],
            "event_type": ["Tornado"],
            "property_damage": [25000.0],
            "crop_damage": [0.0],
            "injuries_direct": [0],
            "deaths_direct": [0],
            "magnitude": [2.0],
            "begin_lat": [32.35],
            "begin_lon": [-86.28],
            "severity_tier": [3],  # FORBIDDEN derived column
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir):
        """A _metadata.json file is written next to each year's parquet."""
        _fetch_with_mocks(ingester)
        assert (tmp_raw_dir / "ncei_storms" / "ncei_storms_2022_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution."""
        _fetch_with_mocks(ingester)
        meta_path = tmp_raw_dir / "ncei_storms" / "ncei_storms_2022_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "NCEI_STORMS"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_combined_metadata_written(self, ingester, tmp_raw_dir):
        """The combined ncei_storms_all file gets a metadata sidecar."""
        _fetch_with_mocks(ingester)
        assert (tmp_raw_dir / "ncei_storms" / "ncei_storms_all.parquet").exists()
        assert (tmp_raw_dir / "ncei_storms" / "ncei_storms_all_metadata.json").exists()


# ---------------------------------------------------------------------------
# Test: Multi-year download & partial failure
# ---------------------------------------------------------------------------

class TestMultiYearDownload:
    """Verify multi-year fetch and partial failure handling."""

    def test_fetches_multiple_years(self, ingester, tmp_raw_dir):
        """Both 2022 and 2023 data are fetched and combined."""
        df = _fetch_with_mocks(ingester, years=[2022, 2023])
        # Both years share the same fixture data, so we get 2x rows
        assert len(df) == 12  # 6 events × 2 years

    def test_partial_failure_still_returns_data(self, ingester, tmp_raw_dir):
        """If one year fails, the other years still succeed."""
        call_count = 0

        def _download_with_failure(filename: str) -> bytes:
            nonlocal call_count
            call_count += 1
            # Fail all 2023 downloads
            if "d2023" in filename:
                raise httpx.HTTPStatusError(
                    "500 Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            if "details" in filename:
                return SAMPLE_DETAILS
            if "locations" in filename:
                return SAMPLE_LOCATIONS
            raise httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=MagicMock(status_code=404),
            )

        with (
            patch.object(ingester, "_get_file_listing", return_value=LISTING_HTML),
            patch.object(ingester, "_download_csv", side_effect=_download_with_failure),
        ):
            df = ingester.fetch(years=[2022, 2023])

        # Only 2022 data should be present
        assert len(df) == 6

    def test_missing_year_skipped(self, ingester, tmp_raw_dir):
        """A year with no file in the listing is skipped gracefully."""
        with (
            patch.object(ingester, "_get_file_listing", return_value=LISTING_HTML),
            patch.object(
                ingester, "_download_csv",
                side_effect=_mock_download_for_year(SAMPLE_DETAILS, SAMPLE_LOCATIONS),
            ),
        ):
            df = ingester.fetch(years=[2019])  # Not in listing

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic triggers on HTTP 500/503."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.text = LISTING_HTML
        ok_resp.raise_for_status = MagicMock()

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
            resp = ingester.api_get(NCEI_STORMS_BASE_URL)

        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "ncei_storms"

    def test_confidence(self, ingester):
        assert ingester.confidence == "B"

    def test_attribution(self, ingester):
        assert ingester.attribution == "proxy"

    def test_calls_per_second(self, ingester):
        """Rate limit is polite (<=1 req/sec for static downloads)."""
        assert ingester.calls_per_second <= 1.0


# ---------------------------------------------------------------------------
# Test: Date parsing
# ---------------------------------------------------------------------------

class TestDateParsing:
    """Verify date construction from BEGIN_YEARMONTH + BEGIN_DAY."""

    def test_date_correct(self, ingester, tmp_raw_dir):
        """Dates are constructed correctly from YEARMONTH + DAY."""
        df = _fetch_with_mocks(ingester)
        row = df[df["event_id"] == "10001"].iloc[0]
        assert row["date"] == date(2022, 1, 15)

    def test_date_different_month(self, ingester, tmp_raw_dir):
        """March date parsed correctly."""
        df = _fetch_with_mocks(ingester)
        row = df[df["event_id"] == "10002"].iloc[0]
        assert row["date"] == date(2022, 3, 20)


# ---------------------------------------------------------------------------
# Test: File discovery
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    """Verify directory listing parsing for filename discovery."""

    def test_finds_details_file(self, ingester):
        details, _ = ingester._find_files_for_year(LISTING_HTML, 2022)
        assert details == "StormEvents_details-ftp_v1.0_d2022_c20240101.csv.gz"

    def test_finds_locations_file(self, ingester):
        _, locations = ingester._find_files_for_year(LISTING_HTML, 2022)
        assert locations == "StormEvents_locations-ftp_v1.0_d2022_c20240101.csv.gz"

    def test_missing_year_returns_none(self, ingester):
        details, locations = ingester._find_files_for_year(LISTING_HTML, 2019)
        assert details is None
        assert locations is None

    def test_latest_compile_date_preferred(self, ingester):
        """When multiple files exist for same year, latest compile date wins."""
        html = """
        <a href="StormEvents_details-ftp_v1.0_d2022_c20230101.csv.gz">old</a>
        <a href="StormEvents_details-ftp_v1.0_d2022_c20240601.csv.gz">new</a>
        """
        details, _ = ingester._find_files_for_year(html, 2022)
        assert details == "StormEvents_details-ftp_v1.0_d2022_c20240601.csv.gz"
