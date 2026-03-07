"""Tests for the EPA Air Quality System ingester (ingest/epa_airnow.py).

All HTTP calls are mocked — no real requests to epa.gov.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from ingest.epa_airnow import (
    EPA_AQS_BULK_URL,
    EPA_PM25_PARAM_CODE,
    EPAAirNowIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _build_epa_csv(rows: list[dict] | None = None) -> str:
    """Build a synthetic EPA daily_88101 CSV string.

    Mimics the EPA pre-generated annual summary file format.

    Args:
        rows: List of dicts with column overrides. Defaults to a small
              set of representative monitor readings.

    Returns:
        CSV string content.
    """
    if rows is None:
        rows = [
            # California monitor — two readings, same monitor different days
            {
                "State Code": "06", "County Code": "037", "Site Num": "0002",
                "Parameter Code": "88101", "POC": "1",
                "Latitude": "34.0667", "Longitude": "-118.2275",
                "Datum": "WGS84", "Parameter Name": "PM2.5 - Local Conditions",
                "Sample Duration": "24 HOUR",
                "Pollutant Standard": "PM25 24-hour 2024",
                "Date Local": "2022-01-15",
                "Units of Measure": "Micrograms/cubic meter (LC)",
                "Event Type": "None", "Observation Count": "24",
                "Observation Percent": "100",
                "Arithmetic Mean": "12.3", "1st Max Value": "15.6",
                "1st Max Hour": "14", "AQI": "52",
                "Method Code": "170", "Method Name": "TEOM",
                "Local Site Name": "Los Angeles - North Main",
                "Address": "1630 N MAIN ST",
                "State Name": "California", "County Name": "Los Angeles",
                "City Name": "Los Angeles", "CBSA Name": "Los Angeles",
                "Date of Last Change": "2022-06-15",
            },
            {
                "State Code": "06", "County Code": "037", "Site Num": "0002",
                "Parameter Code": "88101", "POC": "1",
                "Latitude": "34.0667", "Longitude": "-118.2275",
                "Datum": "WGS84", "Parameter Name": "PM2.5 - Local Conditions",
                "Sample Duration": "24 HOUR",
                "Pollutant Standard": "PM25 24-hour 2024",
                "Date Local": "2022-01-16",
                "Units of Measure": "Micrograms/cubic meter (LC)",
                "Event Type": "None", "Observation Count": "24",
                "Observation Percent": "100",
                "Arithmetic Mean": "8.7", "1st Max Value": "10.2",
                "1st Max Hour": "8", "AQI": "36",
                "Method Code": "170", "Method Name": "TEOM",
                "Local Site Name": "Los Angeles - North Main",
                "Address": "1630 N MAIN ST",
                "State Name": "California", "County Name": "Los Angeles",
                "City Name": "Los Angeles", "CBSA Name": "Los Angeles",
                "Date of Last Change": "2022-06-15",
            },
            # Same site, different POC (co-located instrument)
            {
                "State Code": "06", "County Code": "037", "Site Num": "0002",
                "Parameter Code": "88101", "POC": "3",
                "Latitude": "34.0667", "Longitude": "-118.2275",
                "Datum": "WGS84", "Parameter Name": "PM2.5 - Local Conditions",
                "Sample Duration": "24 HOUR",
                "Pollutant Standard": "PM25 24-hour 2024",
                "Date Local": "2022-01-15",
                "Units of Measure": "Micrograms/cubic meter (LC)",
                "Event Type": "None", "Observation Count": "24",
                "Observation Percent": "100",
                "Arithmetic Mean": "11.9", "1st Max Value": "14.8",
                "1st Max Hour": "14", "AQI": "50",
                "Method Code": "209", "Method Name": "BAM",
                "Local Site Name": "Los Angeles - North Main",
                "Address": "1630 N MAIN ST",
                "State Name": "California", "County Name": "Los Angeles",
                "City Name": "Los Angeles", "CBSA Name": "Los Angeles",
                "Date of Last Change": "2022-06-15",
            },
            # Alabama monitor — different state for FIPS testing
            {
                "State Code": "01", "County Code": "073", "Site Num": "0023",
                "Parameter Code": "88101", "POC": "1",
                "Latitude": "33.5531", "Longitude": "-86.8150",
                "Datum": "WGS84", "Parameter Name": "PM2.5 - Local Conditions",
                "Sample Duration": "24 HOUR",
                "Pollutant Standard": "PM25 24-hour 2024",
                "Date Local": "2022-01-15",
                "Units of Measure": "Micrograms/cubic meter (LC)",
                "Event Type": "None", "Observation Count": "24",
                "Observation Percent": "100",
                "Arithmetic Mean": "9.5", "1st Max Value": "12.1",
                "1st Max Hour": "10", "AQI": "40",
                "Method Code": "170", "Method Name": "TEOM",
                "Local Site Name": "Birmingham - NCORE",
                "Address": "1200 REV ABRAHAM WOODS JR BLVD",
                "State Name": "Alabama", "County Name": "Jefferson",
                "City Name": "Birmingham", "CBSA Name": "Birmingham",
                "Date of Last Change": "2022-06-15",
            },
            # Texas monitor — reading with no AQI
            {
                "State Code": "48", "County Code": "201", "Site Num": "0024",
                "Parameter Code": "88101", "POC": "1",
                "Latitude": "29.7604", "Longitude": "-95.3698",
                "Datum": "WGS84", "Parameter Name": "PM2.5 - Local Conditions",
                "Sample Duration": "24 HOUR",
                "Pollutant Standard": "PM25 24-hour 2024",
                "Date Local": "2022-02-01",
                "Units of Measure": "Micrograms/cubic meter (LC)",
                "Event Type": "None", "Observation Count": "18",
                "Observation Percent": "75",
                "Arithmetic Mean": "7.2", "1st Max Value": "9.5",
                "1st Max Hour": "7", "AQI": "",
                "Method Code": "170", "Method Name": "TEOM",
                "Local Site Name": "Houston Deer Park",
                "Address": "4514 1/2 DURANT ST",
                "State Name": "Texas", "County Name": "Harris",
                "City Name": "Houston", "CBSA Name": "Houston",
                "Date of Last Change": "2022-06-15",
            },
        ]

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def _build_zip(csv_content: str, filename: str = "daily_88101_2022.csv") -> bytes:
    """Wrap CSV content into a zip archive.

    Args:
        csv_content: CSV string to include.
        filename: Name of the CSV file inside the zip.

    Returns:
        Zip file bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, csv_content)
    return buf.getvalue()


# Prebuilt fixture data
SAMPLE_CSV = _build_epa_csv()
SAMPLE_ZIP = _build_zip(SAMPLE_CSV)


def _mock_download_single(zip_bytes: bytes):
    """Return a side_effect for _download_annual_zip that serves one year.

    First call returns the fixture; subsequent calls raise 404.
    """
    call_count = 0

    def _side_effect(year: int) -> bytes:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return zip_bytes
        raise httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

    return _side_effect


def _mock_download_always(zip_bytes: bytes):
    """Return a side_effect for _download_annual_zip that always succeeds."""
    def _side_effect(year: int) -> bytes:
        return zip_bytes
    return _side_effect


@pytest.fixture
def ingester():
    """Return a fresh EPAAirNowIngester instance."""
    return EPAAirNowIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


def _fetch_single_year(ingester, zip_bytes=SAMPLE_ZIP, years=None):
    """Run fetch() with mocked download returning one year of data."""
    if years is None:
        years = [2022]
    with patch.object(
        ingester, "_download_annual_zip",
        side_effect=_mock_download_single(zip_bytes),
    ):
        return ingester.fetch(years=years)


# ---------------------------------------------------------------------------
# Test: Output schema — readings
# ---------------------------------------------------------------------------

class TestReadingsSchema:
    """Verify readings DataFrame has all expected columns with correct dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required readings columns."""
        df = _fetch_single_year(ingester)
        expected = {"monitor_id", "fips", "date", "pm25_value", "aqi_value", "lat", "lon"}
        assert set(df.columns) == expected

    def test_monitor_id_is_string(self, ingester, tmp_raw_dir):
        df = _fetch_single_year(ingester)
        assert pd.api.types.is_string_dtype(df["monitor_id"])

    def test_fips_is_string(self, ingester, tmp_raw_dir):
        df = _fetch_single_year(ingester)
        assert pd.api.types.is_string_dtype(df["fips"])

    def test_pm25_value_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_single_year(ingester)
        assert pd.api.types.is_float_dtype(df["pm25_value"])

    def test_aqi_value_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_single_year(ingester)
        assert pd.api.types.is_float_dtype(df["aqi_value"])

    def test_lat_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_single_year(ingester)
        assert pd.api.types.is_float_dtype(df["lat"])

    def test_lon_is_float(self, ingester, tmp_raw_dir):
        df = _fetch_single_year(ingester)
        assert pd.api.types.is_float_dtype(df["lon"])


# ---------------------------------------------------------------------------
# Test: Output schema — metadata
# ---------------------------------------------------------------------------

class TestMetadataSchema:
    """Verify monitor metadata DataFrame has correct columns."""

    def test_metadata_columns(self, ingester, tmp_raw_dir):
        """Monitor metadata has expected columns."""
        _fetch_single_year(ingester)
        meta = ingester.fetch_monitor_metadata()
        expected = {"monitor_id", "lat", "lon", "county_fips", "state"}
        assert set(meta.columns) == expected

    def test_metadata_monitor_id_is_string(self, ingester, tmp_raw_dir):
        _fetch_single_year(ingester)
        meta = ingester.fetch_monitor_metadata()
        assert pd.api.types.is_string_dtype(meta["monitor_id"])

    def test_metadata_county_fips_is_string(self, ingester, tmp_raw_dir):
        _fetch_single_year(ingester)
        meta = ingester.fetch_monitor_metadata()
        assert pd.api.types.is_string_dtype(meta["county_fips"])

    def test_metadata_state_is_string(self, ingester, tmp_raw_dir):
        _fetch_single_year(ingester)
        meta = ingester.fetch_monitor_metadata()
        assert pd.api.types.is_string_dtype(meta["state"])


# ---------------------------------------------------------------------------
# Test: Monitor ID construction
# ---------------------------------------------------------------------------

class TestMonitorIDConstruction:
    """Verify monitor_id built from state + county + site + POC."""

    def test_monitor_id_format(self, ingester, tmp_raw_dir):
        """Monitor ID follows state-county-site-poc format."""
        df = _fetch_single_year(ingester)
        # California monitor: state 06, county 037, site 0002, POC 1
        assert "06-037-0002-1" in df["monitor_id"].values

    def test_alabama_monitor_id(self, ingester, tmp_raw_dir):
        """Alabama monitor: state 01, county 073, site 0023, POC 1."""
        df = _fetch_single_year(ingester)
        assert "01-073-0023-1" in df["monitor_id"].values

    def test_texas_monitor_id(self, ingester, tmp_raw_dir):
        """Texas monitor: state 48, county 201, site 0024, POC 1."""
        df = _fetch_single_year(ingester)
        assert "48-201-0024-1" in df["monitor_id"].values


# ---------------------------------------------------------------------------
# Test: FIPS construction
# ---------------------------------------------------------------------------

class TestFIPSConstruction:
    """Verify 5-digit county FIPS from state + county codes."""

    def test_california_fips(self, ingester, tmp_raw_dir):
        """State 06 + County 037 → '06037'."""
        df = _fetch_single_year(ingester)
        ca_rows = df[df["monitor_id"].str.startswith("06-037")]
        assert (ca_rows["fips"] == "06037").all()

    def test_alabama_fips(self, ingester, tmp_raw_dir):
        """State 01 + County 073 → '01073'."""
        df = _fetch_single_year(ingester)
        al_rows = df[df["monitor_id"].str.startswith("01-073")]
        assert (al_rows["fips"] == "01073").all()

    def test_texas_fips(self, ingester, tmp_raw_dir):
        """State 48 + County 201 → '48201'."""
        df = _fetch_single_year(ingester)
        tx_rows = df[df["monitor_id"].str.startswith("48-201")]
        assert (tx_rows["fips"] == "48201").all()

    def test_all_fips_5_digit(self, ingester, tmp_raw_dir):
        """All FIPS codes are exactly 5 characters."""
        df = _fetch_single_year(ingester)
        assert (df["fips"].str.len() == 5).all()


# ---------------------------------------------------------------------------
# Test: Multiple POC handling
# ---------------------------------------------------------------------------

class TestMultiplePOC:
    """Verify co-located monitors (same site, different POC) are retained."""

    def test_different_poc_produces_different_monitor_ids(self, ingester, tmp_raw_dir):
        """Same site with POC 1 and POC 3 produces two distinct monitor_ids."""
        df = _fetch_single_year(ingester)
        ca_monitors = df[df["fips"] == "06037"]["monitor_id"].unique()
        assert "06-037-0002-1" in ca_monitors
        assert "06-037-0002-3" in ca_monitors

    def test_both_poc_readings_preserved(self, ingester, tmp_raw_dir):
        """Both POC instruments' readings are in the output."""
        df = _fetch_single_year(ingester)
        poc1_jan15 = df[
            (df["monitor_id"] == "06-037-0002-1")
            & (df["date"] == date(2022, 1, 15))
        ]
        poc3_jan15 = df[
            (df["monitor_id"] == "06-037-0002-3")
            & (df["date"] == date(2022, 1, 15))
        ]
        assert len(poc1_jan15) == 1
        assert len(poc3_jan15) == 1
        # Different readings from different instruments
        assert abs(poc1_jan15.iloc[0]["pm25_value"] - 12.3) < 0.01
        assert abs(poc3_jan15.iloc[0]["pm25_value"] - 11.9) < 0.01

    def test_metadata_has_both_monitors(self, ingester, tmp_raw_dir):
        """Monitor metadata includes both POC instruments."""
        _fetch_single_year(ingester)
        meta = ingester.fetch_monitor_metadata()
        assert "06-037-0002-1" in meta["monitor_id"].values
        assert "06-037-0002-3" in meta["monitor_id"].values


# ---------------------------------------------------------------------------
# Test: Ingest purity check
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics in output."""

    def test_no_derived_columns(self, ingester, tmp_raw_dir):
        """Output must not contain county averages, smoke days, etc."""
        df = _fetch_single_year(ingester)
        forbidden = {
            "county_avg_pm25", "annual_aqi", "smoke_day", "rolling_mean",
            "county_mean", "exceedance_count", "score", "percentile",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        df = _fetch_single_year(ingester)
        expected = set(ingester.required_columns)
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "monitor_id": ["06-037-0002-1"],
            "fips": ["06037"],
            "date": [date(2022, 1, 15)],
            "pm25_value": [12.3],
            "aqi_value": [52.0],
            "lat": [34.0667],
            "lon": [-118.2275],
            "county_avg_pm25": [12.3],  # FORBIDDEN derived column
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_readings_metadata_created(self, ingester, tmp_raw_dir):
        """Readings parquet + metadata JSON exist per year."""
        _fetch_single_year(ingester)
        assert (tmp_raw_dir / "epa_airnow" / "epa_aqs_readings_2022.parquet").exists()
        assert (tmp_raw_dir / "epa_airnow" / "epa_aqs_readings_2022_metadata.json").exists()

    def test_monitors_metadata_created(self, ingester, tmp_raw_dir):
        """Monitor metadata parquet + JSON exist per year."""
        _fetch_single_year(ingester)
        assert (tmp_raw_dir / "epa_airnow" / "epa_aqs_monitors_2022.parquet").exists()
        assert (tmp_raw_dir / "epa_airnow" / "epa_aqs_monitors_2022_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution."""
        _fetch_single_year(ingester)
        meta_path = tmp_raw_dir / "epa_airnow" / "epa_aqs_readings_2022_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "EPA_AIRNOW"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_combined_files_written(self, ingester, tmp_raw_dir):
        """Combined readings_all and monitors_all files exist."""
        _fetch_single_year(ingester)
        assert (tmp_raw_dir / "epa_airnow" / "epa_aqs_readings_all.parquet").exists()
        assert (tmp_raw_dir / "epa_airnow" / "epa_aqs_monitors_all.parquet").exists()


# ---------------------------------------------------------------------------
# Test: Multi-year download & partial failure
# ---------------------------------------------------------------------------

class TestMultiYearDownload:
    """Verify multi-year fetch and partial failure handling."""

    def test_fetches_multiple_years(self, ingester, tmp_raw_dir):
        """Both years fetched and combined."""
        with patch.object(
            ingester, "_download_annual_zip",
            side_effect=_mock_download_always(SAMPLE_ZIP),
        ):
            df = ingester.fetch(years=[2022, 2023])

        # 5 rows per year × 2 years = 10
        assert len(df) == 10

    def test_partial_failure_returns_successful_years(self, ingester, tmp_raw_dir):
        """If one year fails, other years still succeed."""
        call_years = []

        def _download_with_failure(year: int) -> bytes:
            call_years.append(year)
            if year == 2023:
                raise httpx.HTTPStatusError(
                    "500 Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            return SAMPLE_ZIP

        with patch.object(
            ingester, "_download_annual_zip",
            side_effect=_download_with_failure,
        ):
            df = ingester.fetch(years=[2022, 2023])

        assert len(df) == 5  # Only 2022 data
        assert 2022 in call_years
        assert 2023 in call_years

    def test_all_fail_returns_empty(self, ingester, tmp_raw_dir):
        """If all years fail, return empty DataFrame with correct columns."""
        def _download_fail(year: int) -> bytes:
            raise httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )

        with patch.object(
            ingester, "_download_annual_zip",
            side_effect=_download_fail,
        ):
            df = ingester.fetch(years=[2022])

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
        ok_resp.content = SAMPLE_ZIP
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
            url = f"{EPA_AQS_BULK_URL}/daily_{EPA_PM25_PARAM_CODE}_2022.zip"
            resp = ingester.api_get(url)

        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "epa_airnow"

    def test_confidence(self, ingester):
        assert ingester.confidence == "A"

    def test_attribution(self, ingester):
        assert ingester.attribution == "proxy"

    def test_calls_per_second(self, ingester):
        """Rate limit is polite (<=1 req/sec for bulk downloads)."""
        assert ingester.calls_per_second <= 1.0


# ---------------------------------------------------------------------------
# Test: Data values
# ---------------------------------------------------------------------------

class TestDataValues:
    """Verify parsed data values are correct."""

    def test_pm25_value_correct(self, ingester, tmp_raw_dir):
        """PM2.5 arithmetic mean parsed correctly."""
        df = _fetch_single_year(ingester)
        row = df[
            (df["monitor_id"] == "06-037-0002-1")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["pm25_value"] - 12.3) < 0.01

    def test_aqi_value_correct(self, ingester, tmp_raw_dir):
        """AQI value parsed correctly."""
        df = _fetch_single_year(ingester)
        row = df[
            (df["monitor_id"] == "06-037-0002-1")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["aqi_value"] - 52.0) < 0.01

    def test_missing_aqi_is_nan(self, ingester, tmp_raw_dir):
        """Empty AQI becomes NaN, not 0."""
        df = _fetch_single_year(ingester)
        tx_row = df[df["monitor_id"] == "48-201-0024-1"].iloc[0]
        assert pd.isna(tx_row["aqi_value"])

    def test_coordinates_correct(self, ingester, tmp_raw_dir):
        """Lat/lon coordinates parsed from CSV."""
        df = _fetch_single_year(ingester)
        row = df[
            (df["monitor_id"] == "06-037-0002-1")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["lat"] - 34.0667) < 0.001
        assert abs(row["lon"] - (-118.2275)) < 0.001

    def test_date_parsed_correctly(self, ingester, tmp_raw_dir):
        """Date Local string parsed to Python date object."""
        df = _fetch_single_year(ingester)
        row = df[
            (df["monitor_id"] == "01-073-0023-1")
        ].iloc[0]
        assert row["date"] == date(2022, 1, 15)


# ---------------------------------------------------------------------------
# Test: Row count
# ---------------------------------------------------------------------------

class TestRowCount:
    """Verify expected number of rows in output."""

    def test_all_rows_present(self, ingester, tmp_raw_dir):
        """All 5 rows from fixture appear in output."""
        df = _fetch_single_year(ingester)
        assert len(df) == 5

    def test_unique_monitor_count(self, ingester, tmp_raw_dir):
        """4 unique monitors: 2 CA (POC 1 and 3), 1 AL, 1 TX."""
        df = _fetch_single_year(ingester)
        assert df["monitor_id"].nunique() == 4

    def test_metadata_unique_monitors(self, ingester, tmp_raw_dir):
        """Monitor metadata has one row per unique monitor."""
        _fetch_single_year(ingester)
        meta = ingester.fetch_monitor_metadata()
        assert len(meta) == 4


# ---------------------------------------------------------------------------
# Test: Malformed zip
# ---------------------------------------------------------------------------

class TestMalformedInput:
    """Verify handling of malformed zip files."""

    def test_empty_zip_raises(self, ingester):
        """Zip with no CSV raises ValueError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass  # Empty zip
        empty_zip = buf.getvalue()

        with pytest.raises(ValueError, match="no CSV files"):
            ingester._parse_annual_csv(empty_zip)
