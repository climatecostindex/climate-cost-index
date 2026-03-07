"""Tests for the NOAA NCEI GHCN-Daily ingester (ingest/noaa_ncei.py).

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

from ingest.noaa_ncei import (
    BY_YEAR_COLUMNS,
    GHCN_BY_YEAR_URL,
    GHCN_CALLS_PER_SECOND,
    GHCN_ELEMENTS,
    GHCN_MISSING_VALUE,
    GHCN_STATIONS_URL,
    NORMALS_ACCESS_URL,
    NORMALS_TMAX_COL,
    NORMALS_TMIN_COL,
    STATIONS_COLSPECS,
    NOAANCEIIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Synthetic station metadata in GHCN fixed-width format
# Cols: station_id(0:11) lat(12:20) lon(21:30) elev(31:37) state(38:40) name(41:71)
SAMPLE_STATIONS_TEXT = (
    "USW00094728  40.7789  -73.9692   39.6 NY NEW YORK CNTRL PK TWR     \n"
    "USC00010008  30.4767  -87.7567  100.6 AL ATMORE                    \n"
    "USR0000CACT  36.5700 -118.1300 1341.1 CA CALIFORNIA AQUEDUCT       \n"
    "CA006158350  50.2175  -99.9611  267.3    RIDING MOUNTAIN CS         \n"
    "ASN00079027 -33.7500  120.0333    3.0    CONDINGUP                  \n"
)

# Expected U.S. station IDs after filtering
US_STATION_IDS = {"USW00094728", "USC00010008", "USR0000CACT"}


def _build_year_csv_gz(rows: list[dict] | None = None) -> bytes:
    """Build a synthetic by-year CSV.gz file.

    By-year CSVs have no header. Columns are:
    station_id, date, element, value, m_flag, q_flag, s_flag, obs_time

    Args:
        rows: List of dicts with column values. Missing columns get defaults.

    Returns:
        Gzipped CSV bytes.
    """
    if rows is None:
        rows = [
            # Station 1: full obs for 2022-01-15
            {"station_id": "USW00094728", "date_str": "20220115",
             "element": "TMAX", "value": "235", "q_flag": ""},
            {"station_id": "USW00094728", "date_str": "20220115",
             "element": "TMIN", "value": "-50", "q_flag": ""},
            {"station_id": "USW00094728", "date_str": "20220115",
             "element": "PRCP", "value": "152", "q_flag": ""},
            # Station 1: obs with quality flag
            {"station_id": "USW00094728", "date_str": "20220116",
             "element": "TMAX", "value": "300", "q_flag": "S"},
            {"station_id": "USW00094728", "date_str": "20220116",
             "element": "TMIN", "value": "100", "q_flag": ""},
            # Station 2: only TMAX (partial obs)
            {"station_id": "USC00010008", "date_str": "20220115",
             "element": "TMAX", "value": "310", "q_flag": ""},
            # Station 2: missing value sentinel
            {"station_id": "USC00010008", "date_str": "20220116",
             "element": "TMAX", "value": str(GHCN_MISSING_VALUE),
             "q_flag": ""},
            # Non-US station (should be filtered out)
            {"station_id": "CA006158350", "date_str": "20220115",
             "element": "TMAX", "value": "200", "q_flag": ""},
            # Non-target element (SNOW — should be filtered out)
            {"station_id": "USW00094728", "date_str": "20220115",
             "element": "SNOW", "value": "50", "q_flag": ""},
        ]

    defaults = {
        "station_id": "USW00094728",
        "date_str": "20220101",
        "element": "TMAX",
        "value": "200",
        "m_flag": "",
        "q_flag": "",
        "s_flag": "",
        "obs_time": "",
    }
    data_rows = [{**defaults, **row} for row in rows]
    df = pd.DataFrame(data_rows)
    csv_bytes = df.to_csv(index=False, header=False).encode("utf-8")
    return gzip.compress(csv_bytes)


def _build_normals_csv(
    tmax_normals: list[float] | None = None,
    tmin_normals: list[float] | None = None,
) -> str:
    """Build a synthetic normals CSV for a single station.

    Values are in °F (as provided by NCEI normals-monthly product).

    Args:
        tmax_normals: 12 monthly TMAX values in °F.
        tmin_normals: 12 monthly TMIN values in °F.

    Returns:
        CSV text content.
    """
    if tmax_normals is None:
        # Realistic °F values for NYC: Jan-Dec
        tmax_normals = [38.2, 41.3, 50.1, 62.0, 72.1, 80.5,
                        85.3, 83.6, 76.2, 64.5, 53.0, 42.3]
    if tmin_normals is None:
        tmin_normals = [25.8, 28.0, 34.5, 44.2, 54.0, 63.5,
                        68.8, 67.5, 60.1, 49.0, 39.5, 30.2]

    rows = []
    for i in range(12):
        rows.append({
            "STATION": "USW00094728",
            "DATE": f"2006-{i + 1:02d}-01",
            NORMALS_TMAX_COL: tmax_normals[i],
            NORMALS_TMIN_COL: tmin_normals[i],
        })
    return pd.DataFrame(rows).to_csv(index=False)


# Prebuilt fixtures
SAMPLE_YEAR_CSV_GZ = _build_year_csv_gz()
SAMPLE_NORMALS_CSV = _build_normals_csv()


@pytest.fixture
def ingester():
    """Return a fresh NOAANCEIIngester instance."""
    return NOAANCEIIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


def _write_gz_fixture(tmp_raw_dir: Path, gz_bytes: bytes, year: int = 2022) -> Path:
    """Write gzipped CSV fixture to the expected cache location.

    Returns:
        Path to the written .csv.gz file.
    """
    cache_dir = tmp_raw_dir / "noaa_ncei"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / f"ghcn_daily_{year}.csv.gz"
    gz_path.write_bytes(gz_bytes)
    return gz_path


def _mock_stations_response() -> MagicMock:
    """Create a mock HTTP response for station metadata."""
    resp = MagicMock(spec=httpx.Response)
    resp.text = SAMPLE_STATIONS_TEXT
    resp.status_code = 200
    return resp


def _mock_normals_response(csv_text: str = SAMPLE_NORMALS_CSV) -> MagicMock:
    """Create a mock HTTP response for normals CSV."""
    resp = MagicMock(spec=httpx.Response)
    resp.text = csv_text
    resp.status_code = 200
    return resp


def _run_fetch_with_mocks(
    ingester: NOAANCEIIngester,
    tmp_raw_dir: Path,
    gz_bytes: bytes = SAMPLE_YEAR_CSV_GZ,
    normals_csv: str = SAMPLE_NORMALS_CSV,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Run fetch() with fully mocked HTTP and file downloads."""
    if years is None:
        years = [2022]

    # Write gz fixture to expected path
    for year in years:
        _write_gz_fixture(tmp_raw_dir, gz_bytes, year=year)

    def mock_api_get(url, params=None, headers=None):
        if "ghcnd-stations" in url:
            return _mock_stations_response()
        if "normals-monthly" in url:
            return _mock_normals_response(normals_csv)
        raise httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404),
        )

    def mock_download_year(year_arg):
        return tmp_raw_dir / "noaa_ncei" / f"ghcn_daily_{year_arg}.csv.gz"

    with (
        patch.object(ingester, "api_get", side_effect=mock_api_get),
        patch.object(ingester, "_download_year_csv", side_effect=mock_download_year),
    ):
        return ingester.fetch(years=years)


# ---------------------------------------------------------------------------
# Test: Temperature conversion
# ---------------------------------------------------------------------------

class TestTemperatureConversion:
    """Verify GHCN-Daily tenths-of-°C are correctly converted."""

    def test_positive_tmax(self, ingester, tmp_raw_dir):
        """235 tenths → 23.5°C."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USW00094728")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["tmax"] - 23.5) < 0.01

    def test_negative_tmin(self, ingester, tmp_raw_dir):
        """-50 tenths → -5.0°C."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USW00094728")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["tmin"] - (-5.0)) < 0.01

    def test_another_positive_tmax(self, ingester, tmp_raw_dir):
        """310 tenths → 31.0°C."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USC00010008")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["tmax"] - 31.0) < 0.01


# ---------------------------------------------------------------------------
# Test: Precipitation conversion
# ---------------------------------------------------------------------------

class TestPrecipitationConversion:
    """Verify GHCN-Daily tenths-of-mm are correctly converted."""

    def test_prcp_conversion(self, ingester, tmp_raw_dir):
        """152 tenths → 15.2 mm."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USW00094728")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["prcp"] - 15.2) < 0.01


# ---------------------------------------------------------------------------
# Test: Missing value handling
# ---------------------------------------------------------------------------

class TestMissingValueHandling:
    """Verify -9999 sentinel is converted to NaN."""

    def test_missing_value_is_nan(self, ingester, tmp_raw_dir):
        """-9999 → NaN, not a numeric value or dropped row."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USC00010008")
            & (df["date"] == date(2022, 1, 16))
        ].iloc[0]
        assert pd.isna(row["tmax"])

    def test_missing_value_row_not_dropped(self, ingester, tmp_raw_dir):
        """Rows with -9999 values are retained, not filtered out."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        missing_rows = df[
            (df["station_id"] == "USC00010008")
            & (df["date"] == date(2022, 1, 16))
        ]
        assert len(missing_rows) == 1


# ---------------------------------------------------------------------------
# Test: Quality flags
# ---------------------------------------------------------------------------

class TestQualityFlags:
    """Verify quality flags are preserved in output."""

    def test_qflag_columns_present(self, ingester, tmp_raw_dir):
        """All three q_flag columns exist."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert "q_flag_tmax" in df.columns
        assert "q_flag_tmin" in df.columns
        assert "q_flag_prcp" in df.columns

    def test_qflag_populated(self, ingester, tmp_raw_dir):
        """Quality flag 'S' is preserved for the flagged observation."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USW00094728")
            & (df["date"] == date(2022, 1, 16))
        ].iloc[0]
        assert row["q_flag_tmax"] == "S"

    def test_qflag_empty_when_passed(self, ingester, tmp_raw_dir):
        """Quality flag is empty string when observation passed checks."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USW00094728")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert row["q_flag_tmax"] == ""
        assert row["q_flag_tmin"] == ""
        assert row["q_flag_prcp"] == ""


# ---------------------------------------------------------------------------
# Test: Element pivot (by-year CSV format)
# ---------------------------------------------------------------------------

class TestElementPivot:
    """Verify separate TMAX, TMIN, PRCP rows become a single row."""

    def test_all_elements_in_one_row(self, ingester, tmp_raw_dir):
        """Station-date with all three elements produces one row."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        rows = df[
            (df["station_id"] == "USW00094728")
            & (df["date"] == date(2022, 1, 15))
        ]
        assert len(rows) == 1
        row = rows.iloc[0]
        assert not pd.isna(row["tmax"])
        assert not pd.isna(row["tmin"])
        assert not pd.isna(row["prcp"])

    def test_non_target_elements_excluded(self, ingester, tmp_raw_dir):
        """SNOW and other non-target elements are not in output."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert "snow" not in df.columns
        assert "SNOW" not in df.columns


# ---------------------------------------------------------------------------
# Test: Partial observations
# ---------------------------------------------------------------------------

class TestPartialObservations:
    """Verify station-date with missing elements gets NaN."""

    def test_tmax_only_gives_nan_tmin(self, ingester, tmp_raw_dir):
        """Station with TMAX but no TMIN produces tmin=NaN."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USC00010008")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert abs(row["tmax"] - 31.0) < 0.01
        assert pd.isna(row["tmin"])

    def test_tmax_only_gives_nan_prcp(self, ingester, tmp_raw_dir):
        """Station with TMAX but no PRCP produces prcp=NaN."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        row = df[
            (df["station_id"] == "USC00010008")
            & (df["date"] == date(2022, 1, 15))
        ].iloc[0]
        assert pd.isna(row["prcp"])


# ---------------------------------------------------------------------------
# Test: Station metadata (fixed-width parsing)
# ---------------------------------------------------------------------------

class TestStationMetadata:
    """Verify station metadata is parsed from fixed-width format."""

    def test_parses_station_id(self, ingester):
        """Station IDs are correctly extracted."""
        df = ingester._download_stations.__wrapped__(ingester) if hasattr(
            ingester._download_stations, "__wrapped__"
        ) else None

        # Test via the parse logic directly
        resp = MagicMock()
        resp.text = SAMPLE_STATIONS_TEXT
        with patch.object(ingester, "api_get", return_value=resp):
            stations = ingester._download_stations()

        assert "USW00094728" in stations["station_id"].values
        assert "USC00010008" in stations["station_id"].values
        assert "CA006158350" in stations["station_id"].values

    def test_parses_coordinates(self, ingester):
        """Latitude and longitude are correctly parsed."""
        resp = MagicMock()
        resp.text = SAMPLE_STATIONS_TEXT
        with patch.object(ingester, "api_get", return_value=resp):
            stations = ingester._download_stations()

        nyc = stations[stations["station_id"] == "USW00094728"].iloc[0]
        assert abs(nyc["lat"] - 40.7789) < 0.001
        assert abs(nyc["lon"] - (-73.9692)) < 0.001

    def test_parses_elevation(self, ingester):
        """Elevation is correctly parsed."""
        resp = MagicMock()
        resp.text = SAMPLE_STATIONS_TEXT
        with patch.object(ingester, "api_get", return_value=resp):
            stations = ingester._download_stations()

        atmore = stations[stations["station_id"] == "USC00010008"].iloc[0]
        assert abs(atmore["elevation"] - 100.6) < 0.1

    def test_parses_state(self, ingester):
        """State abbreviation is correctly parsed."""
        resp = MagicMock()
        resp.text = SAMPLE_STATIONS_TEXT
        with patch.object(ingester, "api_get", return_value=resp):
            stations = ingester._download_stations()

        nyc = stations[stations["station_id"] == "USW00094728"].iloc[0]
        assert nyc["state"] == "NY"

    def test_parses_name(self, ingester):
        """Station name is correctly parsed."""
        resp = MagicMock()
        resp.text = SAMPLE_STATIONS_TEXT
        with patch.object(ingester, "api_get", return_value=resp):
            stations = ingester._download_stations()

        nyc = stations[stations["station_id"] == "USW00094728"].iloc[0]
        assert "NEW YORK" in nyc["name"]


# ---------------------------------------------------------------------------
# Test: U.S. station filtering
# ---------------------------------------------------------------------------

class TestUSFiltering:
    """Verify only U.S. stations are retained."""

    def test_us_stations_kept(self, ingester):
        """Stations starting with 'US' are retained."""
        resp = MagicMock()
        resp.text = SAMPLE_STATIONS_TEXT
        with patch.object(ingester, "api_get", return_value=resp):
            stations = ingester._download_stations()
            us = ingester._filter_us_stations(stations)

        us_ids = set(us["station_id"].values)
        assert "USW00094728" in us_ids
        assert "USC00010008" in us_ids
        assert "USR0000CACT" in us_ids

    def test_non_us_stations_excluded(self, ingester):
        """Non-US stations (CA, ASN) are excluded."""
        resp = MagicMock()
        resp.text = SAMPLE_STATIONS_TEXT
        with patch.object(ingester, "api_get", return_value=resp):
            stations = ingester._download_stations()
            us = ingester._filter_us_stations(stations)

        us_ids = set(us["station_id"].values)
        assert "CA006158350" not in us_ids
        assert "ASN00079027" not in us_ids

    def test_non_us_data_filtered_from_observations(self, ingester, tmp_raw_dir):
        """Non-US station data is not present in daily observations."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert "CA006158350" not in df["station_id"].values


# ---------------------------------------------------------------------------
# Test: Normals parsing
# ---------------------------------------------------------------------------

class TestNormals:
    """Verify normals table schema and units."""

    def test_normals_schema(self, ingester):
        """Normals DataFrame has correct columns."""
        normals = ingester._parse_normals_csv(SAMPLE_NORMALS_CSV, "USW00094728")
        assert normals is not None
        expected = {"station_id", "month", "normal_tmax", "normal_tmin"}
        assert set(normals.columns) == expected

    def test_normals_has_12_months(self, ingester):
        """Normals table has 12 rows (one per month)."""
        normals = ingester._parse_normals_csv(SAMPLE_NORMALS_CSV, "USW00094728")
        assert normals is not None
        assert len(normals) == 12

    def test_normals_months_1_to_12(self, ingester):
        """Month column contains values 1-12."""
        normals = ingester._parse_normals_csv(SAMPLE_NORMALS_CSV, "USW00094728")
        assert normals is not None
        assert list(normals["month"]) == list(range(1, 13))

    def test_normals_in_celsius(self, ingester):
        """Normals are converted from °F to °C.

        Test: 68.0°F → 20.0°C (exact), using a custom fixture.
        """
        csv = _build_normals_csv(
            tmax_normals=[68.0] * 12,
            tmin_normals=[32.0] * 12,
        )
        normals = ingester._parse_normals_csv(csv, "USW00094728")
        assert normals is not None
        # 68°F = 20.0°C
        assert abs(normals["normal_tmax"].iloc[0] - 20.0) < 0.01
        # 32°F = 0.0°C
        assert abs(normals["normal_tmin"].iloc[0] - 0.0) < 0.01

    def test_normals_realistic_values(self, ingester):
        """Normals for NYC should be in a plausible °C range."""
        normals = ingester._parse_normals_csv(SAMPLE_NORMALS_CSV, "USW00094728")
        assert normals is not None
        # January TMAX: 38.2°F → ~3.44°C
        jan_tmax = normals[normals["month"] == 1]["normal_tmax"].iloc[0]
        assert abs(jan_tmax - (38.2 - 32) * 5 / 9) < 0.01

    def test_normals_missing_columns_returns_none(self, ingester):
        """CSV without normals columns returns None."""
        csv = "STATION,DATE,OTHER_COL\nUSW00094728,2006-01-01,42.0\n"
        normals = ingester._parse_normals_csv(csv, "USW00094728")
        assert normals is None


# ---------------------------------------------------------------------------
# Test: Output schemas
# ---------------------------------------------------------------------------

class TestOutputSchemaObservations:
    """Verify observations DataFrame has correct columns and dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required columns."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        expected = {
            "station_id", "date", "tmax", "tmin", "prcp",
            "q_flag_tmax", "q_flag_tmin", "q_flag_prcp",
        }
        assert set(df.columns) == expected

    def test_station_id_is_string(self, ingester, tmp_raw_dir):
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert pd.api.types.is_string_dtype(df["station_id"])

    def test_tmax_is_float(self, ingester, tmp_raw_dir):
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert pd.api.types.is_float_dtype(df["tmax"])

    def test_tmin_is_float(self, ingester, tmp_raw_dir):
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert pd.api.types.is_float_dtype(df["tmin"])

    def test_prcp_is_float(self, ingester, tmp_raw_dir):
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert pd.api.types.is_float_dtype(df["prcp"])

    def test_qflag_is_string(self, ingester, tmp_raw_dir):
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        for col in ["q_flag_tmax", "q_flag_tmin", "q_flag_prcp"]:
            assert pd.api.types.is_string_dtype(df[col]), f"{col} should be string"


class TestOutputSchemaStationMetadata:
    """Verify station metadata DataFrame schema."""

    def test_metadata_columns(self, ingester, tmp_raw_dir):
        """Station metadata has all expected columns."""
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        meta = ingester.fetch_station_metadata()
        expected = {"station_id", "lat", "lon", "elevation", "state", "name"}
        assert set(meta.columns) == expected

    def test_metadata_lat_is_float(self, ingester, tmp_raw_dir):
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        meta = ingester.fetch_station_metadata()
        assert pd.api.types.is_float_dtype(meta["lat"])

    def test_metadata_station_id_is_string(self, ingester, tmp_raw_dir):
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        meta = ingester.fetch_station_metadata()
        assert pd.api.types.is_string_dtype(meta["station_id"])


class TestOutputSchemaNormals:
    """Verify normals DataFrame schema."""

    def test_normals_schema_from_fetch(self, ingester, tmp_raw_dir):
        """Normals cached during fetch() have correct columns."""
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        normals = ingester.fetch_normals()
        expected = {"station_id", "month", "normal_tmax", "normal_tmin"}
        assert set(normals.columns) == expected

    def test_normals_tmax_is_float(self, ingester, tmp_raw_dir):
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        normals = ingester.fetch_normals()
        assert pd.api.types.is_float_dtype(normals["normal_tmax"])

    def test_normals_tmin_is_float(self, ingester, tmp_raw_dir):
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        normals = ingester.fetch_normals()
        assert pd.api.types.is_float_dtype(normals["normal_tmin"])


# ---------------------------------------------------------------------------
# Test: Ingest purity check
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics in output."""

    def test_no_derived_columns(self, ingester, tmp_raw_dir):
        """Output must not contain degree days, anomalies, etc."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        forbidden = {
            "degree_days", "hdd", "cdd", "anomaly", "extreme_heat_days",
            "county_fips", "fips", "monthly_avg", "annual_avg",
            "heating_degree_days", "cooling_degree_days",
            "temperature_anomaly", "days_above_95f", "days_above_100f",
            "score", "percentile",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir)
        expected = set(ingester.required_columns)
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "station_id": ["USW00094728"],
            "date": [date(2022, 1, 15)],
            "tmax": [23.5],
            "tmin": [-5.0],
            "prcp": [15.2],
            "q_flag_tmax": [""],
            "q_flag_tmin": [""],
            "q_flag_prcp": [""],
            "degree_days": [100.0],  # FORBIDDEN
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_year_metadata_created(self, ingester, tmp_raw_dir):
        """A _metadata.json file is written for each year's parquet."""
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert (tmp_raw_dir / "noaa_ncei" / "noaa_ncei_2022_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution."""
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        meta_path = tmp_raw_dir / "noaa_ncei" / "noaa_ncei_2022_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "NOAA_NCEI"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_stations_metadata_created(self, ingester, tmp_raw_dir):
        """Station metadata parquet and JSON are written."""
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert (tmp_raw_dir / "noaa_ncei" / "noaa_ncei_stations.parquet").exists()
        assert (tmp_raw_dir / "noaa_ncei" / "noaa_ncei_stations_metadata.json").exists()

    def test_normals_metadata_created(self, ingester, tmp_raw_dir):
        """Normals parquet and JSON are written."""
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert (tmp_raw_dir / "noaa_ncei" / "noaa_ncei_normals.parquet").exists()
        assert (tmp_raw_dir / "noaa_ncei" / "noaa_ncei_normals_metadata.json").exists()

    def test_combined_observations_written(self, ingester, tmp_raw_dir):
        """Combined observations file is written."""
        _run_fetch_with_mocks(ingester, tmp_raw_dir)
        assert (tmp_raw_dir / "noaa_ncei" / "noaa_ncei_observations_all.parquet").exists()
        assert (
            tmp_raw_dir / "noaa_ncei" / "noaa_ncei_observations_all_metadata.json"
        ).exists()


# ---------------------------------------------------------------------------
# Test: Multi-year download & partial failure
# ---------------------------------------------------------------------------

class TestMultiYearDownload:
    """Verify multi-year fetch and partial failure handling."""

    def test_fetches_multiple_years(self, ingester, tmp_raw_dir):
        """Both years' data are fetched and combined."""
        df = _run_fetch_with_mocks(ingester, tmp_raw_dir, years=[2022, 2023])
        # Same fixture data for both years, so we get 2x rows
        single_year_count = len(
            _run_fetch_with_mocks(ingester, tmp_raw_dir, years=[2022])
        )
        # Re-run with both years (need fresh fixture files)
        for year in [2022, 2023]:
            _write_gz_fixture(tmp_raw_dir, SAMPLE_YEAR_CSV_GZ, year=year)

        def mock_api_get(url, params=None, headers=None):
            if "ghcnd-stations" in url:
                return _mock_stations_response()
            if "normals-monthly" in url:
                return _mock_normals_response()
            raise httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock(status_code=404),
            )

        def mock_download_year(year_arg):
            return tmp_raw_dir / "noaa_ncei" / f"ghcn_daily_{year_arg}.csv.gz"

        with (
            patch.object(ingester, "api_get", side_effect=mock_api_get),
            patch.object(
                ingester, "_download_year_csv", side_effect=mock_download_year,
            ),
        ):
            df = ingester.fetch(years=[2022, 2023])

        assert len(df) == single_year_count * 2

    def test_partial_failure_still_returns_data(self, ingester, tmp_raw_dir):
        """If one year fails to download, the other years still succeed."""
        _write_gz_fixture(tmp_raw_dir, SAMPLE_YEAR_CSV_GZ, year=2022)

        def mock_api_get(url, params=None, headers=None):
            if "ghcnd-stations" in url:
                return _mock_stations_response()
            if "normals-monthly" in url:
                return _mock_normals_response()
            raise httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock(status_code=404),
            )

        def mock_download_year(year_arg):
            if year_arg == 2023:
                raise httpx.HTTPStatusError(
                    "500", request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            return tmp_raw_dir / "noaa_ncei" / f"ghcn_daily_{year_arg}.csv.gz"

        with (
            patch.object(ingester, "api_get", side_effect=mock_api_get),
            patch.object(
                ingester, "_download_year_csv", side_effect=mock_download_year,
            ),
        ):
            df = ingester.fetch(years=[2022, 2023])

        # Only 2022 data should be present
        assert len(df) > 0
        assert all(d.year == 2022 for d in df["date"] if d is not None)


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic triggers on HTTP 500/503."""

    def test_retries_on_500(self, ingester):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.text = SAMPLE_STATIONS_TEXT
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
            resp = ingester.api_get(GHCN_STATIONS_URL)

        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "noaa_ncei"

    def test_confidence(self, ingester):
        assert ingester.confidence == "A"

    def test_attribution(self, ingester):
        assert ingester.attribution == "proxy"

    def test_calls_per_second(self, ingester):
        """Rate limit is polite (<=1 req/sec for static downloads)."""
        assert ingester.calls_per_second <= 1.0
