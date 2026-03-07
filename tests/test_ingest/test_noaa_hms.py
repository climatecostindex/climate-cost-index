"""Tests for the NOAA HMS smoke plume ingester (ingest/noaa_hms.py).

All HTTP calls are mocked — no real requests to NOAA servers.
Synthetic shapefiles are created using geopandas + shapely in fixtures.
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from ingest.noaa_hms import (
    HMS_BASE_URL,
    HMS_CALLS_PER_SECOND,
    HMS_FILENAME_PATTERN,
    HMS_MONTH_PATTERN,
    NOAAHMSIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_smoke_gdf(
    densities: list[str] | None = None,
    density_col_name: str = "Density",
    polygons: list[Polygon] | None = None,
) -> gpd.GeoDataFrame:
    """Create a synthetic smoke plume GeoDataFrame.

    Args:
        densities: Density classification for each plume.
        density_col_name: Column name for density (tests normalization).
        polygons: Plume polygon geometries.

    Returns:
        GeoDataFrame mimicking HMS shapefile content.
    """
    if densities is None:
        densities = ["Light", "Medium", "Heavy"]
    if polygons is None:
        polygons = [
            box(-100, 30, -95, 35),
            box(-90, 35, -85, 40),
            box(-80, 25, -75, 30),
        ]
    polygons = polygons[: len(densities)]
    return gpd.GeoDataFrame(
        {density_col_name: densities},
        geometry=polygons,
        crs="EPSG:4326",
    )


def _write_shapefile_to_dir(
    gdf: gpd.GeoDataFrame, dest_dir: Path, stem: str
) -> Path:
    """Write a GeoDataFrame to shapefile components on disk.

    Args:
        gdf: Data to write.
        dest_dir: Directory for the shapefile components.
        stem: File stem (e.g., "hms_smoke20230815").

    Returns:
        Path to the .shp file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    shp_path = dest_dir / f"{stem}.shp"
    gdf.to_file(shp_path)
    return shp_path


@pytest.fixture
def ingester():
    """Return a fresh NOAAHMSIngester instance."""
    return NOAAHMSIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


def _setup_cached_shapefiles(
    tmp_raw_dir: Path, year: int, date_str: str,
    gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """Pre-populate the staging directory with shapefile components."""
    if gdf is None:
        gdf = _make_smoke_gdf()
    staging = (
        tmp_raw_dir / "noaa_hms" / "shapefiles" / str(year) / date_str
    )
    _write_shapefile_to_dir(gdf, staging, f"hms_smoke{date_str}")


def _mock_month_discovery(
    ingester: NOAAHMSIngester,
    months: list[str],
    dates_by_month: dict[str, list[str]],
):
    """Return context managers that mock month and date discovery.

    Args:
        ingester: The ingester instance to patch.
        months: Month strings to return from year listing.
        dates_by_month: Mapping of month → list of date strings.
    """
    return (
        patch.object(ingester, "_get_months_for_year", return_value=months),
        patch.object(
            ingester,
            "_get_dates_for_month",
            side_effect=lambda year, month: dates_by_month.get(month, []),
        ),
    )


def _run_fetch_with_cached_data(
    ingester: NOAAHMSIngester,
    tmp_raw_dir: Path,
    year_dates: dict[int, list[str]],
    gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Run fetch() with mocked discovery and pre-cached shapefiles.

    Args:
        ingester: Ingester instance.
        tmp_raw_dir: Temp directory for raw data.
        year_dates: Mapping of year → list of date strings (YYYYMMDD).
        gdf: Synthetic GeoDataFrame to use for each date.

    Returns:
        Result GeoDataFrame from fetch().
    """
    if gdf is None:
        gdf = _make_smoke_gdf()

    # Pre-cache all dates
    for year, dates in year_dates.items():
        for date_str in dates:
            _setup_cached_shapefiles(tmp_raw_dir, year, date_str, gdf)

    # Build mock month discovery per year
    def mock_months(year):
        dates = year_dates.get(year, [])
        months = sorted({d[4:6] for d in dates})
        return months

    def mock_dates(year, month):
        return sorted(
            d for d in year_dates.get(year, []) if d[4:6] == month
        )

    with (
        patch.object(ingester, "_get_months_for_year", side_effect=mock_months),
        patch.object(ingester, "_get_dates_for_month", side_effect=mock_dates),
    ):
        return ingester.fetch(years=list(year_dates.keys()))


# ---------------------------------------------------------------------------
# Test: Geometry and density parsing
# ---------------------------------------------------------------------------

class TestGeometryAndDensityParsing:
    """Verify shapefile geometry and density are correctly parsed."""

    def test_polygon_parsing(self, ingester, tmp_raw_dir):
        """Smoke plume polygons are correctly parsed from shapefiles."""
        gdf = _make_smoke_gdf(densities=["Light", "Medium"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir,
            {2023: ["20230815", "20230816"]}, gdf,
        )
        assert not result.empty
        assert "geometry" in result.columns
        for geom in result.geometry:
            assert geom is not None
            assert geom.geom_type in ("Polygon", "MultiPolygon")

    def test_density_classification_preserved(self, ingester, tmp_raw_dir):
        """Density values are preserved as strings: Light, Medium, Heavy."""
        gdf = _make_smoke_gdf(densities=["Light", "Heavy"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        assert set(result["density"].unique()) == {"Light", "Heavy"}

    def test_density_column_normalization_uppercase(self, ingester, tmp_raw_dir):
        """Ingester handles DENSITY column name (all caps)."""
        gdf = _make_smoke_gdf(
            densities=["Medium"], density_col_name="DENSITY"
        )
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        assert "density" in result.columns
        assert (result["density"] == "Medium").all()

    def test_density_column_normalization_lowercase(self, ingester, tmp_raw_dir):
        """Ingester handles density column name (all lowercase)."""
        gdf = _make_smoke_gdf(
            densities=["Light"], density_col_name="density"
        )
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        assert "density" in result.columns
        assert (result["density"] == "Light").all()

    def test_density_column_normalization_titlecase(self, ingester, tmp_raw_dir):
        """Ingester handles Density column name (title case)."""
        gdf = _make_smoke_gdf(
            densities=["Heavy"], density_col_name="Density"
        )
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        assert "density" in result.columns
        assert (result["density"] == "Heavy").all()

    def test_date_extraction_from_filename(self, ingester, tmp_raw_dir):
        """Date is correctly extracted from filename hms_smoke20230815."""
        gdf = _make_smoke_gdf(densities=["Light"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        dates = result["date"].unique()
        assert len(dates) == 1
        assert dates[0] == date(2023, 8, 15)

    def test_multiple_dates_parsed(self, ingester, tmp_raw_dir):
        """Multiple dates produce distinct date values in output."""
        gdf = _make_smoke_gdf(densities=["Light"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir,
            {2023: ["20230815", "20230816"]}, gdf,
        )
        dates = sorted(result["date"].unique())
        assert date(2023, 8, 15) in dates
        assert date(2023, 8, 16) in dates


# ---------------------------------------------------------------------------
# Test: Missing and malformed data
# ---------------------------------------------------------------------------

class TestMissingAndMalformedData:
    """Verify graceful handling of missing or broken shapefiles."""

    def test_missing_date_not_fatal(self, ingester, tmp_raw_dir):
        """Dates with no shapefile on server are logged, not errors."""
        gdf = _make_smoke_gdf(densities=["Light"])
        # Only cache one of two dates
        _setup_cached_shapefiles(tmp_raw_dir, 2023, "20230815", gdf)

        def mock_months(year):
            return ["08"]

        def mock_dates(year, month):
            return ["20230815", "20230816"]

        with (
            patch.object(ingester, "_get_months_for_year", side_effect=mock_months),
            patch.object(ingester, "_get_dates_for_month", side_effect=mock_dates),
            patch.object(ingester, "_download_date", return_value=False),
        ):
            result = ingester.fetch(years=[2023])

        assert not result.empty
        assert date(2023, 8, 15) in result["date"].values

    def test_empty_shapefile_handled_gracefully(self, ingester, tmp_raw_dir):
        """Empty shapefile (no features) is logged and skipped."""
        empty_gdf = gpd.GeoDataFrame(
            {"Density": pd.Series(dtype=str)},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
        _setup_cached_shapefiles(tmp_raw_dir, 2023, "20230815", empty_gdf)

        valid_gdf = _make_smoke_gdf(densities=["Light"])
        _setup_cached_shapefiles(tmp_raw_dir, 2023, "20230816", valid_gdf)

        def mock_months(year):
            return ["08"]

        def mock_dates(year, month):
            return ["20230815", "20230816"]

        with (
            patch.object(ingester, "_get_months_for_year", side_effect=mock_months),
            patch.object(ingester, "_get_dates_for_month", side_effect=mock_dates),
        ):
            result = ingester.fetch(years=[2023])

        assert len(result) == 1
        assert result.iloc[0]["date"] == date(2023, 8, 16)

    def test_malformed_shapefile_logged_and_skipped(self, ingester, tmp_raw_dir):
        """Corrupted shapefile is warned and skipped, not fatal."""
        staging = (
            tmp_raw_dir / "noaa_hms" / "shapefiles" / "2023" / "20230815"
        )
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "hms_smoke20230815.shp").write_bytes(b"not a shapefile")
        (staging / "hms_smoke20230815.dbf").write_bytes(b"corrupt")

        def mock_months(year):
            return ["08"]

        def mock_dates(year, month):
            return ["20230815"]

        with (
            patch.object(ingester, "_get_months_for_year", side_effect=mock_months),
            patch.object(ingester, "_get_dates_for_month", side_effect=mock_dates),
        ):
            result = ingester.fetch(years=[2023])

        assert result.empty


# ---------------------------------------------------------------------------
# Test: Output schema and caching
# ---------------------------------------------------------------------------

class TestOutputAndCaching:
    """Verify output schema, geometry types, and metadata sidecar."""

    def test_output_schema_columns(self, ingester, tmp_raw_dir):
        """Output GeoDataFrame contains exactly: date, geometry, density."""
        gdf = _make_smoke_gdf(densities=["Light"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        expected = {"date", "geometry", "density"}
        assert set(result.columns) == expected

    def test_geometry_type_polygon_or_multipolygon(self, ingester, tmp_raw_dir):
        """Geometry column contains Polygon or MultiPolygon types."""
        polys = [box(-100, 30, -95, 35)]
        multi = [MultiPolygon([box(-80, 25, -75, 30), box(-70, 20, -65, 25)])]
        gdf = _make_smoke_gdf(
            densities=["Light", "Heavy"],
            polygons=polys + multi,
        )
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        for geom in result.geometry:
            assert geom.geom_type in ("Polygon", "MultiPolygon")

    def test_ingest_purity_no_derived_columns(self, ingester, tmp_raw_dir):
        """Output must NOT contain any derived metrics."""
        gdf = _make_smoke_gdf(densities=["Light"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        forbidden = {
            "county_fips", "fips", "smoke_day_count", "pm25_overlay",
            "intersects_county", "annual_smoke_days", "score", "percentile",
        }
        assert forbidden.isdisjoint(set(result.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        gdf = _make_smoke_gdf(densities=["Light"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        assert set(result.columns) == set(ingester.required_columns)

    def test_metadata_sidecar_written(self, ingester, tmp_raw_dir):
        """Metadata JSON is written alongside the yearly parquet."""
        gdf = _make_smoke_gdf(densities=["Light"])
        _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        meta_path = tmp_raw_dir / "noaa_hms" / "noaa_hms_2023_metadata.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["source"] == "NOAA_HMS"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_yearly_parquet_cached(self, ingester, tmp_raw_dir):
        """Yearly GeoParquet file is cached to data/raw/noaa_hms/."""
        gdf = _make_smoke_gdf(densities=["Light"])
        _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        parquet_path = tmp_raw_dir / "noaa_hms" / "noaa_hms_2023.parquet"
        assert parquet_path.exists()

    def test_combined_parquet_cached(self, ingester, tmp_raw_dir):
        """Combined GeoParquet file noaa_hms_all.parquet is cached."""
        gdf = _make_smoke_gdf(densities=["Light"])
        _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        parquet_path = tmp_raw_dir / "noaa_hms" / "noaa_hms_all.parquet"
        assert parquet_path.exists()

    def test_cached_geoparquet_readable(self, ingester, tmp_raw_dir):
        """Cached GeoParquet can be loaded back with geopandas."""
        gdf = _make_smoke_gdf(densities=["Light"])
        _run_fetch_with_cached_data(
            ingester, tmp_raw_dir, {2023: ["20230815"]}, gdf,
        )
        parquet_path = tmp_raw_dir / "noaa_hms" / "noaa_hms_2023.parquet"
        loaded = gpd.read_parquet(parquet_path)
        assert "geometry" in loaded.columns
        assert not loaded.empty


# ---------------------------------------------------------------------------
# Test: Download behavior
# ---------------------------------------------------------------------------

class TestDownloadBehavior:
    """Verify incremental download, multi-year, partial failure, retry."""

    def test_incremental_download_skips_cached(self, ingester, tmp_raw_dir):
        """Dates already cached locally are not re-downloaded."""
        gdf = _make_smoke_gdf(densities=["Light"])
        _setup_cached_shapefiles(tmp_raw_dir, 2023, "20230815", gdf)
        _setup_cached_shapefiles(tmp_raw_dir, 2023, "20230816", gdf)

        download_calls = []

        def _tracking_download(year, date_str):
            download_calls.append(date_str)
            return False

        def mock_months(year):
            return ["08"]

        def mock_dates(year, month):
            return ["20230815", "20230816"]

        with (
            patch.object(ingester, "_get_months_for_year", side_effect=mock_months),
            patch.object(ingester, "_get_dates_for_month", side_effect=mock_dates),
            patch.object(ingester, "_download_date", side_effect=_tracking_download),
        ):
            result = ingester.fetch(years=[2023])

        assert len(download_calls) == 0
        assert not result.empty

    def test_multi_year_download(self, ingester, tmp_raw_dir):
        """Ingester fetches data across multiple years."""
        gdf = _make_smoke_gdf(densities=["Light"])
        result = _run_fetch_with_cached_data(
            ingester, tmp_raw_dir,
            {2022: ["20220701"], 2023: ["20230815"]}, gdf,
        )
        dates = sorted(result["date"].unique())
        assert date(2022, 7, 1) in dates
        assert date(2023, 8, 15) in dates

    def test_partial_failure_returns_successful(self, ingester, tmp_raw_dir):
        """If some dates fail download, other dates' data is still cached."""
        gdf = _make_smoke_gdf(densities=["Light"])
        _setup_cached_shapefiles(tmp_raw_dir, 2023, "20230815", gdf)

        def _mock_download(year, date_str):
            return False

        def mock_months(year):
            return ["08"]

        def mock_dates(year, month):
            return ["20230815", "20230816"]

        with (
            patch.object(ingester, "_get_months_for_year", side_effect=mock_months),
            patch.object(ingester, "_get_dates_for_month", side_effect=mock_dates),
            patch.object(ingester, "_download_date", side_effect=_mock_download),
        ):
            result = ingester.fetch(years=[2023])

        assert not result.empty
        assert date(2023, 8, 15) in result["date"].values
        assert date(2023, 8, 16) not in result["date"].values

    def test_all_fail_returns_empty(self, ingester, tmp_raw_dir):
        """If all years fail, return empty GeoDataFrame with correct columns."""
        with patch.object(
            ingester, "_get_months_for_year",
            side_effect=Exception("connection failed"),
        ):
            result = ingester.fetch(years=[2023])

        assert result.empty
        assert set(result.columns) == {"date", "geometry", "density"}

    def test_retry_on_500(self, ingester, tmp_raw_dir):
        """Retry logic triggers on HTTP 500/503 via base class api_get."""
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.text = '<a href="08/">08/</a>'
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
            resp = ingester.api_get(f"{HMS_BASE_URL}2023/")

        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "noaa_hms"

    def test_confidence(self, ingester):
        assert ingester.confidence == "B"

    def test_attribution(self, ingester):
        assert ingester.attribution == "proxy"

    def test_calls_per_second(self, ingester):
        """Rate limit is polite (<=1 req/sec for NOAA server)."""
        assert ingester.calls_per_second <= 1.0


# ---------------------------------------------------------------------------
# Test: _parse_shapefile directly
# ---------------------------------------------------------------------------

class TestParseShapefile:
    """Unit tests for the _parse_shapefile method."""

    def test_parse_valid_shapefile(self, ingester, tmp_path):
        """Valid shapefile is parsed correctly."""
        gdf = _make_smoke_gdf(densities=["Light", "Heavy"])
        shp_path = _write_shapefile_to_dir(
            gdf, tmp_path / "20230815", "hms_smoke20230815"
        )
        result = ingester._parse_shapefile(shp_path, "20230815")

        assert result is not None
        assert len(result) == 2
        assert set(result.columns) == {"date", "geometry", "density"}
        assert set(result["density"]) == {"Light", "Heavy"}
        assert (result["date"] == date(2023, 8, 15)).all()

    def test_parse_empty_shapefile_returns_none(self, ingester, tmp_path):
        """Empty shapefile (no features) returns None."""
        empty_gdf = gpd.GeoDataFrame(
            {"Density": pd.Series(dtype=str)},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
        shp_path = _write_shapefile_to_dir(
            empty_gdf, tmp_path / "20230815", "hms_smoke20230815"
        )
        result = ingester._parse_shapefile(shp_path, "20230815")
        assert result is None

    def test_parse_no_density_column_returns_none(self, ingester, tmp_path):
        """Shapefile without a density column returns None."""
        gdf = gpd.GeoDataFrame(
            {"other_col": ["foo"]},
            geometry=[box(-100, 30, -95, 35)],
            crs="EPSG:4326",
        )
        shp_path = _write_shapefile_to_dir(
            gdf, tmp_path / "20230815", "hms_smoke20230815"
        )
        result = ingester._parse_shapefile(shp_path, "20230815")
        assert result is None

    def test_parse_multipolygon_geometry(self, ingester, tmp_path):
        """MultiPolygon geometries are preserved."""
        multi = MultiPolygon([
            box(-100, 30, -95, 35),
            box(-80, 25, -75, 30),
        ])
        gdf = gpd.GeoDataFrame(
            {"Density": ["Heavy"]},
            geometry=[multi],
            crs="EPSG:4326",
        )
        shp_path = _write_shapefile_to_dir(
            gdf, tmp_path / "20230815", "hms_smoke20230815"
        )
        result = ingester._parse_shapefile(shp_path, "20230815")
        assert result is not None
        assert result.geometry.iloc[0].geom_type == "MultiPolygon"


# ---------------------------------------------------------------------------
# Test: Directory listing parsing
# ---------------------------------------------------------------------------

class TestDirectoryParsing:
    """Unit tests for HTML directory listing parsing."""

    def test_month_pattern_finds_months(self):
        """Month regex finds two-digit month directories."""
        html = '<a href="01/">01/</a> <a href="08/">08/</a> <a href="12/">12/</a>'
        months = HMS_MONTH_PATTERN.findall(html)
        assert months == ["01", "08", "12"]

    def test_month_pattern_ignores_non_months(self):
        """Month regex ignores non-month hrefs."""
        html = '<a href="../">../</a> <a href="readme.txt">readme.txt</a>'
        months = HMS_MONTH_PATTERN.findall(html)
        assert months == []

    def test_filename_pattern_finds_zips(self):
        """Filename regex finds hms_smoke zip files (deduped)."""
        html = (
            '<a href="hms_smoke20230815.zip">hms_smoke20230815.zip</a>'
            '<a href="hms_smoke20230816.zip">hms_smoke20230816.zip</a>'
        )
        # findall matches in both href and link text; deduplicate like the ingester
        dates = sorted(set(HMS_FILENAME_PATTERN.findall(html)))
        assert dates == ["20230815", "20230816"]

    def test_filename_pattern_ignores_non_matching(self):
        """Filename regex ignores unrelated files."""
        html = '<a href="readme.txt">readme.txt</a>'
        dates = HMS_FILENAME_PATTERN.findall(html)
        assert dates == []


# ---------------------------------------------------------------------------
# Test: validate_output
# ---------------------------------------------------------------------------

class TestValidateOutput:
    """Verify base class validate_output works with GeoDataFrame schema."""

    def test_validate_rejects_extra_columns(self, ingester):
        """Extra columns cause validation failure."""
        gdf = gpd.GeoDataFrame(
            {
                "date": [date(2023, 8, 15)],
                "density": ["Light"],
                "county_fips": ["06037"],  # FORBIDDEN
            },
            geometry=[box(-100, 30, -95, 35)],
        )
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(gdf)

    def test_validate_passes_for_correct_schema(self, ingester):
        """Correct schema passes validation."""
        gdf = gpd.GeoDataFrame(
            {
                "date": [date(2023, 8, 15)],
                "density": ["Light"],
            },
            geometry=[box(-100, 30, -95, 35)],
        )
        # Should not raise
        ingester.validate_output(gdf)
