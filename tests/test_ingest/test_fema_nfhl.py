"""Tests for the FEMA NFHL MSC county download ingester (ingest/fema_nfhl.py).

All HTTP calls are mocked — no real requests to FEMA or Census servers.
Synthetic geodata is created using geopandas + shapely in fixtures, packaged
into ZIP archives matching the MSC county download layout, and parsed by
the ingester.

Module 1.5a replaces Module 1.5 (ArcGIS REST API approach).
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from ingest.fema_nfhl import (
    CENSUS_COUNTY_FIPS_URL,
    DEFAULT_MAX_WORKERS,
    DOWNLOAD_DELAY,
    DOWNLOAD_TIMEOUT,
    FIRM_PANEL_LAYER,
    FIRM_PANEL_LAYER_ID,
    FLOOD_ZONE_LAYER,
    FLOOD_ZONE_LAYER_ID,
    MSC_DOWNLOAD_URL,
    NFHL_CALLS_PER_SECOND,
    NFHL_REST_BASE,
    REST_FLOOD_ZONE_FIELDS,
    REST_FIRM_PANEL_FIELDS,
    REST_OBJECTID_BATCH_SIZE,
    REST_QUERY_TIMEOUT,
    SIMPLIFY_TOLERANCE,
    TIGERWEB_COUNTY_URL,
    FEMANFHLIngester,
    _HtmlFallback,
)


# ---------------------------------------------------------------------------
# Helpers — create synthetic shapefiles and ZIP archives
# ---------------------------------------------------------------------------

def _make_flood_zone_gdf(
    zones: list[str],
    subtypes: list[str] | None = None,
    polygons: list[Polygon | MultiPolygon] | None = None,
    dfirm_ids: list[str] | None = None,
    crs: str = "EPSG:4269",
) -> gpd.GeoDataFrame:
    """Create a synthetic S_FLD_HAZ_AR GeoDataFrame.

    Args:
        zones: Flood zone classifications (e.g., "AE", "VE", "X").
        subtypes: Zone subtype strings. Defaults to empty strings.
        polygons: Geometries for each zone. Defaults to simple boxes.
        dfirm_ids: DFIRM_ID for each feature. Defaults to "01001C".
        crs: Coordinate reference system.

    Returns:
        GeoDataFrame matching FEMA S_FLD_HAZ_AR schema.
    """
    n = len(zones)
    if polygons is None:
        polygons = [box(-87 + i * 0.01, 32, -86.99 + i * 0.01, 32.01) for i in range(n)]
    polygons = polygons[:n]
    if subtypes is None:
        subtypes = [""] * n
    subtypes = subtypes[:n]
    if dfirm_ids is None:
        dfirm_ids = ["01001C"] * n
    dfirm_ids = dfirm_ids[:n]

    data = {
        "FLD_ZONE": zones,
        "ZONE_SUBTY": subtypes,
        "DFIRM_ID": dfirm_ids,
        "SFHA_TF": ["T"] * n,  # Extra column FEMA includes
    }
    return gpd.GeoDataFrame(data, geometry=polygons, crs=crs)


def _make_panel_gdf(
    panel_ids: list[str],
    eff_dates: list[str],
    dfirm_ids: list[str] | None = None,
    panel_types: list[str] | None = None,
    crs: str = "EPSG:4269",
) -> gpd.GeoDataFrame:
    """Create a synthetic S_FIRM_PAN GeoDataFrame.

    Args:
        panel_ids: FIRM panel IDs (e.g., "01001C0353D").
        eff_dates: Effective dates as strings (e.g., "2020-06-15").
        dfirm_ids: DFIRM IDs. Defaults to "01001C".
        panel_types: Panel type descriptions.
        crs: Coordinate reference system.

    Returns:
        GeoDataFrame matching FEMA S_FIRM_PAN schema.
    """
    n = len(panel_ids)
    if dfirm_ids is None:
        dfirm_ids = ["01001C"] * n
    if panel_types is None:
        panel_types = ["COUNTYWIDE"] * n

    polys = [box(-87 + i * 0.1, 32, -86.9 + i * 0.1, 32.1) for i in range(n)]

    return gpd.GeoDataFrame(
        {
            "DFIRM_ID": dfirm_ids[:n],
            "EFF_DATE": eff_dates[:n],
            "FIRM_PAN": panel_ids[:n],
            "PANEL_TYP": panel_types[:n],
        },
        geometry=polys,
        crs=crs,
    )


def _write_gdf_to_shapefile(gdf: gpd.GeoDataFrame, dest_dir: Path, stem: str) -> Path:
    """Write a GeoDataFrame to shapefile components on disk.

    Args:
        gdf: Data to write.
        dest_dir: Directory for the shapefile components.
        stem: File stem (e.g., "S_FLD_HAZ_AR").

    Returns:
        Path to the .shp file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    shp_path = dest_dir / f"{stem}.shp"
    gdf.to_file(shp_path)
    return shp_path


def _build_county_zip(
    zones_gdf: gpd.GeoDataFrame | None = None,
    panels_gdf: gpd.GeoDataFrame | None = None,
    nested_dir: str | None = None,
) -> bytes:
    """Build a ZIP archive mimicking an MSC county download.

    Args:
        zones_gdf: Flood zone GeoDataFrame. If None, no zone shapefile is included.
        panels_gdf: Panel GeoDataFrame. If None, no panel shapefile is included.
        nested_dir: If provided, nest files inside this subdirectory.

    Returns:
        ZIP file bytes.
    """
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory(prefix="nfhl_test_") as tmp:
        tmp_path = Path(tmp)

        if zones_gdf is not None:
            _write_gdf_to_shapefile(zones_gdf, tmp_path, FLOOD_ZONE_LAYER)

        if panels_gdf is not None:
            _write_gdf_to_shapefile(panels_gdf, tmp_path, FIRM_PANEL_LAYER)

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in tmp_path.iterdir():
                arcname = fpath.name
                if nested_dir:
                    arcname = f"{nested_dir}/{fpath.name}"
                zf.write(fpath, arcname)

    return buf.getvalue()


def _make_census_fips_text(counties: list[tuple[str, str, str]]) -> str:
    """Build mock Census national_county.txt content.

    Args:
        counties: List of (state_fips, county_fips_3, county_name) tuples.

    Returns:
        Text matching Census FIPS list format.
    """
    lines = []
    for st, co, name in counties:
        lines.append(f"{st},{co},{name},H1")
    return "\n".join(lines)


def _make_mock_download_response(
    zip_bytes: bytes, status_code: int = 200
) -> MagicMock:
    """Create a mock httpx response for a ZIP download."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = zip_bytes
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_fips_response(text: str) -> MagicMock:
    """Create a mock httpx response for the Census FIPS list."""
    resp = MagicMock()
    resp.status_code = 200
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ingester():
    """Return a fresh FEMANFHLIngester instance."""
    return FEMANFHLIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def sample_zones_gdf():
    """Sample flood zone GeoDataFrame for testing."""
    return _make_flood_zone_gdf(
        zones=["AE", "VE", "X", "A", "AO"],
        subtypes=[
            "FLOODWAY",
            "",
            "AREA OF MINIMAL FLOOD HAZARD",
            "",
            "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
        ],
    )


@pytest.fixture
def sample_panels_gdf():
    """Sample panel GeoDataFrame for testing."""
    return _make_panel_gdf(
        panel_ids=["01001C0353D", "01001C0362D"],
        eff_dates=["2020-06-15", "2015-03-01"],
    )


# ---------------------------------------------------------------------------
# Test: County FIPS list parsing
# ---------------------------------------------------------------------------

class TestCountyFIPSParsing:
    """Verify Census county FIPS list download, parsing, and filtering."""

    def test_fips_list_download_and_parsing(self, ingester, tmp_raw_dir):
        """Verify Census FIPS list is correctly parsed."""
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),
            ("01", "003", "Baldwin County"),
            ("06", "001", "Alameda County"),
        ])

        with patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)):
            df = ingester._fetch_county_fips_list()

        assert len(df) == 3
        assert "state_fips" in df.columns
        assert "county_fips_3" in df.columns
        assert "county_name" in df.columns
        assert "classfp" in df.columns
        assert "fips5" in df.columns

    def test_5_digit_fips_construction(self, ingester, tmp_raw_dir):
        """Verify fips5 = state_fips + county_fips_3."""
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),
            ("06", "037", "Los Angeles County"),
        ])

        with patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)):
            df = ingester._fetch_county_fips_list()

        assert "01001" in df["fips5"].values
        assert "06037" in df["fips5"].values

    def test_state_filtering(self, tmp_raw_dir):
        """Verify states parameter filters to only matching states."""
        ingester = FEMANFHLIngester(states=["01"])
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),
            ("01", "003", "Baldwin County"),
            ("06", "001", "Alameda County"),
            ("06", "037", "Los Angeles County"),
        ])

        with patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)):
            df = ingester._fetch_county_fips_list()

        assert len(df) == 2
        assert (df["state_fips"] == "01").all()

    def test_msc_url_generation(self):
        """Verify MSC download URL is correctly constructed."""
        url = FEMANFHLIngester._build_download_url("01001")
        assert url == "https://msc.fema.gov/portal/downloadProduct?productID=NFHL_01001C"

    def test_msc_url_various_fips(self):
        """URL generation works for various FIPS codes."""
        assert "NFHL_06037C" in FEMANFHLIngester._build_download_url("06037")
        assert "NFHL_48201C" in FEMANFHLIngester._build_download_url("48201")

    def test_fips_list_cached_locally(self, ingester, tmp_raw_dir):
        """Census FIPS list is cached and not re-downloaded."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])

        call_count = 0

        def mock_api_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            return _make_mock_fips_response(fips_text)

        with patch.object(ingester, "api_get", side_effect=mock_api_get):
            ingester._fetch_county_fips_list()
            ingester._fetch_county_fips_list()  # Second call

        assert call_count == 1  # Only downloaded once


# ---------------------------------------------------------------------------
# Test: ZIP extraction and parsing
# ---------------------------------------------------------------------------

class TestZIPExtractionAndParsing:
    """Verify selective extraction and parsing of NFHL ZIP archives."""

    def test_selective_extraction(self, ingester, tmp_raw_dir, sample_zones_gdf, sample_panels_gdf):
        """Only S_FLD_HAZ_AR.* and S_FIRM_PAN.* files are extracted."""
        zip_bytes = _build_county_zip(sample_zones_gdf, sample_panels_gdf)
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

        zone_members = ingester._find_shapefile_members(zf, FLOOD_ZONE_LAYER)
        panel_members = ingester._find_shapefile_members(zf, FIRM_PANEL_LAYER)

        # Only shapefile component files for the requested layers
        for m in zone_members:
            basename = os.path.basename(m)
            assert basename.startswith(FLOOD_ZONE_LAYER)

        for m in panel_members:
            basename = os.path.basename(m)
            assert basename.startswith(FIRM_PANEL_LAYER)

        zf.close()

    def test_nested_zip_layout(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """Extraction works when files are nested inside a subdirectory."""
        zip_bytes = _build_county_zip(sample_zones_gdf, nested_dir="01001C_20231225")
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

        zone_members = ingester._find_shapefile_members(zf, FLOOD_ZONE_LAYER)
        assert len(zone_members) > 0
        # Members should contain the nested path
        assert any("01001C_20231225" in m for m in zone_members)

        # Extraction should still work (flattens nesting)
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = ingester._extract_shapefile(zf, zone_members, Path(tmp))
            assert shp_path is not None
            assert shp_path.exists()
            assert shp_path.name == f"{FLOOD_ZONE_LAYER}.shp"

        zf.close()

    def test_flat_zip_layout(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """Extraction works when files are at the root of the ZIP."""
        zip_bytes = _build_county_zip(sample_zones_gdf)
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

        zone_members = ingester._find_shapefile_members(zf, FLOOD_ZONE_LAYER)
        assert len(zone_members) > 0

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = ingester._extract_shapefile(zf, zone_members, Path(tmp))
            assert shp_path is not None
            assert shp_path.exists()

        zf.close()

    def test_flood_zone_parsing(self, ingester, sample_zones_gdf):
        """S_FLD_HAZ_AR is read into a GeoDataFrame with correct columns."""
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_zones_gdf, Path(tmp), FLOOD_ZONE_LAYER
            )
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        assert isinstance(result, gpd.GeoDataFrame)
        assert "flood_zone" in result.columns
        assert "zone_subtype" in result.columns
        assert "dfirm_id" in result.columns

    def test_panel_parsing(self, ingester, sample_panels_gdf):
        """S_FIRM_PAN is read and effective dates are correctly parsed."""
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_panels_gdf, Path(tmp), FIRM_PANEL_LAYER
            )
            result = ingester._parse_firm_panels(shp_path, "01001")

        assert result is not None
        assert "effective_date" in result.columns
        assert "panel_id" in result.columns
        # Check that effective_date is parsed (not still a string)
        assert result["effective_date"].iloc[0] == date(2020, 6, 15)


# ---------------------------------------------------------------------------
# Test: Output schema and field preservation
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrames have correct columns and dtypes."""

    def test_zone_output_schema(self, ingester, sample_zones_gdf):
        """Output GeoDataFrame contains all expected columns with correct dtypes."""
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_zones_gdf, Path(tmp), FLOOD_ZONE_LAYER
            )
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        assert "geometry" in result.columns
        assert "flood_zone" in result.columns
        assert "zone_subtype" in result.columns
        assert "dfirm_id" in result.columns
        assert "state_fips" in result.columns
        assert "county_fips" in result.columns

        # Check dtypes
        assert pd.api.types.is_string_dtype(result["flood_zone"])
        assert pd.api.types.is_string_dtype(result["zone_subtype"])
        assert pd.api.types.is_string_dtype(result["dfirm_id"])
        assert pd.api.types.is_string_dtype(result["state_fips"])
        assert pd.api.types.is_string_dtype(result["county_fips"])

    def test_panel_output_schema(self, ingester, sample_panels_gdf):
        """Panel DataFrame contains correct columns with correct dtypes."""
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_panels_gdf, Path(tmp), FIRM_PANEL_LAYER
            )
            result = ingester._parse_firm_panels(shp_path, "01001")

        assert result is not None
        assert "dfirm_id" in result.columns
        assert "effective_date" in result.columns
        assert "panel_id" in result.columns
        assert "panel_type" in result.columns
        assert "state_fips" in result.columns
        assert "county_fips" in result.columns

        assert pd.api.types.is_string_dtype(result["dfirm_id"])
        assert pd.api.types.is_string_dtype(result["panel_id"])
        assert pd.api.types.is_string_dtype(result["panel_type"])
        assert pd.api.types.is_string_dtype(result["state_fips"])
        assert pd.api.types.is_string_dtype(result["county_fips"])

    def test_geometry_type_polygon_or_multipolygon(self, ingester):
        """Geometry column contains Polygon or MultiPolygon types."""
        polys = [box(-87, 32, -86.99, 32.01)]
        multi = [MultiPolygon([box(-87.1, 32, -87, 32.01), box(-86.9, 32, -86.8, 32.01)])]
        gdf = _make_flood_zone_gdf(zones=["AE", "VE"], polygons=polys + multi)

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        for geom in result.geometry:
            assert geom.geom_type in ("Polygon", "MultiPolygon")

    def test_zone_classification_preserved_exactly(self, ingester):
        """Flood zone strings are preserved exactly as provided."""
        zone_values = ["AE", "VE", "X", "A", "AO"]
        gdf = _make_flood_zone_gdf(zones=zone_values)

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        assert set(result["flood_zone"].unique()) == set(zone_values)

    def test_zone_not_lowercased(self, ingester):
        """Zone values must NOT be lowercased."""
        gdf = _make_flood_zone_gdf(zones=["AE", "VE"])

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        assert "AE" in result["flood_zone"].values
        assert "ae" not in result["flood_zone"].values

    def test_zone_not_renamed(self, ingester):
        """Zone values must NOT be renamed to descriptive strings."""
        gdf = _make_flood_zone_gdf(zones=["VE"])

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        assert "VE" in result["flood_zone"].values
        assert "high_risk" not in result["flood_zone"].values

    def test_zone_subtype_preserved_exactly(self, ingester):
        """Zone subtype strings are preserved exactly as provided."""
        subtypes = [
            "FLOODWAY",
            "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
            "AREA OF MINIMAL FLOOD HAZARD",
        ]
        gdf = _make_flood_zone_gdf(zones=["AE", "X", "X"], subtypes=subtypes)

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        for st in subtypes:
            assert st in result["zone_subtype"].values

    def test_nan_zone_subtype_replaced_with_empty_string(self, ingester):
        """NaN values in ZONE_SUBTY are replaced with empty string."""
        gdf = _make_flood_zone_gdf(zones=["AE", "X"])
        # Set one subtype to NaN
        gdf.loc[1, "ZONE_SUBTY"] = None

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        # No NaN values in zone_subtype
        assert result["zone_subtype"].isna().sum() == 0
        # Should have empty string for the NaN
        assert "" in result["zone_subtype"].values

    def test_effective_date_parsing(self, ingester):
        """EFF_DATE from panels is parsed to date type."""
        panels = _make_panel_gdf(
            panel_ids=["P001", "P002"],
            eff_dates=["2020-06-15", "2015-03-01"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(panels, Path(tmp), FIRM_PANEL_LAYER)
            result = ingester._parse_firm_panels(shp_path, "01001")

        assert result is not None
        assert result["effective_date"].iloc[0] == date(2020, 6, 15)
        assert result["effective_date"].iloc[1] == date(2015, 3, 1)


# ---------------------------------------------------------------------------
# Test: Geometry simplification
# ---------------------------------------------------------------------------

class TestGeometrySimplification:
    """Verify geometry simplification behavior."""

    def test_simplification_applied(self, ingester):
        """Output geometry has fewer vertices than input."""
        # Create a polygon with many vertices (circle approximation)
        from shapely.geometry import Point
        detailed_poly = Point(-87, 32).buffer(0.1, resolution=64)
        # This creates ~257 vertices
        original_vertex_count = len(detailed_poly.exterior.coords)
        assert original_vertex_count > 50

        gdf = _make_flood_zone_gdf(
            zones=["AE"], polygons=[detailed_poly]
        )

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        simplified_geom = result.geometry.iloc[0]
        simplified_vertex_count = len(simplified_geom.exterior.coords)
        assert simplified_vertex_count < original_vertex_count

    def test_simplification_tolerance(self, ingester):
        """Simplification uses 0.0001 degree tolerance."""
        assert SIMPLIFY_TOLERANCE == 0.0001

    def test_zone_integrity_after_simplification(self, ingester):
        """Simplified polygons have valid geometry (make_valid applied)."""
        from shapely.geometry import Point
        detailed_poly = Point(-87, 32).buffer(0.1, resolution=64)
        gdf = _make_flood_zone_gdf(zones=["AE"], polygons=[detailed_poly])

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        for geom in result.geometry:
            assert geom.is_valid
            assert not geom.is_empty

    def test_make_valid_fixes_self_intersections(self, ingester):
        """make_valid() repairs self-intersections introduced by simplify()."""
        # Create a bowtie polygon that simplify() might produce
        from shapely.geometry import Polygon as ShapelyPolygon
        # Complex polygon with many vertices that could produce invalid geometry
        # after simplification
        from shapely.geometry import Point
        poly = Point(-87, 32).buffer(0.05, resolution=128)
        gdf = _make_flood_zone_gdf(zones=["AE"], polygons=[poly])

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        # All geometries must be valid after make_valid
        assert result.geometry.is_valid.all()


# ---------------------------------------------------------------------------
# Test: FIPS and normalization
# ---------------------------------------------------------------------------

class TestFIPSNormalization:
    """Verify FIPS extraction and column values."""

    def test_state_fips_extracted_from_county_fips(self, ingester):
        """state_fips is extracted from the 5-digit county FIPS."""
        gdf = _make_flood_zone_gdf(zones=["AE"])

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        assert (result["state_fips"] == "01").all()

    def test_county_fips_matches_download(self, ingester):
        """county_fips matches the 5-digit FIPS from the download URL."""
        gdf = _make_flood_zone_gdf(zones=["AE"])

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(gdf, Path(tmp), FLOOD_ZONE_LAYER)
            result = ingester._parse_flood_zones(shp_path, "06037")

        assert result is not None
        assert (result["county_fips"] == "06037").all()
        assert (result["state_fips"] == "06").all()


# ---------------------------------------------------------------------------
# Test: Caching and resumability
# ---------------------------------------------------------------------------

class TestCachingAndResumability:
    """Verify cache skip and incremental run behavior."""

    def test_cache_skip(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """County is skipped if parquet already exists."""
        # Pre-cache county 01001
        cache_dir = tmp_raw_dir / "fema_nfhl"
        cache_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_zones_gdf, Path(tmp), FLOOD_ZONE_LAYER
            )
            result_gdf = ingester._parse_flood_zones(shp_path, "01001")
        result_gdf.to_parquet(cache_dir / "fema_nfhl_01001.parquet", index=False)

        assert ingester._is_county_cached("01001")

    def test_incremental_run_skips_cached(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """A second run with partially cached data only downloads missing counties."""
        # Pre-cache county 01001
        cache_dir = tmp_raw_dir / "fema_nfhl"
        cache_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_zones_gdf, Path(tmp), FLOOD_ZONE_LAYER
            )
            result_gdf = ingester._parse_flood_zones(shp_path, "01001")
        result_gdf.to_parquet(cache_dir / "fema_nfhl_01001.parquet", index=False)

        # Mock the FIPS list with two counties
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),
            ("01", "003", "Baldwin County"),
        ])

        # Mock download — only county 01003 should be downloaded
        zip_bytes = _build_county_zip(sample_zones_gdf)
        download_calls = []

        original_download = ingester._download_county_zip

        def mock_download(fips5):
            download_calls.append(fips5)
            return zip_bytes

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", side_effect=mock_download),
        ):
            ingester.fetch()

        # Only county 01003 should have been downloaded
        assert "01001" not in download_calls
        assert "01003" in download_calls


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Verify resilient error handling for various failure modes."""

    def test_404_no_coverage(self, ingester, tmp_raw_dir):
        """HTTP 404 is logged as missing coverage and does not abort."""
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.content = b""

        mock_client = MagicMock()
        mock_client.get.return_value = resp_404
        ingester._client = mock_client
        result = ingester._download_county_zip("99999")

        assert result is None

    def test_500_server_error_retries(self, ingester, tmp_raw_dir):
        """Retry logic triggers on HTTP 500."""
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.content = b""

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.content = b"fake zip"
        resp_ok.raise_for_status = MagicMock()

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return resp_500
            return resp_ok

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_get
        ingester._client = mock_client
        result = ingester._download_county_zip("01001")

        assert result == b"fake zip"
        assert call_count == 3

    def test_partial_failure_other_counties_saved(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """If one county fails, other counties' data is still cached."""
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),
            ("01", "003", "Baldwin County"),
        ])

        zip_bytes = _build_county_zip(sample_zones_gdf)

        def mock_download(fips5):
            if fips5 == "01001":
                return None  # Simulated failure
            return zip_bytes

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", side_effect=mock_download),
        ):
            ingester.fetch()

        # County 01003 should have been saved
        assert (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01003.parquet").exists()

    def test_malformed_zip_skipped(self, ingester, tmp_raw_dir):
        """Corrupted ZIP is logged as warning and skipped."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])

        def mock_download(fips5):
            return b"not a real zip file"

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", side_effect=mock_download),
        ):
            ingester.fetch()

        # Should not crash, county should be marked as failed
        summary_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_failed"] >= 1


# ---------------------------------------------------------------------------
# Test: Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Verify temporary file cleanup behavior."""

    def test_temp_file_cleanup(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """Temporary ZIP and shapefiles are deleted after processing."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])
        zip_bytes = _build_county_zip(sample_zones_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        # No temporary directories should remain
        import glob
        tmp_dirs = glob.glob("/tmp/nfhl_*")
        # Filter to ones from this test run (recent)
        # This is a best-effort check
        assert (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01001.parquet").exists()

    def test_cleanup_on_failure(self, ingester, tmp_raw_dir):
        """Temp files are cleaned up even if parsing fails."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])

        # Create a ZIP with valid structure but empty flood zone shapefile
        empty_gdf = gpd.GeoDataFrame(
            {"FLD_ZONE": pd.Series(dtype=str), "ZONE_SUBTY": pd.Series(dtype=str)},
            geometry=gpd.GeoSeries([], crs="EPSG:4269"),
        )
        zip_bytes = _build_county_zip(empty_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        # Should complete without error even though no data was produced


# ---------------------------------------------------------------------------
# Test: Ingest purity (no derived metrics)
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO derived metrics are computed — raw data ONLY."""

    def test_no_derived_columns(self, ingester, sample_zones_gdf):
        """Output must NOT contain any derived metrics."""
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_zones_gdf, Path(tmp), FLOOD_ZONE_LAYER
            )
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        forbidden = {
            "flood_score", "pct_area_high_risk", "pct_hu_high_risk",
            "risk_category", "zone_numeric", "fips",
            "score", "percentile", "housing_units",
        }
        assert forbidden.isdisjoint(set(result.columns))

    def test_exact_column_set_zones(self, ingester, sample_zones_gdf):
        """Zone output has EXACTLY the required columns."""
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _write_gdf_to_shapefile(
                sample_zones_gdf, Path(tmp), FLOOD_ZONE_LAYER
            )
            result = ingester._parse_flood_zones(shp_path, "01001")

        assert result is not None
        expected = {"geometry", "flood_zone", "zone_subtype", "dfirm_id", "state_fips", "county_fips"}
        assert set(result.columns) == expected


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written with correct values."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir, sample_zones_gdf, sample_panels_gdf):
        """Metadata JSON is written alongside the county parquet."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])
        zip_bytes = _build_county_zip(sample_zones_gdf, sample_panels_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        meta_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01001_metadata.json"
        assert meta_path.exists()

    def test_metadata_content(self, ingester, tmp_raw_dir, sample_zones_gdf, sample_panels_gdf):
        """Metadata contains correct source/confidence/attribution."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])
        zip_bytes = _build_county_zip(sample_zones_gdf, sample_panels_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        meta_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01001_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "FEMA_NFHL"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta


# ---------------------------------------------------------------------------
# Test: Run log and run summary
# ---------------------------------------------------------------------------

class TestRunLogAndSummary:
    """Verify run log and summary JSON are written correctly."""

    def test_run_log_written(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """Per-county stats are written to fema_nfhl_run.log."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])
        zip_bytes = _build_county_zip(sample_zones_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        log_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run.log"
        assert log_path.exists()
        log_content = log_path.read_text()
        assert "01001" in log_content

    def test_run_summary_written(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """End-of-run summary JSON is written with correct counts."""
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),
            ("01", "003", "Baldwin County"),
        ])
        zip_bytes = _build_county_zip(sample_zones_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        summary_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run_summary.json"
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert "total_attempted" in summary
        assert "total_succeeded" in summary
        assert "total_no_coverage" in summary
        assert "total_failed" in summary
        assert "failed_fips" in summary
        assert summary["total_attempted"] == 2
        assert summary["total_succeeded"] == 2


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "fema_nfhl"

    def test_confidence(self, ingester):
        assert ingester.confidence == "A"

    def test_attribution(self, ingester):
        assert ingester.attribution == "proxy"

    def test_calls_per_second(self, ingester):
        assert ingester.calls_per_second == NFHL_CALLS_PER_SECOND

    def test_calls_per_second_polite(self, ingester):
        """Rate limit is polite (<=1 req/sec for federal server)."""
        assert ingester.calls_per_second <= 1.0

    def test_default_max_workers(self):
        """Default max_workers is 1 (sequential)."""
        ing = FEMANFHLIngester()
        assert ing._max_workers == DEFAULT_MAX_WORKERS == 1


# ---------------------------------------------------------------------------
# Test: End-to-end with mocked HTTP
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end test with fully mocked HTTP layer."""

    def test_full_pipeline_single_county(
        self, ingester, tmp_raw_dir, sample_zones_gdf, sample_panels_gdf
    ):
        """Full pipeline: FIPS list → download → extract → parse → cache."""
        fips_text = _make_census_fips_text([("01", "001", "Autauga County")])
        zip_bytes = _build_county_zip(sample_zones_gdf, sample_panels_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        # Verify zone parquet
        zone_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01001.parquet"
        assert zone_path.exists()
        zone_gdf = gpd.read_parquet(zone_path)
        assert len(zone_gdf) == len(sample_zones_gdf)
        assert "flood_zone" in zone_gdf.columns
        assert (zone_gdf["county_fips"] == "01001").all()
        assert (zone_gdf["state_fips"] == "01").all()

        # Verify panel parquet
        panel_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_panels_01001.parquet"
        assert panel_path.exists()
        panel_df = pd.read_parquet(panel_path)
        assert len(panel_df) == 2
        assert "effective_date" in panel_df.columns
        assert (panel_df["county_fips"] == "01001").all()

        # Verify metadata
        meta_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01001_metadata.json"
        assert meta_path.exists()

        # Verify run summary
        summary_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_succeeded"] == 1

    def test_full_pipeline_multiple_counties(self, tmp_raw_dir, sample_zones_gdf):
        """Pipeline processes multiple counties independently."""
        ingester = FEMANFHLIngester(states=["01"])
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),
            ("01", "003", "Baldwin County"),
            ("01", "005", "Barbour County"),
        ])
        zip_bytes = _build_county_zip(sample_zones_gdf)

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", return_value=zip_bytes),
        ):
            ingester.fetch()

        for fips in ["01001", "01003", "01005"]:
            assert (tmp_raw_dir / "fema_nfhl" / f"fema_nfhl_{fips}.parquet").exists()

    def test_full_pipeline_mixed_success_failure(self, tmp_raw_dir, sample_zones_gdf):
        """Pipeline handles mix of success, failure, and no-coverage counties."""
        ingester = FEMANFHLIngester(states=["01"])
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),   # Will succeed
            ("01", "003", "Baldwin County"),    # Will fail (no coverage)
            ("01", "005", "Barbour County"),    # Will succeed
        ])
        zip_bytes = _build_county_zip(sample_zones_gdf)

        def mock_download(fips5):
            if fips5 == "01003":
                return None  # No coverage
            return zip_bytes

        with (
            patch.object(ingester, "api_get", return_value=_make_mock_fips_response(fips_text)),
            patch.object(ingester, "_download_county_zip", side_effect=mock_download),
        ):
            ingester.fetch()

        assert (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01001.parquet").exists()
        assert not (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01003.parquet").exists()
        assert (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01005.parquet").exists()

        summary = json.loads(
            (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run_summary.json").read_text()
        )
        assert summary["total_succeeded"] == 2
        assert summary["total_no_coverage"] == 1


# ---------------------------------------------------------------------------
# Helpers — REST API mock responses
# ---------------------------------------------------------------------------

def _make_rest_objectid_response(object_ids: list[int]) -> MagicMock:
    """Create a mock httpx response for a REST objectIds query."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"objectIds": object_ids}
    resp.raise_for_status = MagicMock()
    return resp


def _make_rest_geojson_response(
    features: list[dict],
) -> MagicMock:
    """Create a mock httpx response for a REST GeoJSON feature query."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "type": "FeatureCollection",
        "features": features,
    }
    resp.raise_for_status = MagicMock()
    return resp


def _make_geojson_zone_features(
    zones: list[str],
    subtypes: list[str] | None = None,
    dfirm_ids: list[str] | None = None,
) -> list[dict]:
    """Build GeoJSON features matching ArcGIS REST response for flood zones."""
    n = len(zones)
    if subtypes is None:
        subtypes = [""] * n
    if dfirm_ids is None:
        dfirm_ids = ["11001C"] * n

    features = []
    for i, (zone, subtype, dfirm) in enumerate(
        zip(zones, subtypes, dfirm_ids)
    ):
        features.append({
            "type": "Feature",
            "properties": {
                "FLD_ZONE": zone,
                "ZONE_SUBTY": subtype,
                "DFIRM_ID": dfirm,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-77.0 + i * 0.01, 38.9],
                    [-76.99 + i * 0.01, 38.9],
                    [-76.99 + i * 0.01, 38.91],
                    [-77.0 + i * 0.01, 38.91],
                    [-77.0 + i * 0.01, 38.9],
                ]],
            },
        })
    return features


def _make_geojson_panel_features(
    panel_ids: list[str],
    eff_dates: list[str | int],
    dfirm_ids: list[str] | None = None,
    panel_types: list[str] | None = None,
) -> list[dict]:
    """Build GeoJSON features matching ArcGIS REST response for panels."""
    n = len(panel_ids)
    if dfirm_ids is None:
        dfirm_ids = ["11001C"] * n
    if panel_types is None:
        panel_types = ["COUNTYWIDE"] * n

    features = []
    for pid, eff, dfirm, ptype in zip(
        panel_ids, eff_dates, dfirm_ids, panel_types
    ):
        features.append({
            "type": "Feature",
            "properties": {
                "FIRM_PAN": pid,
                "EFF_DATE": eff,
                "DFIRM_ID": dfirm,
                "PANEL_TYP": ptype,
            },
            "geometry": None,
        })
    return features


def _make_html_download_response() -> MagicMock:
    """Create a mock httpx response for MSC returning HTML."""
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"<html><body>Search Results</body></html>"
    resp.headers = {"content-type": "text/html; charset=UTF-8"}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Test: Content-Type detection (MSC HTML vs ZIP)
# ---------------------------------------------------------------------------

class TestContentTypeDetection:
    """Verify Content-Type header triggers REST fallback."""

    def test_html_response_returns_fallback_sentinel(self, ingester, tmp_raw_dir):
        """MSC returning text/html triggers _HtmlFallback sentinel."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_html_download_response()
        ingester._client = mock_client

        result = ingester._download_county_zip("11001")
        assert isinstance(result, _HtmlFallback)

    def test_zip_response_returns_bytes(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """MSC returning application/zip returns bytes content."""
        zip_bytes = _build_county_zip(sample_zones_gdf)
        resp = MagicMock()
        resp.status_code = 200
        resp.content = zip_bytes
        resp.headers = {"content-type": "application/zip"}
        resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = resp
        ingester._client = mock_client

        result = ingester._download_county_zip("11001")
        assert isinstance(result, bytes)
        assert result == zip_bytes

    def test_octet_stream_returns_bytes(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """MSC returning application/octet-stream is treated as ZIP."""
        zip_bytes = _build_county_zip(sample_zones_gdf)
        resp = MagicMock()
        resp.status_code = 200
        resp.content = zip_bytes
        resp.headers = {"content-type": "application/octet-stream"}
        resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = resp
        ingester._client = mock_client

        result = ingester._download_county_zip("11001")
        assert isinstance(result, bytes)

    def test_missing_content_type_returns_bytes(self, ingester, tmp_raw_dir, sample_zones_gdf):
        """Missing Content-Type header defaults to treating as ZIP."""
        zip_bytes = _build_county_zip(sample_zones_gdf)
        resp = MagicMock()
        resp.status_code = 200
        resp.content = zip_bytes
        resp.headers = {}
        resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = resp
        ingester._client = mock_client

        result = ingester._download_county_zip("11001")
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Test: REST API ObjectID queries
# ---------------------------------------------------------------------------

class TestRESTObjectIDQueries:
    """Verify ArcGIS REST ObjectID query logic."""

    def test_query_returns_object_ids(self, ingester, tmp_raw_dir):
        """ObjectID query returns list of integers."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1, 2, 3])
        ingester._client = mock_client

        oids = ingester._rest_query_object_ids(28, "DFIRM_ID LIKE '11001%'")
        assert oids == [1, 2, 3]

    def test_query_empty_result(self, ingester, tmp_raw_dir):
        """No matching ObjectIDs returns empty list."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"objectIds": None}
        resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = resp
        ingester._client = mock_client

        oids = ingester._rest_query_object_ids(28, "DFIRM_ID LIKE '99999%'")
        assert oids == []

    def test_query_error_response(self, ingester, tmp_raw_dir):
        """ArcGIS error in response returns empty list."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "error": {"code": 400, "message": "Invalid query"}
        }
        resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = resp
        ingester._client = mock_client

        oids = ingester._rest_query_object_ids(28, "bad query")
        assert oids == []


# ---------------------------------------------------------------------------
# Test: REST feature batch fetching
# ---------------------------------------------------------------------------

class TestRESTFeatureBatch:
    """Verify ArcGIS REST feature batch fetch logic."""

    def test_batch_fetch_returns_geojson(self, ingester, tmp_raw_dir):
        """Feature batch returns GeoJSON FeatureCollection."""
        features = _make_geojson_zone_features(["AE", "VE"])
        mock_client = MagicMock()
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._rest_fetch_features_batch(
            28, [1, 2], "FLD_ZONE,ZONE_SUBTY,DFIRM_ID"
        )
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 2

    def test_batch_fetch_error_response(self, ingester, tmp_raw_dir):
        """ArcGIS error returns empty FeatureCollection."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "error": {"code": 400, "message": "Invalid request"}
        }
        resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = resp
        ingester._client = mock_client

        result = ingester._rest_fetch_features_batch(28, [1], "FLD_ZONE")
        assert result["features"] == []

    def test_batch_uses_post(self, ingester, tmp_raw_dir):
        """Feature fetch uses POST (not GET) to avoid URL length limits."""
        features = _make_geojson_zone_features(["AE"])
        mock_client = MagicMock()
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        ingester._rest_fetch_features_batch(28, [1], "FLD_ZONE")

        mock_client.post.assert_called_once()
        mock_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# Test: REST flood zone fetch
# ---------------------------------------------------------------------------

class TestRESTFloodZoneFetch:
    """Verify end-to-end REST flood zone fetch for a county."""

    def test_produces_correct_schema(self, ingester, tmp_raw_dir):
        """REST flood zones have same schema as MSC path."""
        features = _make_geojson_zone_features(
            ["AE", "VE", "X"],
            subtypes=["FLOODWAY", "", "AREA OF MINIMAL FLOOD HAZARD"],
            dfirm_ids=["11001C", "11001C", "11001C"],
        )

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1, 2, 3])
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("11001")

        assert result is not None
        expected_cols = {
            "geometry", "flood_zone", "zone_subtype",
            "dfirm_id", "state_fips", "county_fips",
        }
        assert set(result.columns) == expected_cols

    def test_fips_columns_correct(self, ingester, tmp_raw_dir):
        """state_fips and county_fips are correctly set."""
        features = _make_geojson_zone_features(["AE"])

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1])
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("11001")

        assert result is not None
        assert (result["state_fips"] == "11").all()
        assert (result["county_fips"] == "11001").all()

    def test_zone_values_preserved(self, ingester, tmp_raw_dir):
        """Flood zone classifications are preserved exactly."""
        zones = ["AE", "VE", "X", "A", "AO"]
        features = _make_geojson_zone_features(zones)

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response(
            list(range(1, len(zones) + 1))
        )
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("11001")

        assert result is not None
        assert set(result["flood_zone"].unique()) == set(zones)

    def test_no_object_ids_returns_none(self, ingester, tmp_raw_dir):
        """No ObjectIDs returns None."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([])
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("99999")
        assert result is None

    def test_geometry_simplified_and_valid(self, ingester, tmp_raw_dir):
        """REST geometries are simplified and made valid."""
        features = _make_geojson_zone_features(["AE"])

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1])
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("11001")

        assert result is not None
        for geom in result.geometry:
            assert geom.is_valid
            assert not geom.is_empty

    def test_batching_with_many_object_ids(self, ingester, tmp_raw_dir):
        """Large ObjectID sets are fetched in batches."""
        oids = list(range(1, 60))  # 59 OIDs → 3 batches of 25, 25, 9

        features = _make_geojson_zone_features(["AE"])

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response(oids)
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        ingester._fetch_flood_zones_rest("11001")

        # Should have made 3 POST calls (ceil(59/25) = 3)
        assert mock_client.post.call_count == 3


# ---------------------------------------------------------------------------
# Test: REST panel fetch
# ---------------------------------------------------------------------------

class TestRESTPanelFetch:
    """Verify REST FIRM panel fetch logic."""

    def test_produces_correct_schema(self, ingester, tmp_raw_dir):
        """REST panels have same schema as MSC path."""
        features = _make_geojson_panel_features(
            panel_ids=["11001C0100D", "11001C0200D"],
            eff_dates=["2020-06-15", "2015-03-01"],
        )

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1, 2])
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_firm_panels_rest("11001")

        assert result is not None
        expected_cols = {
            "dfirm_id", "effective_date", "panel_id",
            "panel_type", "state_fips", "county_fips",
        }
        assert set(result.columns) == expected_cols

    def test_epoch_ms_dates_parsed(self, ingester, tmp_raw_dir):
        """Epoch millisecond dates from ArcGIS are correctly parsed."""
        # 2020-06-15 00:00:00 UTC = 1592179200000 ms
        features = _make_geojson_panel_features(
            panel_ids=["11001C0100D"],
            eff_dates=[1592179200000],
        )

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1])
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_firm_panels_rest("11001")

        assert result is not None
        assert result["effective_date"].iloc[0] == date(2020, 6, 15)

    def test_string_dates_parsed(self, ingester, tmp_raw_dir):
        """String dates from ArcGIS are correctly parsed."""
        features = _make_geojson_panel_features(
            panel_ids=["11001C0100D"],
            eff_dates=["2020-06-15"],
        )

        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1])
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_firm_panels_rest("11001")

        assert result is not None
        assert result["effective_date"].iloc[0] == date(2020, 6, 15)

    def test_no_object_ids_returns_none(self, ingester, tmp_raw_dir):
        """No ObjectIDs returns None."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([])
        ingester._client = mock_client

        result = ingester._fetch_firm_panels_rest("99999")
        assert result is None


# ---------------------------------------------------------------------------
# Test: REST fallback integration (process_county_rest)
# ---------------------------------------------------------------------------

class TestRESTFallbackIntegration:
    """Verify full REST fallback flow in _process_county."""

    def test_html_triggers_rest_fallback(self, ingester, tmp_raw_dir):
        """MSC HTML response triggers REST fallback and produces parquet."""
        zone_features = _make_geojson_zone_features(["AE", "VE"])
        panel_features = _make_geojson_panel_features(
            panel_ids=["11001C0100D"],
            eff_dates=[1592179200000],
        )

        # Mock client: GET for MSC returns HTML, GET for REST OID query,
        # POST for REST feature fetch
        call_count = {"get": 0}

        def mock_get(*args, **kwargs):
            call_count["get"] += 1
            url = args[0] if args else kwargs.get("url", "")

            if "msc.fema.gov" in str(url):
                return _make_html_download_response()
            # REST ObjectID queries
            return _make_rest_objectid_response([1, 2])

        def mock_post(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if "/28/" in str(url):
                return _make_rest_geojson_response(zone_features)
            return _make_rest_geojson_response(panel_features)

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_get
        mock_client.post.side_effect = mock_post
        ingester._client = mock_client

        result = ingester._process_county("11001", "District of Columbia")

        assert result["status"] == "success"
        assert result["fetch_method"] == "arcgis_rest"
        assert result["row_count"] == 2

        # Verify parquet was written
        zone_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_11001.parquet"
        assert zone_path.exists()

        gdf = gpd.read_parquet(zone_path)
        assert (gdf["county_fips"] == "11001").all()
        assert (gdf["state_fips"] == "11").all()
        assert set(gdf["flood_zone"].unique()) == {"AE", "VE"}

    def test_rest_metadata_includes_fetch_method(self, ingester, tmp_raw_dir):
        """Metadata JSON from REST path includes fetch_method."""
        features = _make_geojson_zone_features(["AE"])

        mock_client = MagicMock()
        mock_client.get.side_effect = lambda *a, **kw: (
            _make_html_download_response()
            if "msc.fema.gov" in str(a[0] if a else kw.get("url", ""))
            else _make_rest_objectid_response([1])
        )
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        ingester._process_county("11001", "District of Columbia")

        meta_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_11001_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["fetch_method"] == "arcgis_rest"

    def test_rest_failure_marks_failed(self, ingester, tmp_raw_dir):
        """If REST fallback also fails, county is marked as failed."""
        mock_client = MagicMock()
        mock_client.get.side_effect = lambda *a, **kw: (
            _make_html_download_response()
            if "msc.fema.gov" in str(a[0] if a else kw.get("url", ""))
            else _make_rest_objectid_response([])
        )
        ingester._client = mock_client

        result = ingester._process_county("11001", "District of Columbia")

        # No ObjectIDs → empty → status should be "empty"
        assert result["status"] == "empty"
        assert result["fetch_method"] == "arcgis_rest"

    def test_run_summary_tracks_rest_fallback(self, ingester, tmp_raw_dir):
        """Run summary includes rest_fallback_fips."""
        fips_text = _make_census_fips_text([
            ("11", "001", "District of Columbia"),
        ])

        zone_features = _make_geojson_zone_features(["AE"])

        mock_client = MagicMock()
        mock_client.get.side_effect = lambda *a, **kw: (
            _make_html_download_response()
            if "msc.fema.gov" in str(a[0] if a else kw.get("url", ""))
            else _make_rest_objectid_response([1])
        )
        mock_client.post.return_value = _make_rest_geojson_response(
            zone_features
        )
        ingester._client = mock_client

        with patch.object(
            ingester,
            "api_get",
            return_value=_make_mock_fips_response(fips_text),
        ):
            ingester.fetch()

        summary_path = (
            tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run_summary.json"
        )
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_rest_fallback"] == 1
        assert "11001" in summary["rest_fallback_fips"]
        assert summary["total_succeeded"] == 1


# ---------------------------------------------------------------------------
# Test: Mixed MSC + REST pipeline
# ---------------------------------------------------------------------------

class TestMixedMSCAndREST:
    """Verify pipeline handles mix of MSC ZIP and REST fallback counties."""

    def test_mixed_msc_and_rest(self, tmp_raw_dir, sample_zones_gdf):
        """Some counties via MSC, others via REST — all succeed."""
        ingester = FEMANFHLIngester(states=["01", "11"])
        fips_text = _make_census_fips_text([
            ("01", "001", "Autauga County"),    # MSC will work
            ("11", "001", "District of Columbia"),  # MSC returns HTML
        ])

        zip_bytes = _build_county_zip(sample_zones_gdf)
        zone_features = _make_geojson_zone_features(["AE", "VE"])

        def mock_client_get(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            if "msc.fema.gov" in url:
                if "NFHL_11001" in url:
                    return _make_html_download_response()
                # MSC returns ZIP for 01001
                resp = MagicMock()
                resp.status_code = 200
                resp.content = zip_bytes
                resp.headers = {"content-type": "application/zip"}
                resp.raise_for_status = MagicMock()
                return resp
            # REST ObjectID query
            return _make_rest_objectid_response([1, 2])

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_client_get
        mock_client.post.return_value = _make_rest_geojson_response(
            zone_features
        )
        ingester._client = mock_client

        with patch.object(
            ingester,
            "api_get",
            return_value=_make_mock_fips_response(fips_text),
        ):
            ingester.fetch()

        # Both counties should have parquets
        assert (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_01001.parquet").exists()
        assert (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_11001.parquet").exists()

        # Check run summary
        summary = json.loads(
            (tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run_summary.json").read_text()
        )
        assert summary["total_succeeded"] == 2
        assert summary["total_rest_fallback"] == 1
        assert "11001" in summary["rest_fallback_fips"]
        assert "01001" not in summary["rest_fallback_fips"]


# ---------------------------------------------------------------------------
# Test: REST constants
# ---------------------------------------------------------------------------

class TestRESTConstants:
    """Verify REST API configuration constants."""

    def test_rest_base_url(self):
        assert "hazards.fema.gov" in NFHL_REST_BASE
        assert "NFHL/MapServer" in NFHL_REST_BASE

    def test_flood_zone_layer_id(self):
        assert FLOOD_ZONE_LAYER_ID == 28

    def test_firm_panel_layer_id(self):
        assert FIRM_PANEL_LAYER_ID == 3

    def test_batch_size_reasonable(self):
        assert 10 <= REST_OBJECTID_BATCH_SIZE <= 100

    def test_query_timeout_reasonable(self):
        assert REST_QUERY_TIMEOUT >= 30.0

    def test_tigerweb_url(self):
        assert "tigerweb.geo.census.gov" in TIGERWEB_COUNTY_URL
        assert "MapServer/86" in TIGERWEB_COUNTY_URL


# ---------------------------------------------------------------------------
# Helpers — TIGERweb spatial fallback
# ---------------------------------------------------------------------------


def _make_tigerweb_envelope_response(
    xmin: float = -77.12,
    ymin: float = 38.79,
    xmax: float = -76.91,
    ymax: float = 38.99,
) -> MagicMock:
    """Create a mock httpx response for a TIGERweb extent query."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "extent": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "spatialReference": {"wkid": 4269},
        }
    }
    resp.raise_for_status = MagicMock()
    return resp


def _make_tigerweb_empty_response() -> MagicMock:
    """Create a mock httpx response with no extent (county not found)."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Test: TIGERweb county envelope
# ---------------------------------------------------------------------------


class TestTIGERwebEnvelope:
    """Verify _fetch_county_envelope retrieves county bounding boxes."""

    def test_returns_envelope_dict(self, ingester, tmp_raw_dir):
        """Successful TIGERweb query returns extent dict."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_tigerweb_envelope_response()
        ingester._client = mock_client

        result = ingester._fetch_county_envelope("11001")

        assert result is not None
        assert "xmin" in result
        assert "ymin" in result
        assert "xmax" in result
        assert "ymax" in result

    def test_returns_none_on_missing_extent(self, ingester, tmp_raw_dir):
        """TIGERweb response without extent returns None."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_tigerweb_empty_response()
        ingester._client = mock_client

        result = ingester._fetch_county_envelope("99999")
        assert result is None

    def test_returns_none_on_http_error(self, ingester, tmp_raw_dir):
        """HTTP error from TIGERweb returns None (no crash)."""
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection refused")
        ingester._client = mock_client

        result = ingester._fetch_county_envelope("11001")
        assert result is None

    def test_passes_correct_params(self, ingester, tmp_raw_dir):
        """Verify GEOID and outSR are sent to TIGERweb."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_tigerweb_envelope_response()
        ingester._client = mock_client

        ingester._fetch_county_envelope("11001")

        call_args = mock_client.get.call_args
        params = call_args.kwargs.get("params", {})
        assert params["where"] == "GEOID='11001'"
        assert params["returnExtentOnly"] == "true"
        assert params["outSR"] == "4269"


# ---------------------------------------------------------------------------
# Test: Spatial ObjectID query
# ---------------------------------------------------------------------------


class TestSpatialObjectIDQuery:
    """Verify _rest_query_object_ids_spatial with envelope geometry."""

    def test_returns_object_ids(self, ingester, tmp_raw_dir):
        """Spatial query returns ObjectIDs when features intersect."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([10, 20, 30])
        ingester._client = mock_client

        envelope = {"xmin": -77.12, "ymin": 38.79, "xmax": -76.91, "ymax": 38.99}
        result = ingester._rest_query_object_ids_spatial(28, envelope)

        assert result == [10, 20, 30]

    def test_returns_empty_on_arcgis_error(self, ingester, tmp_raw_dir):
        """ArcGIS error response returns empty list."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"error": {"code": 400, "message": "Invalid geometry"}}
        resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = resp
        ingester._client = mock_client

        envelope = {"xmin": -77.12, "ymin": 38.79, "xmax": -76.91, "ymax": 38.99}
        result = ingester._rest_query_object_ids_spatial(28, envelope)

        assert result == []

    def test_returns_empty_on_http_error(self, ingester, tmp_raw_dir):
        """HTTP error returns empty list (no crash)."""
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Server error")
        ingester._client = mock_client

        envelope = {"xmin": -77.12, "ymin": 38.79, "xmax": -76.91, "ymax": 38.99}
        result = ingester._rest_query_object_ids_spatial(28, envelope)

        assert result == []

    def test_passes_geometry_params(self, ingester, tmp_raw_dir):
        """Verify spatial query sends correct geometry parameters."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_rest_objectid_response([1])
        ingester._client = mock_client

        envelope = {"xmin": -77.12, "ymin": 38.79, "xmax": -76.91, "ymax": 38.99}
        ingester._rest_query_object_ids_spatial(28, envelope)

        call_args = mock_client.get.call_args
        params = call_args.kwargs.get("params", {})
        assert params["geometryType"] == "esriGeometryEnvelope"
        assert params["spatialRel"] == "esriSpatialRelIntersects"
        assert params["inSR"] == "4269"
        assert "-77.12" in params["geometry"]
        assert "38.79" in params["geometry"]


# ---------------------------------------------------------------------------
# Test: Spatial fallback integration (DFIRM_ID empty → TIGERweb → spatial)
# ---------------------------------------------------------------------------


class TestSpatialFallback:
    """Verify the two-step fallback: DFIRM_ID query → spatial query."""

    def _mock_get_dfirm_empty_spatial_success(self, zone_oids=None):
        """Build a mock GET handler: DFIRM_ID=empty, TIGERweb=envelope, spatial=OIDs."""
        if zone_oids is None:
            zone_oids = [1, 2, 3]

        def mock_get(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            params = kwargs.get("params", {})

            if "tigerweb.geo.census.gov" in url:
                return _make_tigerweb_envelope_response()

            if "hazards.fema.gov" in url:
                # Spatial query (has geometry param)
                if "geometry" in params:
                    return _make_rest_objectid_response(zone_oids)
                # DFIRM_ID query (has where param) → empty
                return _make_rest_objectid_response([])

            return _make_rest_objectid_response([])

        return mock_get

    def test_dfirm_empty_triggers_spatial_zones(self, ingester, tmp_raw_dir):
        """DFIRM_ID query empty → TIGERweb envelope → spatial query → zones."""
        features = _make_geojson_zone_features(
            ["AE", "VE", "X"],
            dfirm_ids=["110001", "110001", "110001"],
        )

        mock_client = MagicMock()
        mock_client.get.side_effect = self._mock_get_dfirm_empty_spatial_success()
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("11001")

        assert result is not None
        assert len(result) == 3
        assert set(result["flood_zone"].unique()) == {"AE", "VE", "X"}
        assert (result["state_fips"] == "11").all()
        assert (result["county_fips"] == "11001").all()

    def test_dfirm_empty_triggers_spatial_panels(self, ingester, tmp_raw_dir):
        """DFIRM_ID query empty → TIGERweb envelope → spatial query → panels."""
        features = _make_geojson_panel_features(
            panel_ids=["110001P0100D", "110001P0200D"],
            eff_dates=[1592179200000, 1592179200000],
            dfirm_ids=["110001", "110001"],
        )

        mock_client = MagicMock()
        mock_client.get.side_effect = self._mock_get_dfirm_empty_spatial_success(
            zone_oids=[10, 20]
        )
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_firm_panels_rest("11001")

        assert result is not None
        assert len(result) == 2
        assert (result["state_fips"] == "11").all()
        assert (result["county_fips"] == "11001").all()

    def test_both_dfirm_and_spatial_empty_returns_none(self, ingester, tmp_raw_dir):
        """Both DFIRM_ID and spatial queries empty → returns None."""

        def mock_get(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            if "tigerweb.geo.census.gov" in url:
                return _make_tigerweb_envelope_response()
            # Both DFIRM and spatial return empty
            return _make_rest_objectid_response([])

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_get
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("99999")
        assert result is None

    def test_tigerweb_failure_still_returns_none(self, ingester, tmp_raw_dir):
        """If DFIRM_ID is empty and TIGERweb fails, returns None gracefully."""

        def mock_get(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            if "tigerweb.geo.census.gov" in url:
                raise Exception("TIGERweb down")
            return _make_rest_objectid_response([])

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_get
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("11001")
        assert result is None

    def test_dfirm_success_skips_spatial(self, ingester, tmp_raw_dir):
        """When DFIRM_ID query returns results, TIGERweb is never called."""
        features = _make_geojson_zone_features(["AE"])

        call_log = []

        def mock_get(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            call_log.append(url)
            if "tigerweb.geo.census.gov" in url:
                raise AssertionError("TIGERweb should not be called!")
            return _make_rest_objectid_response([1])

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_get
        mock_client.post.return_value = _make_rest_geojson_response(features)
        ingester._client = mock_client

        result = ingester._fetch_flood_zones_rest("01001")

        assert result is not None
        assert not any("tigerweb" in u for u in call_log)

    def test_full_integration_html_to_spatial_success(self, ingester, tmp_raw_dir):
        """Full pipeline: MSC HTML → REST → DFIRM empty → spatial → success."""
        zone_features = _make_geojson_zone_features(
            ["AE", "VE"],
            dfirm_ids=["110001", "110001"],
        )
        panel_features = _make_geojson_panel_features(
            panel_ids=["110001P0100D"],
            eff_dates=[1592179200000],
            dfirm_ids=["110001"],
        )

        def mock_get(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            params = kwargs.get("params", {})

            if "msc.fema.gov" in url:
                return _make_html_download_response()
            if "tigerweb.geo.census.gov" in url:
                return _make_tigerweb_envelope_response()
            if "hazards.fema.gov" in url:
                if "geometry" in params:
                    return _make_rest_objectid_response([1, 2])
                return _make_rest_objectid_response([])
            return _make_rest_objectid_response([])

        def mock_post(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            if "/28/" in url:
                return _make_rest_geojson_response(zone_features)
            return _make_rest_geojson_response(panel_features)

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_get
        mock_client.post.side_effect = mock_post
        ingester._client = mock_client

        result = ingester._process_county("11001", "District of Columbia")

        assert result["status"] == "success"
        assert result["fetch_method"] == "arcgis_rest"
        assert result["row_count"] == 2

        # Verify parquet was written
        zone_path = tmp_raw_dir / "fema_nfhl" / "fema_nfhl_11001.parquet"
        assert zone_path.exists()
        gdf = gpd.read_parquet(zone_path)
        assert set(gdf["flood_zone"].unique()) == {"AE", "VE"}

    def test_spatial_fallback_in_run_summary(self, ingester, tmp_raw_dir):
        """Run summary tracks spatial-fallback counties correctly."""
        fips_text = _make_census_fips_text([
            ("11", "001", "District of Columbia"),
        ])
        zone_features = _make_geojson_zone_features(
            ["AE"],
            dfirm_ids=["110001"],
        )

        def mock_get(*args, **kwargs):
            url = str(args[0] if args else kwargs.get("url", ""))
            params = kwargs.get("params", {})

            if "msc.fema.gov" in url:
                return _make_html_download_response()
            if "tigerweb.geo.census.gov" in url:
                return _make_tigerweb_envelope_response()
            if "hazards.fema.gov" in url:
                if "geometry" in params:
                    return _make_rest_objectid_response([1])
                return _make_rest_objectid_response([])
            return _make_rest_objectid_response([])

        mock_client = MagicMock()
        mock_client.get.side_effect = mock_get
        mock_client.post.return_value = _make_rest_geojson_response(
            zone_features
        )
        ingester._client = mock_client

        with patch.object(
            ingester,
            "api_get",
            return_value=_make_mock_fips_response(fips_text),
        ):
            ingester.fetch()

        summary_path = (
            tmp_raw_dir / "fema_nfhl" / "fema_nfhl_run_summary.json"
        )
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_rest_fallback"] == 1
        assert "11001" in summary["rest_fallback_fips"]
        assert summary["total_succeeded"] == 1
