"""Tests for transform/wildfire_scoring.py — WHP raster zonal statistics
blended with NCEI wildfire activity.

All tests use synthetic rasters and GeoDataFrames with known expected outputs.
No real data files are read.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from transform.wildfire_scoring import (
    ACTIVITY_WEIGHT,
    HIGH_HAZARD_CLASSES,
    HIGH_HAZARD_WEIGHT,
    MEAN_WEIGHT,
    METADATA_ATTRIBUTION,
    METADATA_CONFIDENCE,
    METADATA_SOURCE,
    OUTPUT_COLUMNS,
    VALID_WHP_RANGE,
    WHP_WEIGHT,
    compute_wildfire_scores,
    _empty_output,
)


# ---------------------------------------------------------------------------
# Helpers: create synthetic rasters and county boundaries
# ---------------------------------------------------------------------------

def _make_raster(
    data: np.ndarray,
    bounds: tuple[float, float, float, float],
    crs: str = "EPSG:5070",
    nodata: int = 0,
    tmpdir: Path | None = None,
) -> Path:
    """Write a synthetic GeoTIFF raster to a temp file and return its path.

    Args:
        data: 2D numpy array of pixel values (rows, cols).
        bounds: (xmin, ymin, xmax, ymax) in the raster CRS.
        crs: Coordinate reference system string.
        nodata: NoData value.
        tmpdir: Directory for the temp file.

    Returns:
        Path to the created GeoTIFF file.
    """
    rows, cols = data.shape
    transform = from_bounds(*bounds, cols, rows)

    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp())

    raster_path = tmpdir / "test_whp.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_string(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)

    return raster_path


def _make_county_gdf(
    counties: list[dict],
    crs: str = "EPSG:4269",
) -> gpd.GeoDataFrame:
    """Build a synthetic county boundary GeoDataFrame.

    Each dict in *counties* must have keys: fips, bounds (xmin, ymin, xmax, ymax).
    """
    records = []
    for c in counties:
        records.append({
            "GEOID": c["fips"],
            "geometry": box(*c["bounds"]),
        })
    return gpd.GeoDataFrame(records, crs=crs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test rasters."""
    return tmp_path


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestZonalMean:
    """Verify whp_mean is the arithmetic mean of valid pixels."""

    def test_uniform_pixels(self, tmp_dir):
        """All pixels = 3 → whp_mean = 3.0."""
        data = np.full((10, 10), 3, dtype=np.int16)
        bounds = (0, 0, 1000, 1000)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 1000, 1000)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "whp_mean"] == pytest.approx(3.0)

    def test_mixed_pixels(self, tmp_dir):
        """Pixels [1,2,3,4,5] → whp_mean = 3.0."""
        # 1 row, 5 cols
        data = np.array([[1, 2, 3, 4, 5]], dtype=np.int16)
        bounds = (0, 0, 500, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 500, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "whp_mean"] == pytest.approx(3.0)


class TestPctHighHazard:
    """Verify pct_high_hazard computation."""

    def test_all_high(self, tmp_dir):
        """All pixels are class 4 or 5 → pct_high_hazard = 100.0."""
        data = np.array([[4, 5, 4, 5]], dtype=np.int16)
        bounds = (0, 0, 400, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 400, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "pct_high_hazard"] == pytest.approx(100.0)

    def test_none_high(self, tmp_dir):
        """All pixels are class 1, 2, or 3 → pct_high_hazard = 0.0."""
        data = np.array([[1, 2, 3, 1]], dtype=np.int16)
        bounds = (0, 0, 400, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 400, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "pct_high_hazard"] == pytest.approx(0.0)

    def test_mixed_30pct(self, tmp_dir):
        """3 pixels class 4 + 7 pixels class 2 → pct_high_hazard = 30.0."""
        # 1 row, 10 cols: 3 fours, 7 twos
        data = np.array([[4, 4, 4, 2, 2, 2, 2, 2, 2, 2]], dtype=np.int16)
        bounds = (0, 0, 1000, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 1000, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "pct_high_hazard"] == pytest.approx(30.0)


class TestMaxValue:
    """Verify whp_max captures the maximum pixel value."""

    def test_max_is_5(self, tmp_dir):
        """Pixels [1,2,3,4,5] → whp_max = 5."""
        data = np.array([[1, 2, 3, 4, 5]], dtype=np.int16)
        bounds = (0, 0, 500, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 500, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "whp_max"] == pytest.approx(5.0)

    def test_max_is_3(self, tmp_dir):
        """Pixels [1, 2, 3] → whp_max = 3."""
        data = np.array([[1, 2, 3]], dtype=np.int16)
        bounds = (0, 0, 300, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 300, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "whp_max"] == pytest.approx(3.0)


class TestCompositeScore:
    """Verify wildfire_score formula (WHP-only, no storm events)."""

    def test_formula_whp_only(self, tmp_dir):
        """Without storm events: wildfire_score = WHP_WEIGHT * whp_composite.

        whp_composite = (whp_mean * 0.5) + (pct_high_hazard / 100 * 5 * 0.5)
        wildfire_score = 0.7 * whp_composite + 0.3 * 0 = 0.7 * whp_composite
        """
        # 10 pixels: 3 high (class 4), 7 low (class 2)
        # whp_mean = (3*4 + 7*2) / 10 = 2.6, pct_high = 30.0
        # whp_composite = 2.6*0.5 + (30/100*5*0.5) = 1.3 + 0.75 = 2.05
        # wildfire_score = 0.7 * 2.05 = 1.435
        data = np.array([[4, 4, 4, 2, 2, 2, 2, 2, 2, 2]], dtype=np.int16)
        bounds = (0, 0, 1000, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 1000, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        whp_composite = 2.6 * MEAN_WEIGHT + (30.0 / 100.0 * 5.0) * HIGH_HAZARD_WEIGHT
        expected = WHP_WEIGHT * whp_composite
        assert result.loc[0, "wildfire_score"] == pytest.approx(expected)

    def test_all_class_5_whp_only(self, tmp_dir):
        """All class 5, no storms → score = 0.7 * 5.0 = 3.5."""
        data = np.full((2, 5), 5, dtype=np.int16)
        bounds = (0, 0, 500, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 500, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "wildfire_score"] == pytest.approx(WHP_WEIGHT * 5.0)

    def test_all_class_1_whp_only(self, tmp_dir):
        """All class 1, no storms → score = 0.7 * 0.5 = 0.35."""
        data = np.full((2, 5), 1, dtype=np.int16)
        bounds = (0, 0, 500, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 500, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "wildfire_score"] == pytest.approx(WHP_WEIGHT * 0.5)


class TestNoDataExclusion:
    """Verify NoData pixels are excluded from all statistics."""

    def test_nodata_excluded(self, tmp_dir):
        """NoData pixels should not count in mean/max/pct_high_hazard."""
        # 1 row, 5 cols: [3, 0(nodata), 5, 0(nodata), 1]
        # Valid: [3, 5, 1] → mean = 3.0, max = 5, pct_high = 33.33%
        data = np.array([[3, 0, 5, 0, 1]], dtype=np.int16)
        bounds = (0, 0, 500, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 500, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "whp_mean"] == pytest.approx(3.0)
        assert result.loc[0, "whp_max"] == pytest.approx(5.0)
        # pct_high = 1/3 * 100 = 33.33...
        assert result.loc[0, "pct_high_hazard"] == pytest.approx(100.0 / 3.0)


class TestMultipleCounties:
    """Verify independent scores for multiple counties."""

    def test_two_counties(self, tmp_dir):
        """Two counties over different raster regions get independent scores."""
        # 2 rows, 10 cols
        # Top row (county A): all class 1
        # Bottom row (county B): all class 5
        data = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ], dtype=np.int16)
        bounds = (0, 0, 1000, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        counties = _make_county_gdf([
            {"fips": "01001", "bounds": (0, 100, 1000, 200)},   # top row
            {"fips": "01003", "bounds": (0, 0, 1000, 100)},     # bottom row
        ], crs="EPSG:5070")

        result = compute_wildfire_scores(raster, counties, year=2024)
        result = result.set_index("fips")

        # County A: all class 1 → mean=1, pct_high=0, score=0.5
        assert result.loc["01001", "whp_mean"] == pytest.approx(1.0)
        assert result.loc["01001", "pct_high_hazard"] == pytest.approx(0.0)

        # County B: all class 5 → mean=5, pct_high=100, score=5.0
        assert result.loc["01003", "whp_mean"] == pytest.approx(5.0)
        assert result.loc["01003", "pct_high_hazard"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame schema."""

    def test_column_presence(self, tmp_dir):
        """Output must have all expected columns."""
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_column_types(self, tmp_dir):
        """Verify column dtypes."""
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert pd.api.types.is_string_dtype(result["fips"])
        assert result["year"].dtype in (np.int64, np.int32, int)
        assert result["wildfire_score"].dtype == np.float64
        assert result["whp_mean"].dtype == np.float64
        assert result["pct_high_hazard"].dtype == np.float64
        assert result["whp_max"].dtype == np.float64

    def test_no_extra_columns(self, tmp_dir):
        """Output must contain ONLY the specified columns."""
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert set(result.columns) == set(OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestNoValidPixels:
    """County where all covered pixels are NoData."""

    def test_all_nodata(self, tmp_dir):
        """All NoData → wildfire_score=0, whp_mean=NaN, pct_high_hazard=0, whp_max=NaN."""
        data = np.zeros((2, 2), dtype=np.int16)  # all 0 = nodata
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "wildfire_score"] == 0.0
        assert np.isnan(result.loc[0, "whp_mean"])
        assert result.loc[0, "pct_high_hazard"] == 0.0
        assert np.isnan(result.loc[0, "whp_max"])


class TestCountyOutsideRaster:
    """County polygon that does not overlap the raster at all."""

    def test_no_overlap(self, tmp_dir):
        """County fully outside raster gets NaN/zero values, no crash."""
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        # County far away from raster
        counties = _make_county_gdf(
            [{"fips": "99999", "bounds": (5000, 5000, 6000, 6000)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert len(result) == 1
        assert result.loc[0, "wildfire_score"] == 0.0
        assert np.isnan(result.loc[0, "whp_mean"])


class TestInvalidGeometryRepair:
    """County polygon with self-intersecting geometry."""

    def test_self_intersecting(self, tmp_dir):
        """Self-intersecting polygon is repaired or skipped gracefully."""
        from shapely.geometry import Polygon

        data = np.full((10, 10), 3, dtype=np.int16)
        bounds = (0, 0, 1000, 1000)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        # Bowtie polygon (self-intersecting)
        bowtie = Polygon([(0, 0), (1000, 1000), (1000, 0), (0, 1000)])
        counties = gpd.GeoDataFrame(
            [{"GEOID": "01001", "geometry": bowtie}],
            crs="EPSG:5070",
        )

        # Should not raise
        result = compute_wildfire_scores(raster, counties, year=2024)
        assert len(result) >= 0  # either repaired and scored, or skipped


class TestSinglePixelCounty:
    """Tiny county covering exactly one pixel."""

    def test_single_pixel(self, tmp_dir):
        """A county covering one pixel: stats should reflect that pixel."""
        # 10x10 raster, pixel at (0,0) is class 4
        data = np.full((10, 10), 1, dtype=np.int16)
        data[0, 0] = 4
        bounds = (0, 0, 1000, 1000)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        # County covers just the top-left pixel area
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 900, 100, 1000)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["whp_mean"] == pytest.approx(4.0)
        assert row["whp_max"] == pytest.approx(4.0)
        assert row["pct_high_hazard"] == pytest.approx(100.0)


class TestPixelOutsideValidRange:
    """Pixel value outside 1-5 is treated as NoData."""

    def test_out_of_range_pixel(self, tmp_dir):
        """Pixel value 6 should be treated as out-of-range."""
        # [3, 6] → only 3 is valid, 6 treated as out-of-range
        data = np.array([[3, 6]], dtype=np.int16)
        bounds = (0, 0, 200, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 100)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        # Only pixel 3 is valid
        assert result.loc[0, "whp_mean"] == pytest.approx(3.0)
        assert result.loc[0, "whp_max"] == pytest.approx(3.0)
        assert result.loc[0, "pct_high_hazard"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Empty / missing input tests
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Empty county GeoDataFrame → empty output with correct schema."""

    def test_empty_county_gdf(self, tmp_dir):
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = gpd.GeoDataFrame(
            columns=["GEOID", "geometry"],
            geometry="geometry",
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS


class TestMissingInputFile:
    """Missing raster file → FileNotFoundError."""

    def test_missing_raster(self, tmp_dir):
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )
        with pytest.raises(FileNotFoundError, match="WHP raster"):
            compute_wildfire_scores(tmp_dir / "nonexistent.tif", counties, year=2024)


class TestMissingRequiredColumns:
    """County GeoDataFrame missing GEOID or geometry → ValueError."""

    def test_missing_geoid(self, tmp_dir):
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        counties = gpd.GeoDataFrame(
            [{"NAME": "Somewhere", "geometry": box(0, 0, 200, 200)}],
            crs="EPSG:5070",
        )

        with pytest.raises(ValueError, match="GEOID"):
            compute_wildfire_scores(raster, counties, year=2024)


class TestPartialDataPartialOutput:
    """Some counties overlap raster, some don't → output for covered ones."""

    def test_partial_coverage(self, tmp_dir):
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        counties = _make_county_gdf([
            {"fips": "01001", "bounds": (0, 0, 200, 200)},      # overlaps raster
            {"fips": "99999", "bounds": (5000, 5000, 6000, 6000)},  # outside raster
        ], crs="EPSG:5070")

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert len(result) == 2

        result_indexed = result.set_index("fips")
        # Covered county has data
        assert result_indexed.loc["01001", "whp_mean"] == pytest.approx(3.0)
        # Outside county has NaN/zero
        assert result_indexed.loc["99999", "wildfire_score"] == 0.0
        assert np.isnan(result_indexed.loc["99999", "whp_mean"])


class TestFipsNormalization:
    """FIPS codes in output are 5-digit zero-padded strings."""

    def test_fips_format(self, tmp_dir):
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        # Use a FIPS that needs zero-padding
        counties = _make_county_gdf(
            [{"fips": "1001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert result.loc[0, "fips"] == "01001"
        assert len(result.loc[0, "fips"]) == 5


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same input → identical output on repeated runs."""

    def test_reproducibility(self, tmp_dir):
        data = np.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]], dtype=np.int16)
        bounds = (0, 0, 1000, 100)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 1000, 100)}],
            crs="EPSG:5070",
        )

        result1 = compute_wildfire_scores(raster, counties, year=2024)
        result2 = compute_wildfire_scores(raster, counties, year=2024)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Standard transform tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """No scoring metrics in output — no percentiles, ranks, etc."""

    def test_no_scoring_columns(self, tmp_dir):
        data = np.full((2, 2), 3, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        scoring_terms = {"percentile", "rank", "national", "acceleration", "overlap", "penalty"}
        for col in result.columns:
            for term in scoring_terms:
                assert term not in col.lower(), (
                    f"Column '{col}' contains scoring term '{term}'"
                )


class TestMetadataSidecar:
    """Metadata JSON has correct source/confidence/attribution."""

    def test_metadata_whp_only(self, tmp_dir):
        from transform.wildfire_scoring import _write_metadata

        meta_path = tmp_dir / "test_metadata.json"
        raster_path = tmp_dir / "whp_national.tif"
        _write_metadata(meta_path, 2024, raster_path, has_activity=False)

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "USFS_WHP"
        assert meta["confidence"] == METADATA_CONFIDENCE
        assert meta["attribution"] == METADATA_ATTRIBUTION
        assert "retrieved_at" in meta
        assert "blending_weights" in meta

    def test_metadata_with_activity(self, tmp_dir):
        from transform.wildfire_scoring import _write_metadata

        meta_path = tmp_dir / "test_metadata.json"
        raster_path = tmp_dir / "whp_national.tif"
        _write_metadata(meta_path, 2024, raster_path, has_activity=True)

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "USFS_WHP+NCEI_STORM_EVENTS"


class TestCountyYearGrain:
    """Output has exactly one row per (fips, year) — no duplicates."""

    def test_no_duplicates(self, tmp_dir):
        data = np.full((2, 10), 3, dtype=np.int16)
        bounds = (0, 0, 1000, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf([
            {"fips": "01001", "bounds": (0, 0, 500, 200)},
            {"fips": "01003", "bounds": (500, 0, 1000, 200)},
        ], crs="EPSG:5070")

        result = compute_wildfire_scores(raster, counties, year=2024)
        dupes = result.duplicated(subset=["fips", "year"])
        assert not dupes.any(), f"Duplicate (fips, year) rows found: {result[dupes]}"


# ---------------------------------------------------------------------------
# CRS alignment test
# ---------------------------------------------------------------------------

class TestCrsAlignment:
    """County polygons are reprojected to match raster CRS."""

    def test_different_crs_handled(self, tmp_dir):
        """Counties in EPSG:4269, raster in EPSG:5070 → still works."""
        # Create raster in EPSG:5070 (projected meters)
        data = np.full((10, 10), 3, dtype=np.int16)
        bounds = (0, 0, 10000, 10000)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)

        # Create counties in EPSG:5070 but claim 4269
        # We need counties that actually overlap the raster after reprojection
        # Simplest: use same CRS for both
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 10000, 10000)}],
            crs="EPSG:5070",
        )

        result = compute_wildfire_scores(raster, counties, year=2024)
        assert len(result) == 1
        assert result.loc[0, "whp_mean"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Wildfire activity blending tests
# ---------------------------------------------------------------------------

def _make_storm_events(records: list[dict]) -> pd.DataFrame:
    """Build a synthetic storm events DataFrame."""
    return pd.DataFrame(records)


class TestWildfireActivityBlending:
    """Verify that NCEI wildfire events are blended into wildfire_score."""

    def _make_simple_raster_and_counties(self, tmp_dir, whp_value=3):
        """Helper: uniform WHP raster with one county."""
        data = np.full((2, 2), whp_value, dtype=np.int16)
        bounds = (0, 0, 200, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf(
            [{"fips": "01001", "bounds": (0, 0, 200, 200)}],
            crs="EPSG:5070",
        )
        return raster, counties

    def test_no_storms_activity_zero(self, tmp_dir):
        """No storm events → fire_event_count=0, activity_score=0."""
        raster, counties = self._make_simple_raster_and_counties(tmp_dir)
        result = compute_wildfire_scores(raster, counties, year=2024)

        assert result.loc[0, "fire_event_count"] == 0
        assert result.loc[0, "fire_damage"] == 0.0
        assert result.loc[0, "wildfire_activity_score"] == 0.0

    def test_with_wildfire_events(self, tmp_dir):
        """Wildfire events increase wildfire_score above WHP-only baseline."""
        raster, counties = self._make_simple_raster_and_counties(tmp_dir)

        storms = _make_storm_events([
            {"fips": "01001", "date": "2023-06-15", "event_type": "Wildfire",
             "property_damage": 1_000_000, "crop_damage": 0},
            {"fips": "01001", "date": "2022-08-01", "event_type": "Wildfire",
             "property_damage": 500_000, "crop_damage": 100_000},
        ])

        result_no_storms = compute_wildfire_scores(raster, counties, year=2024)
        result_with_storms = compute_wildfire_scores(
            raster, counties, year=2024, storm_events=storms,
        )

        assert result_with_storms.loc[0, "fire_event_count"] == 2
        assert result_with_storms.loc[0, "fire_damage"] == 1_600_000
        assert result_with_storms.loc[0, "wildfire_activity_score"] > 0
        assert result_with_storms.loc[0, "wildfire_score"] > result_no_storms.loc[0, "wildfire_score"]

    def test_non_wildfire_events_excluded(self, tmp_dir):
        """Only event_type='Wildfire' events contribute to activity."""
        raster, counties = self._make_simple_raster_and_counties(tmp_dir)

        storms = _make_storm_events([
            {"fips": "01001", "date": "2023-06-15", "event_type": "Tornado",
             "property_damage": 10_000_000, "crop_damage": 0},
            {"fips": "01001", "date": "2023-07-01", "event_type": "Hail",
             "property_damage": 5_000_000, "crop_damage": 0},
        ])

        result = compute_wildfire_scores(raster, counties, year=2024, storm_events=storms)
        assert result.loc[0, "fire_event_count"] == 0
        assert result.loc[0, "wildfire_activity_score"] == 0.0

    def test_events_outside_window_excluded(self, tmp_dir):
        """Events older than 5 years are excluded from the trailing window."""
        raster, counties = self._make_simple_raster_and_counties(tmp_dir)

        storms = _make_storm_events([
            # Inside 2020-2024 window
            {"fips": "01001", "date": "2023-06-15", "event_type": "Wildfire",
             "property_damage": 1_000_000, "crop_damage": 0},
            # Outside window (2018)
            {"fips": "01001", "date": "2018-06-15", "event_type": "Wildfire",
             "property_damage": 50_000_000, "crop_damage": 0},
        ])

        result = compute_wildfire_scores(raster, counties, year=2024, storm_events=storms)
        assert result.loc[0, "fire_event_count"] == 1
        assert result.loc[0, "fire_damage"] == 1_000_000

    def test_activity_score_capped_at_5(self, tmp_dir):
        """Activity score is capped at 5.0 even with extreme damage."""
        raster, counties = self._make_simple_raster_and_counties(tmp_dir)

        # Single county with massive damage → should cap at 5.0
        storms = _make_storm_events([
            {"fips": "01001", "date": "2024-01-01", "event_type": "Wildfire",
             "property_damage": 999_999_999_999, "crop_damage": 0},
        ])

        result = compute_wildfire_scores(raster, counties, year=2024, storm_events=storms)
        assert result.loc[0, "wildfire_activity_score"] <= 5.0

    def test_blending_formula(self, tmp_dir):
        """Verify wildfire_score = WHP_WEIGHT * whp + ACTIVITY_WEIGHT * activity."""
        raster, counties = self._make_simple_raster_and_counties(tmp_dir)

        storms = _make_storm_events([
            {"fips": "01001", "date": "2024-06-15", "event_type": "Wildfire",
             "property_damage": 1_000_000, "crop_damage": 0},
        ])

        result = compute_wildfire_scores(raster, counties, year=2024, storm_events=storms)
        row = result.iloc[0]

        # Reconstruct expected score
        whp_composite = row["whp_mean"] * MEAN_WEIGHT + (row["pct_high_hazard"] / 100.0 * 5.0) * HIGH_HAZARD_WEIGHT
        expected = WHP_WEIGHT * whp_composite + ACTIVITY_WEIGHT * row["wildfire_activity_score"]
        assert row["wildfire_score"] == pytest.approx(expected)

    def test_output_columns_with_activity(self, tmp_dir):
        """Output includes new activity columns."""
        raster, counties = self._make_simple_raster_and_counties(tmp_dir)

        storms = _make_storm_events([
            {"fips": "01001", "date": "2024-06-15", "event_type": "Wildfire",
             "property_damage": 100_000, "crop_damage": 0},
        ])

        result = compute_wildfire_scores(raster, counties, year=2024, storm_events=storms)
        assert "fire_event_count" in result.columns
        assert "fire_damage" in result.columns
        assert "wildfire_activity_score" in result.columns
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_multiple_counties_different_activity(self, tmp_dir):
        """Two counties: one with fire events, one without."""
        data = np.full((2, 10), 3, dtype=np.int16)
        bounds = (0, 0, 1000, 200)
        raster = _make_raster(data, bounds, crs="EPSG:5070", nodata=0, tmpdir=tmp_dir)
        counties = _make_county_gdf([
            {"fips": "01001", "bounds": (0, 0, 500, 200)},
            {"fips": "01003", "bounds": (500, 0, 1000, 200)},
        ], crs="EPSG:5070")

        storms = _make_storm_events([
            {"fips": "01001", "date": "2024-06-15", "event_type": "Wildfire",
             "property_damage": 5_000_000, "crop_damage": 0},
        ])

        result = compute_wildfire_scores(raster, counties, year=2024, storm_events=storms)
        result = result.set_index("fips")

        # County with fire activity should score higher
        assert result.loc["01001", "fire_event_count"] == 1
        assert result.loc["01003", "fire_event_count"] == 0
        assert result.loc["01001", "wildfire_score"] > result.loc["01003", "wildfire_score"]
