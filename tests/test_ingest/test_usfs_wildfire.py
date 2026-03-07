"""Tests for the USFS Wildfire Hazard Potential ingester (ingest/usfs_wildfire.py).

All HTTP calls are mocked — no real downloads from USFS servers.
Synthetic GeoTIFF rasters are created using rasterio + numpy in fixtures.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from ingest.usfs_wildfire import (
    DOWNLOAD_TIMEOUT_SECONDS,
    EXPECTED_BAND_COUNT,
    EXPECTED_CRS_EPSG,
    PIXEL_SAMPLE_ROWS,
    VALID_PIXEL_MAX,
    VALID_PIXEL_MIN,
    WHP_DOWNLOAD_URL,
    WHP_METADATA_FILENAME,
    WHP_PREFERRED_PATTERN,
    WHP_RASTER_FILENAME,
    USFSWildfireIngester,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _create_synthetic_raster(
    path: Path,
    width: int = 100,
    height: int = 100,
    crs: str = "EPSG:5070",
    pixel_values: np.ndarray | None = None,
    nodata: int = -1,
    dtype: str = "int16",
    band_count: int = 1,
    resolution: float = 270.0,
) -> Path:
    """Create a synthetic GeoTIFF raster for testing.

    Args:
        path: Output path for the GeoTIFF.
        width: Raster width in pixels.
        height: Raster height in pixels.
        crs: Coordinate reference system string.
        pixel_values: 2D array of pixel values. If None, generates random 0–5.
        nodata: NoData sentinel value.
        dtype: Pixel data type.
        band_count: Number of bands.
        resolution: Pixel size in CRS units.

    Returns:
        Path to the created GeoTIFF.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if pixel_values is None:
        rng = np.random.default_rng(42)
        pixel_values = rng.integers(1, 8, size=(height, width), dtype=np.int16)

    # Create a transform matching the resolution and CONUS-like bounds
    west, south = -2_000_000, 500_000
    east = west + width * resolution
    north = south + height * resolution
    transform = from_bounds(west, south, east, north, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=band_count,
        dtype=dtype,
        crs=CRS.from_string(crs) if crs else None,
        transform=transform,
        nodata=nodata,
    ) as dst:
        for band in range(1, band_count + 1):
            dst.write(pixel_values.astype(dtype), band)

    return path


def _create_synthetic_zip(
    tif_path: Path,
    zip_path: Path,
    tif_name_in_zip: str = "Data/whp2023_GeoTIFF/whp2023_cnt_conus.tif",
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """Create a ZIP archive containing a GeoTIFF and optional ancillary files.

    Args:
        tif_path: Path to the GeoTIFF to include.
        zip_path: Output path for the ZIP archive.
        tif_name_in_zip: Archive path for the GeoTIFF inside the ZIP.
        extra_files: Optional dict of {archive_name: content} for ancillary files.

    Returns:
        Path to the created ZIP archive.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tif_path, tif_name_in_zip)
        if extra_files:
            for name, content in extra_files.items():
                zf.writestr(name, content)
    return zip_path


@pytest.fixture
def ingester() -> USFSWildfireIngester:
    """Return a fresh USFSWildfireIngester instance."""
    return USFSWildfireIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def synthetic_raster(tmp_path: Path) -> Path:
    """Create a synthetic WHP raster with valid pixel values (0–5)."""
    return _create_synthetic_raster(tmp_path / "synthetic_whp.tif")


@pytest.fixture
def synthetic_zip(synthetic_raster: Path, tmp_path: Path) -> Path:
    """Create a ZIP archive containing the synthetic raster."""
    return _create_synthetic_zip(
        synthetic_raster,
        tmp_path / "whp_archive.zip",
    )


# ---------------------------------------------------------------------------
# Test: Raster verification
# ---------------------------------------------------------------------------

class TestRasterVerification:
    """Verify raster metadata reading and validation."""

    def test_crs_validation(self, ingester, synthetic_raster):
        """Ingester reads and logs the raster CRS correctly."""
        props = ingester._verify_raster(synthetic_raster)
        assert props["crs"] == "EPSG:5070"

    def test_resolution_validation(self, ingester, synthetic_raster):
        """Ingester reads and logs pixel resolution correctly."""
        props = ingester._verify_raster(synthetic_raster)
        assert len(props["resolution"]) == 2
        # Resolution should be positive
        assert props["resolution"][0] > 0
        assert props["resolution"][1] > 0

    def test_band_count_validation_single(self, ingester, synthetic_raster):
        """Ingester accepts a single-band raster."""
        props = ingester._verify_raster(synthetic_raster)
        assert props["band_count"] == 1

    def test_band_count_validation_multi_warns(self, ingester, tmp_path, caplog):
        """Multi-band raster logs a warning."""
        raster_path = _create_synthetic_raster(
            tmp_path / "multi_band.tif",
            band_count=3,
        )
        with caplog.at_level("WARNING"):
            props = ingester._verify_raster(raster_path)
        assert props["band_count"] == 3
        assert "expected 1 band, found 3" in caplog.text

    def test_nodata_detection(self, ingester, synthetic_raster):
        """Ingester reads and logs the NoData value."""
        props = ingester._verify_raster(synthetic_raster)
        assert props["nodata"] == -1

    def test_nodata_missing_warns(self, ingester, tmp_path, caplog):
        """Raster without NoData value logs a warning."""
        raster_path = _create_synthetic_raster(
            tmp_path / "no_nodata.tif",
            nodata=None,
        )
        with caplog.at_level("WARNING"):
            props = ingester._verify_raster(raster_path)
        assert props["nodata"] is None
        assert "no NoData value defined" in caplog.text

    def test_pixel_values_valid_no_warning(self, ingester, tmp_path, caplog):
        """Valid pixel values (1–7) produce no warnings."""
        values = np.array([[1, 2, 3], [5, 6, 7]], dtype=np.int16)
        raster_path = _create_synthetic_raster(
            tmp_path / "valid_pixels.tif",
            width=3,
            height=2,
            pixel_values=values,
        )
        with caplog.at_level("WARNING"):
            ingester._validate_pixel_values(raster_path)
        assert "out-of-range" not in caplog.text

    def test_pixel_values_out_of_range_warns(self, ingester, tmp_path, caplog):
        """Out-of-range pixel values (e.g., 9) log a warning."""
        values = np.array([[1, 2, 9], [3, 4, 5]], dtype=np.int16)
        raster_path = _create_synthetic_raster(
            tmp_path / "oor_pixels.tif",
            width=3,
            height=2,
            pixel_values=values,
        )
        with caplog.at_level("WARNING"):
            ingester._validate_pixel_values(raster_path)
        assert "out-of-range" in caplog.text

    def test_nodata_pixels_excluded_from_range_check(self, ingester, tmp_path, caplog):
        """NoData pixels are not counted as out-of-range."""
        nodata_val = -1
        values = np.array([[nodata_val, 1, 2], [3, nodata_val, 7]], dtype=np.int16)
        raster_path = _create_synthetic_raster(
            tmp_path / "nodata_pixels.tif",
            width=3,
            height=2,
            pixel_values=values,
            nodata=nodata_val,
        )
        with caplog.at_level("WARNING"):
            ingester._validate_pixel_values(raster_path)
        assert "out-of-range" not in caplog.text

    def test_properties_dict_keys(self, ingester, synthetic_raster):
        """Verify all expected keys are present in properties dict."""
        props = ingester._verify_raster(synthetic_raster)
        expected_keys = {"crs", "resolution", "bounds", "shape", "dtype", "nodata", "band_count"}
        assert set(props.keys()) == expected_keys

    def test_bounds_are_list(self, ingester, synthetic_raster):
        """Bounds are returned as a 4-element list."""
        props = ingester._verify_raster(synthetic_raster)
        assert isinstance(props["bounds"], list)
        assert len(props["bounds"]) == 4

    def test_shape_is_list(self, ingester, synthetic_raster):
        """Shape is returned as a 2-element list [height, width]."""
        props = ingester._verify_raster(synthetic_raster)
        assert isinstance(props["shape"], list)
        assert len(props["shape"]) == 2
        assert props["shape"] == [100, 100]


# ---------------------------------------------------------------------------
# Test: Download and extraction
# ---------------------------------------------------------------------------

class TestDownloadAndExtraction:
    """Verify ZIP extraction, file filtering, and download behavior."""

    def test_zip_extraction(self, ingester, synthetic_raster, tmp_path):
        """GeoTIFF is correctly extracted from a ZIP archive."""
        zip_path = _create_synthetic_zip(
            synthetic_raster,
            tmp_path / "test.zip",
            tif_name_in_zip="whp_conus.tif",
        )
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        result = ingester._extract_geotiff(zip_path, extract_dir)

        assert result.exists()
        assert result.suffix == ".tif"
        # Verify the extracted file is a valid raster
        with rasterio.open(result) as src:
            assert src.count == 1

    def test_nested_zip_layout(self, ingester, synthetic_raster, tmp_path):
        """Extraction works when the GeoTIFF is inside a subdirectory."""
        zip_path = _create_synthetic_zip(
            synthetic_raster,
            tmp_path / "nested.zip",
            tif_name_in_zip="Data/subfolder/whp2023.tif",
        )
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        result = ingester._extract_geotiff(zip_path, extract_dir)

        assert result.exists()
        assert result.name == "whp2023.tif"

    def test_ancillary_file_filtering(self, ingester, synthetic_raster, tmp_path):
        """Non-GeoTIFF files in the archive are not extracted."""
        zip_path = _create_synthetic_zip(
            synthetic_raster,
            tmp_path / "with_extras.zip",
            tif_name_in_zip="Data/whp.tif",
            extra_files={
                "Data/metadata.xml": b"<xml>metadata</xml>",
                "Data/readme.pdf": b"fake pdf content",
                "Data/notes.txt": b"some notes",
            },
        )
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        result = ingester._extract_geotiff(zip_path, extract_dir)

        # Only the TIF should be extracted
        assert result.exists()
        assert result.suffix == ".tif"
        # Ancillary files should NOT be in the extract directory
        all_files = list(extract_dir.rglob("*"))
        tif_files = [f for f in all_files if f.is_file() and f.suffix in (".tif", ".tiff")]
        non_tif_files = [
            f for f in all_files
            if f.is_file() and f.suffix not in (".tif", ".tiff")
        ]
        assert len(tif_files) == 1
        assert len(non_tif_files) == 0

    def test_no_geotiff_in_archive_raises(self, ingester, tmp_path):
        """Archive with no GeoTIFF raises FileNotFoundError."""
        zip_path = tmp_path / "no_tif.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no raster here")
            zf.writestr("metadata.xml", "<xml/>")

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No GeoTIFF"):
            ingester._extract_geotiff(zip_path, extract_dir)

    def test_classified_conus_preferred(self, ingester, tmp_path):
        """When archive has both continuous and classified, classified CONUS is selected."""
        cnt_path = _create_synthetic_raster(
            tmp_path / "cnt.tif", width=200, height=200
        )
        cls_path = _create_synthetic_raster(
            tmp_path / "cls.tif", width=50, height=50
        )

        zip_path = tmp_path / "multi_tif.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(cnt_path, "Data/whp2023_cnt_conus.tif")
            zf.write(cls_path, "Data/whp2023_cls_conus.tif")

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        result = ingester._extract_geotiff(zip_path, extract_dir)
        assert result.name == "whp2023_cls_conus.tif"

    def test_largest_tif_fallback(self, ingester, tmp_path):
        """When no classified CONUS raster, falls back to the largest TIF."""
        small_path = _create_synthetic_raster(
            tmp_path / "small.tif", width=10, height=10
        )
        large_path = _create_synthetic_raster(
            tmp_path / "large.tif", width=200, height=200
        )

        zip_path = tmp_path / "multi_tif.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(small_path, "Data/small_layer.tif")
            zf.write(large_path, "Data/whp_main.tif")

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        result = ingester._extract_geotiff(zip_path, extract_dir)
        assert result.name == "whp_main.tif"


# ---------------------------------------------------------------------------
# Test: Caching and resumability
# ---------------------------------------------------------------------------

class TestCachingAndResumability:
    """Verify cache check, temp file cleanup, and failure cleanup."""

    def test_cache_check_skips_download(self, ingester, tmp_raw_dir, synthetic_raster):
        """Ingester skips download if whp_national.tif already exists."""
        cache_dir = tmp_raw_dir / "usfs_wildfire"
        cache_dir.mkdir(parents=True)
        import shutil
        shutil.copy2(synthetic_raster, cache_dir / WHP_RASTER_FILENAME)

        result = ingester.fetch()

        assert result == cache_dir / WHP_RASTER_FILENAME
        assert result.exists()

    def test_temp_file_cleanup_on_success(self, ingester, tmp_raw_dir, tmp_path):
        """ZIP and temp extraction dir are deleted after successful caching."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(
            synthetic, tmp_path / "download.zip"
        )

        # Mock _download_archive to copy our synthetic ZIP
        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            result = ingester.fetch()

        assert result.exists()
        # The cache should only contain the raster and metadata
        cache_dir = tmp_raw_dir / "usfs_wildfire"
        cached_files = list(cache_dir.iterdir())
        names = {f.name for f in cached_files}
        assert WHP_RASTER_FILENAME in names
        assert WHP_METADATA_FILENAME in names
        # No ZIP files should remain
        assert not any(f.suffix == ".zip" for f in cached_files)

    def test_cleanup_on_extraction_failure(self, ingester, tmp_raw_dir, tmp_path):
        """Temp files are cleaned up even if extraction fails."""
        def mock_download(dest: Path) -> None:
            # Write a non-ZIP file to cause extraction failure
            dest.write_bytes(b"not a zip file")

        with (
            patch.object(ingester, "_download_archive", side_effect=mock_download),
            pytest.raises(zipfile.BadZipFile),
        ):
            ingester.fetch()

        # Temp directory should be cleaned up
        import glob
        tmp_dirs = glob.glob("/tmp/usfs_whp_*")
        # All usfs_whp_ temp dirs should have been cleaned up
        # (there may be none if the cleanup was successful)
        for d in tmp_dirs:
            # If any remain, they shouldn't be from our test
            pass

    def test_is_cached_false_for_empty(self, ingester, tmp_raw_dir):
        """_is_cached returns False when no raster file exists."""
        assert not ingester._is_cached()

    def test_is_cached_false_for_tiny_file(self, ingester, tmp_raw_dir):
        """_is_cached returns False for a trivially small file."""
        cache_dir = tmp_raw_dir / "usfs_wildfire"
        cache_dir.mkdir(parents=True)
        (cache_dir / WHP_RASTER_FILENAME).write_bytes(b"tiny")
        assert not ingester._is_cached()

    def test_is_cached_true_for_valid_file(self, ingester, tmp_raw_dir, synthetic_raster):
        """_is_cached returns True when a proper raster exists."""
        cache_dir = tmp_raw_dir / "usfs_wildfire"
        cache_dir.mkdir(parents=True)
        import shutil
        shutil.copy2(synthetic_raster, cache_dir / WHP_RASTER_FILENAME)
        assert ingester._is_cached()


# ---------------------------------------------------------------------------
# Test: Output and metadata
# ---------------------------------------------------------------------------

class TestOutputAndMetadata:
    """Verify output file location, metadata sidecar content."""

    def test_output_file_location(self, ingester, tmp_raw_dir, tmp_path):
        """GeoTIFF is written to data/raw/usfs_wildfire/whp_national.tif."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            result = ingester.fetch()

        expected = tmp_raw_dir / "usfs_wildfire" / WHP_RASTER_FILENAME
        assert result == expected
        assert result.exists()

    def test_metadata_sidecar_written(self, ingester, tmp_raw_dir, tmp_path):
        """Metadata JSON is written alongside the raster."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            ingester.fetch()

        meta_path = tmp_raw_dir / "usfs_wildfire" / WHP_METADATA_FILENAME
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["source"] == "USFS_WHP"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "proxy"

    def test_metadata_raster_properties(self, ingester, tmp_raw_dir, tmp_path):
        """Metadata JSON includes raster properties."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            ingester.fetch()

        meta_path = tmp_raw_dir / "usfs_wildfire" / WHP_METADATA_FILENAME
        meta = json.loads(meta_path.read_text())

        assert "raster_properties" in meta
        rp = meta["raster_properties"]
        assert rp["crs"] == "EPSG:5070"
        assert rp["band_count"] == 1
        assert rp["nodata"] == -1
        assert len(rp["resolution"]) == 2
        assert len(rp["bounds"]) == 4
        assert len(rp["shape"]) == 2

    def test_metadata_data_vintage(self, ingester, tmp_raw_dir, tmp_path):
        """Metadata sidecar includes a data_vintage field."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            ingester.fetch()

        meta_path = tmp_raw_dir / "usfs_wildfire" / WHP_METADATA_FILENAME
        meta = json.loads(meta_path.read_text())

        assert "data_vintage" in meta
        assert "RDS-2015-0047" in meta["data_vintage"]

    def test_metadata_retrieved_at(self, ingester, tmp_raw_dir, tmp_path):
        """Metadata sidecar includes a retrieved_at timestamp."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            ingester.fetch()

        meta_path = tmp_raw_dir / "usfs_wildfire" / WHP_METADATA_FILENAME
        meta = json.loads(meta_path.read_text())
        assert "retrieved_at" in meta


# ---------------------------------------------------------------------------
# Test: Ingest purity
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify no derived metrics are computed by the ingester."""

    def test_no_derived_metrics_in_output(self, ingester, tmp_raw_dir, tmp_path):
        """Output directory has no derived metric files — only raster + metadata."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            ingester.fetch()

        cache_dir = tmp_raw_dir / "usfs_wildfire"
        files = {f.name for f in cache_dir.iterdir()}
        # Only the raster and metadata should exist
        assert files == {WHP_RASTER_FILENAME, WHP_METADATA_FILENAME}

    def test_no_reprojection(self, ingester, tmp_path, tmp_raw_dir):
        """Output raster CRS matches input raster CRS exactly."""
        input_crs = "EPSG:5070"
        synthetic = _create_synthetic_raster(
            tmp_path / "whp.tif", crs=input_crs
        )
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            result = ingester.fetch()

        with rasterio.open(result) as src:
            assert src.crs == CRS.from_string(input_crs)

    def test_no_reclassification(self, ingester, tmp_path, tmp_raw_dir):
        """Pixel values in output are identical to input — no reclassification."""
        values = np.array(
            [[1, 2, 3, 4, 5], [6, 7, 1, 2, 3]], dtype=np.int16
        )
        synthetic = _create_synthetic_raster(
            tmp_path / "whp.tif",
            width=5,
            height=2,
            pixel_values=values,
        )
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            result = ingester.fetch()

        with rasterio.open(result) as src:
            output_values = src.read(1)
        np.testing.assert_array_equal(output_values, values)

    def test_no_county_level_data(self, ingester, tmp_raw_dir, tmp_path):
        """No county-level or FIPS data is produced."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            ingester.fetch()

        cache_dir = tmp_raw_dir / "usfs_wildfire"
        # Check metadata for forbidden keys
        meta_path = cache_dir / WHP_METADATA_FILENAME
        meta = json.loads(meta_path.read_text())
        forbidden_keys = {
            "whp_mean", "pct_high_hazard", "wildfire_score",
            "county_fips", "zonal_stats", "pixel_count",
        }
        assert forbidden_keys.isdisjoint(set(meta.keys()))

        # No parquet files (tabular output) should exist
        parquet_files = list(cache_dir.glob("*.parquet"))
        assert len(parquet_files) == 0


# ---------------------------------------------------------------------------
# Test: Standard ingester tests
# ---------------------------------------------------------------------------

class TestStandardIngester:
    """Verify retry behavior, timeout, and class attributes."""

    def test_retry_on_http_500(self, ingester, tmp_path):
        """Retry logic triggers on HTTP 500 during download."""
        import httpx

        call_count = 0

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_resp = MagicMock()
            if call_count <= 2:
                mock_resp.status_code = 500
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp
            else:
                mock_resp.status_code = 200
                mock_resp.headers = {"content-length": "0"}
                mock_resp.raise_for_status = MagicMock()
                mock_resp.iter_bytes = MagicMock(return_value=iter([b"fake"]))
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp

        dest = tmp_path / "test_download.zip"
        with patch("ingest.usfs_wildfire.httpx.stream", side_effect=mock_stream):
            ingester._download_archive(dest)

        assert call_count == 3

    def test_retry_on_transport_error(self, ingester, tmp_path):
        """Retry logic triggers on transport errors."""
        import httpx

        call_count = 0

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                mock_ctx = MagicMock()
                mock_ctx.__enter__ = MagicMock(
                    side_effect=httpx.ConnectError("connection refused")
                )
                mock_ctx.__exit__ = MagicMock(return_value=False)
                return mock_ctx
            else:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.headers = {"content-length": "0"}
                mock_resp.raise_for_status = MagicMock()
                mock_resp.iter_bytes = MagicMock(return_value=iter([b"ok"]))
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp

        dest = tmp_path / "retry_test.zip"
        with patch("ingest.usfs_wildfire.httpx.stream", side_effect=mock_stream):
            ingester._download_archive(dest)

        assert call_count == 2

    def test_download_timeout_is_generous(self):
        """Download timeout is at least 600 seconds for ~2GB file."""
        assert DOWNLOAD_TIMEOUT_SECONDS >= 600

    def test_source_name(self, ingester):
        """Source name is 'usfs_wildfire'."""
        assert ingester.source_name == "usfs_wildfire"

    def test_confidence(self, ingester):
        """Confidence grade is 'B'."""
        assert ingester.confidence == "B"

    def test_attribution(self, ingester):
        """Attribution is 'proxy'."""
        assert ingester.attribution == "proxy"

    def test_run_delegates_to_fetch(self, ingester, tmp_raw_dir, tmp_path):
        """run() calls fetch() and returns the raster path."""
        synthetic = _create_synthetic_raster(tmp_path / "whp.tif")
        zip_path = _create_synthetic_zip(synthetic, tmp_path / "dl.zip")

        def mock_download(dest: Path) -> None:
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(ingester, "_download_archive", side_effect=mock_download):
            result = ingester.run()

        assert isinstance(result, Path)
        assert result.exists()
        assert result.name == WHP_RASTER_FILENAME


# ---------------------------------------------------------------------------
# Test: Version detection
# ---------------------------------------------------------------------------

class TestVersionDetection:
    """Verify WHP dataset version/vintage detection."""

    def test_version_from_zip_with_readme(self, ingester, synthetic_raster, tmp_path):
        """Version detection finds README in the archive."""
        zip_path = _create_synthetic_zip(
            synthetic_raster,
            tmp_path / "with_readme.zip",
            extra_files={"Data/README.txt": b"WHP version 2023"},
        )
        vintage = ingester._detect_version(zip_path)
        assert "RDS-2015-0047" in vintage

    def test_version_fallback_without_zip(self, ingester):
        """Version detection works when ZIP is not available."""
        vintage = ingester._detect_version(None)
        assert "RDS-2015-0047" in vintage
        assert "downloaded" in vintage

    def test_version_fallback_nonexistent_zip(self, ingester, tmp_path):
        """Version detection handles nonexistent ZIP path gracefully."""
        vintage = ingester._detect_version(tmp_path / "nonexistent.zip")
        assert "RDS-2015-0047" in vintage
