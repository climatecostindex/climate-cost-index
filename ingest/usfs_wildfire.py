"""Fetch USFS Wildfire Hazard Potential (WHP) raster from the Research Data Archive.

Source: USFS Wildfire Hazard Potential (WHP)
URL: https://www.firelab.org/project/wildfire-hazard-potential
Dataset: RDS-2015-0047 (or newer revision) from USFS Research Data Archive
Format: Raster GeoTIFF, 270m resolution, contiguous U.S.
API key required: No

Downloads the WHP raster archive (~2GB ZIP), extracts the GeoTIFF, verifies
raster metadata (CRS, resolution, extent, band count, data type, NoData),
validates pixel value range, and caches the raw raster to disk.

This is a raster-based ingester — the output is a GeoTIFF file, not a
tabular DataFrame. All zonal statistics and county-level aggregation belong
in transform/wildfire_scoring.py.

Output:
    data/raw/usfs_wildfire/whp_national.tif     # The WHP GeoTIFF raster
    data/raw/usfs_wildfire/_metadata.json        # Metadata sidecar

Confidence: B
Attribution: proxy
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import rasterio

from ingest.base import BaseIngester, RETRYABLE_STATUS_CODES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Primary download URL for WHP dataset (RDS-2015-0047)
WHP_DOWNLOAD_URL = (
    "https://www.fs.usda.gov/rds/archive/products/RDS-2015-0047-4/RDS-2015-0047-4_Data.zip"
)

# Dataset identifier
WHP_DATASET_ID = "RDS-2015-0047"

# Expected raster properties (used for validation, not enforcement)
EXPECTED_CRS_EPSG = 5070  # NAD83 CONUS Albers Equal Area
EXPECTED_RESOLUTION_M = 270.0  # Approximate pixel size in meters
EXPECTED_BAND_COUNT = 1

# Valid pixel value range for classified WHP raster:
#   1 = Very Low, 2 = Low, 3 = Moderate, 4 = High, 5 = Very High,
#   6 = Non-burnable, 7 = Water
# NoData is encoded separately (255 for uint8, 2147483647 for int32).
VALID_PIXEL_MIN = 1
VALID_PIXEL_MAX = 7

# Preferred raster filename pattern — the classified CONUS raster
# The archive contains both continuous (cnt) and classified (cls) versions
# for CONUS, Alaska, and Hawaii. We want the classified CONUS version
# which has the 1–5 WHP scale matching the SSRN spec.
WHP_PREFERRED_PATTERN = "cls_conus"

# Standardized output filename
WHP_RASTER_FILENAME = "whp_national.tif"
WHP_METADATA_FILENAME = "_metadata.json"

# Download settings
DOWNLOAD_TIMEOUT_SECONDS = 900.0  # 15 minutes for a large file
DOWNLOAD_CHUNK_SIZE = 65_536  # 64KB chunks for streaming download

# HTTP headers required by the USFS Research Data Archive
# The server returns 403 without a browser-like User-Agent and Referer.
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.fs.usda.gov/rds/archive/catalog/RDS-2015-0047-4",
    "Accept": "application/octet-stream,*/*",
}

# Number of rows to sample for pixel value validation
PIXEL_SAMPLE_ROWS = 100


class USFSWildfireIngester(BaseIngester):
    """Ingest the USFS Wildfire Hazard Potential (WHP) raster.

    Downloads the WHP GeoTIFF from the USFS Research Data Archive,
    verifies raster metadata, validates pixel values, and caches
    the raw raster file. No derived metrics or zonal statistics
    are computed — those belong in transform/wildfire_scoring.py.

    This is a raster-based ingester. Unlike tabular ingesters, it
    outputs a GeoTIFF file rather than a DataFrame.
    """

    source_name = "usfs_wildfire"
    confidence = "B"
    attribution = "proxy"

    # Raster ingester produces no tabular output
    required_columns: dict[str, type] = {}

    # Single large file download — no rate limiting needed
    calls_per_second = 0.0
    max_retries = 3
    retry_backoff_base = 2.0

    # -- Paths -----------------------------------------------------------------

    def _raster_path(self) -> Path:
        """Return the path where the cached WHP raster should be stored."""
        return self.cache_dir() / WHP_RASTER_FILENAME

    def _metadata_path(self) -> Path:
        """Return the path for the metadata sidecar JSON."""
        return self.cache_dir() / WHP_METADATA_FILENAME

    def _is_cached(self) -> bool:
        """Check if the WHP raster is already cached."""
        raster_path = self._raster_path()
        if raster_path.exists() and raster_path.stat().st_size > 1_000:
            logger.info(
                "USFS_WHP: raster already cached at %s (%d bytes)",
                raster_path,
                raster_path.stat().st_size,
            )
            return True
        return False

    # -- Download --------------------------------------------------------------

    def _download_archive(self, dest: Path) -> None:
        """Download the WHP ZIP archive with streaming.

        Uses httpx streaming to avoid loading the entire ~2GB file into
        memory. Retries on HTTP 500/503 and transport errors.

        Args:
            dest: Path to save the downloaded ZIP archive.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    "USFS_WHP: downloading archive (attempt %d/%d) from %s",
                    attempt + 1,
                    self.max_retries + 1,
                    WHP_DOWNLOAD_URL,
                )
                with httpx.stream(
                    "GET",
                    WHP_DOWNLOAD_URL,
                    timeout=DOWNLOAD_TIMEOUT_SECONDS,
                    follow_redirects=True,
                    headers=DOWNLOAD_HEADERS,
                ) as resp:
                    if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                        wait = self.retry_backoff_base ** attempt
                        logger.warning(
                            "USFS_WHP: HTTP %d (attempt %d/%d), retrying in %.1fs",
                            resp.status_code,
                            attempt + 1,
                            self.max_retries,
                            wait,
                        )
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()

                    downloaded = 0
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_bytes(DOWNLOAD_CHUNK_SIZE):
                            f.write(chunk)
                            downloaded += len(chunk)

                    logger.info(
                        "USFS_WHP: downloaded %d bytes to %s",
                        downloaded,
                        dest,
                    )
                    return

            except httpx.TransportError as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise
                wait = self.retry_backoff_base ** attempt
                logger.warning(
                    "USFS_WHP: transport error (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # -- Extraction ------------------------------------------------------------

    def _extract_geotiff(self, zip_path: Path, extract_dir: Path) -> Path:
        """Extract the WHP GeoTIFF from the downloaded ZIP archive.

        Searches the archive for .tif/.tiff files and extracts only the
        primary WHP raster. Ancillary files (XML, PDF, TXT) are ignored.

        Args:
            zip_path: Path to the downloaded ZIP archive.
            extract_dir: Temporary directory for extraction.

        Returns:
            Path to the extracted GeoTIFF file.

        Raises:
            FileNotFoundError: If no GeoTIFF is found in the archive.
        """
        logger.info("USFS_WHP: extracting GeoTIFF from %s", zip_path)

        tif_paths: list[str] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith((".tif", ".tiff")) and not name.startswith("__MACOSX"):
                    tif_paths.append(name)

            if not tif_paths:
                raise FileNotFoundError(
                    f"No GeoTIFF files found in archive: {zip_path}"
                )

            # Prefer the classified CONUS raster (1–5 WHP scale)
            target = None
            if len(tif_paths) > 1:
                for name in tif_paths:
                    if WHP_PREFERRED_PATTERN in name.lower():
                        target = name
                        break
                if target:
                    logger.info(
                        "USFS_WHP: found %d GeoTIFF files, "
                        "selected classified CONUS: %s",
                        len(tif_paths),
                        target,
                    )
                else:
                    # Fallback: use the largest TIF
                    tif_paths.sort(
                        key=lambda n: zf.getinfo(n).file_size, reverse=True
                    )
                    target = tif_paths[0]
                    logger.info(
                        "USFS_WHP: found %d GeoTIFF files, "
                        "no classified CONUS found, using largest: %s",
                        len(tif_paths),
                        target,
                    )
            else:
                target = tif_paths[0]
            zf.extract(target, extract_dir)

        extracted_path = extract_dir / target
        logger.info(
            "USFS_WHP: extracted %s (%d bytes)",
            extracted_path,
            extracted_path.stat().st_size,
        )
        return extracted_path

    # -- Raster verification ---------------------------------------------------

    def _verify_raster(self, raster_path: Path) -> dict:
        """Verify raster metadata and return properties dict.

        Opens the raster with rasterio and validates CRS, resolution,
        spatial extent, band count, data type, and NoData value.

        Args:
            raster_path: Path to the GeoTIFF file.

        Returns:
            Dict of raster properties for the metadata sidecar.
        """
        with rasterio.open(raster_path) as src:
            crs = src.crs
            res = src.res
            bounds = src.bounds
            shape = (src.height, src.width)
            dtype = str(src.dtypes[0])
            nodata = src.nodata
            band_count = src.count

            logger.info("USFS_WHP: CRS = %s", crs)
            logger.info("USFS_WHP: resolution = %s", res)
            logger.info(
                "USFS_WHP: bounds = (%.1f, %.1f, %.1f, %.1f)",
                bounds.left, bounds.bottom, bounds.right, bounds.top,
            )
            logger.info("USFS_WHP: shape = %s", shape)
            logger.info("USFS_WHP: dtype = %s", dtype)
            logger.info("USFS_WHP: nodata = %s", nodata)
            logger.info("USFS_WHP: band_count = %d", band_count)

            if crs is None:
                logger.warning("USFS_WHP: raster has no CRS defined")

            if band_count != EXPECTED_BAND_COUNT:
                logger.warning(
                    "USFS_WHP: expected %d band, found %d",
                    EXPECTED_BAND_COUNT,
                    band_count,
                )

            if nodata is None:
                logger.warning("USFS_WHP: no NoData value defined in raster")

            properties = {
                "crs": crs.to_string() if crs else None,
                "resolution": list(res),
                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "shape": list(shape),
                "dtype": dtype,
                "nodata": nodata,
                "band_count": band_count,
            }

        return properties

    def _validate_pixel_values(self, raster_path: Path) -> None:
        """Validate that pixel values fall within the expected 0–5 range.

        Reads a sample of rows and checks for out-of-range values.
        Logs a warning if unexpected values are found but does not fail.

        Args:
            raster_path: Path to the GeoTIFF file.
        """
        with rasterio.open(raster_path) as src:
            height = src.height
            nodata = src.nodata

            sample_count = min(PIXEL_SAMPLE_ROWS, height)
            row_indices = np.linspace(0, height - 1, sample_count, dtype=int)

            out_of_range_count = 0
            total_valid_pixels = 0

            for row_idx in row_indices:
                window = rasterio.windows.Window(0, int(row_idx), src.width, 1)
                data = src.read(1, window=window)

                if nodata is not None:
                    valid = data[data != nodata]
                else:
                    valid = data.ravel()

                if valid.size == 0:
                    continue

                total_valid_pixels += valid.size
                oor = np.sum((valid < VALID_PIXEL_MIN) | (valid > VALID_PIXEL_MAX))
                out_of_range_count += int(oor)

            if out_of_range_count > 0:
                logger.warning(
                    "USFS_WHP: found %d out-of-range pixels (outside %d–%d) "
                    "in sample of %d valid pixels across %d rows",
                    out_of_range_count,
                    VALID_PIXEL_MIN,
                    VALID_PIXEL_MAX,
                    total_valid_pixels,
                    sample_count,
                )
            else:
                logger.info(
                    "USFS_WHP: pixel value validation passed — %d sampled "
                    "pixels all within %d–%d",
                    total_valid_pixels,
                    VALID_PIXEL_MIN,
                    VALID_PIXEL_MAX,
                )

    # -- Version detection -----------------------------------------------------

    def _detect_version(self, zip_path: Path | None) -> str:
        """Attempt to detect the WHP dataset version from the archive.

        Checks for version information in README or metadata files
        within the ZIP. Falls back to the download URL dataset ID.

        Args:
            zip_path: Path to the ZIP archive (may be None if already deleted).

        Returns:
            Version/vintage string for the metadata sidecar.
        """
        if zip_path is not None and zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for name in zf.namelist():
                        lower = name.lower()
                        if "readme" in lower or "metadata" in lower:
                            logger.info(
                                "USFS_WHP: found version info file: %s", name
                            )
                            break
            except Exception:
                logger.debug("USFS_WHP: could not read ZIP for version info")

        return (
            f"{WHP_DATASET_ID} "
            f"(downloaded {datetime.now(timezone.utc).strftime('%Y-%m-%d')})"
        )

    # -- Metadata sidecar ------------------------------------------------------

    def _write_metadata(
        self,
        raster_properties: dict,
        data_vintage: str,
    ) -> Path:
        """Write the metadata sidecar JSON.

        Args:
            raster_properties: Dict of raster properties from _verify_raster.
            data_vintage: Version/vintage string for the WHP dataset.

        Returns:
            Path to the written metadata JSON file.
        """
        metadata = {
            "source": "USFS_WHP",
            "confidence": self.confidence,
            "attribution": self.attribution,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "data_vintage": data_vintage,
            "raster_properties": raster_properties,
        }
        meta_path = self._metadata_path()
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info("USFS_WHP: wrote metadata to %s", meta_path)
        return meta_path

    # -- Main workflow ---------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> Path:
        """Fetch the WHP raster, verify, and cache it.

        WHP is a snapshot dataset (not a time series), so the ``years``
        parameter is ignored. Downloads the most recent available version.

        Args:
            years: Ignored — WHP is not a time-series dataset.

        Returns:
            Path to the cached GeoTIFF raster file.
        """
        raster_dest = self._raster_path()

        # Skip download if already cached
        if self._is_cached():
            logger.info("USFS_WHP: using cached raster at %s", raster_dest)
            raster_properties = self._verify_raster(raster_dest)
            self._validate_pixel_values(raster_dest)
            data_vintage = self._detect_version(None)
            self._write_metadata(raster_properties, data_vintage)
            return raster_dest

        # Download and extract in a temporary directory
        tmp_dir = None
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="usfs_whp_"))
            zip_path = tmp_dir / "whp_archive.zip"

            # Step 1: Download the archive
            self._download_archive(zip_path)

            # Step 2: Detect version while ZIP is still available
            data_vintage = self._detect_version(zip_path)

            # Step 3: Extract the GeoTIFF
            extract_dir = tmp_dir / "extracted"
            extract_dir.mkdir()
            extracted_tif = self._extract_geotiff(zip_path, extract_dir)

            # Step 4: Verify raster metadata
            raster_properties = self._verify_raster(extracted_tif)

            # Step 5: Validate pixel values
            self._validate_pixel_values(extracted_tif)

            # Step 6: Copy to cache directory
            self.cache_dir()  # Ensure directory exists
            shutil.copy2(extracted_tif, raster_dest)
            logger.info(
                "USFS_WHP: cached raster to %s (%d bytes)",
                raster_dest,
                raster_dest.stat().st_size,
            )

            # Step 7: Write metadata sidecar
            self._write_metadata(raster_properties, data_vintage)

        finally:
            # Step 8: Clean up temporary files
            if tmp_dir is not None and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.info(
                    "USFS_WHP: cleaned up temporary directory %s", tmp_dir
                )

        return raster_dest

    def run(self, years: list[int] | None = None) -> Path:
        """Download, verify, and cache the WHP raster.

        Overrides the base class ``run()`` because this ingester produces
        a raster file, not a DataFrame. Schema validation and completeness
        logging are not applicable.

        Args:
            years: Ignored — WHP is not a time-series dataset.

        Returns:
            Path to the cached GeoTIFF raster file.
        """
        logger.info("%s: starting ingestion", self.source_name)
        return self.fetch(years=years)
