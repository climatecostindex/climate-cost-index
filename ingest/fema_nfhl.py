"""Fetch FEMA National Flood Hazard Layer (NFHL) via MSC county downloads.

Primary: FEMA Map Service Center — Countywide NFHL Product Downloads
    URL: https://msc.fema.gov/portal/downloadProduct?productID=NFHL_{FIPS5}C
Fallback: ArcGIS REST MapServer (for counties where MSC returns HTML)
    URL: https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer
Spatial fallback: TIGERweb county envelope (when DFIRM_ID ≠ county FIPS)
    URL: https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Current/MapServer/86
County FIPS list: https://www2.census.gov/geo/docs/reference/codes/files/national_county.txt
Format: ZIP archive per county containing shapefiles (S_FLD_HAZ_AR, S_FIRM_PAN)
API key required: No

Downloads pre-built county-level NFHL ZIP packages from the FEMA MSC portal.
Each ZIP contains complete shapefiles for a single county. For counties where
MSC returns an HTML page instead of a ZIP (e.g., DC/FIPS 11001), falls back
to querying the ArcGIS REST MapServer layer 28 (S_Fld_Haz_Ar) and layer 3
(S_FIRM_Pan) directly. DFIRM_ID uses FEMA community numbers (not county
FIPS), so if a DFIRM_ID prefix query returns nothing, a spatial intersection
query using the county bounding box from TIGERweb is used as a second
fallback. All paths produce identical output schemas.

Fetches raw flood zone polygons and FIRM panel effective dates ONLY.
Does NOT compute county-level scores, percent-area calculations, housing
unit overlays, or any derived metrics — those belong in
transform/flood_zone_scoring.py.

Output (zones): GeoDataFrame per county with columns:
    geometry, flood_zone, zone_subtype, dfirm_id, state_fips, county_fips
Output (panels): DataFrame per county with columns:
    dfirm_id, effective_date, panel_id, panel_type, state_fips, county_fips
Confidence: A
Attribution: proxy
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
import pandas as pd

from ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEMA MSC configuration
# ---------------------------------------------------------------------------

# FEMA Map Service Center download URL template
MSC_DOWNLOAD_URL = "https://msc.fema.gov/portal/downloadProduct?productID=NFHL_{fips5}C"

# Census county FIPS list URL
CENSUS_COUNTY_FIPS_URL = (
    "https://www2.census.gov/geo/docs/reference/codes/files/national_county.txt"
)

# Shapefile layer names within the ZIP
FLOOD_ZONE_LAYER = "S_FLD_HAZ_AR"
FIRM_PANEL_LAYER = "S_FIRM_PAN"

# Shapefile extensions needed for reading
SHAPEFILE_EXTENSIONS = (".shp", ".shx", ".dbf", ".prj")

# Geometry simplification tolerance (degrees, ~11m at mid-latitudes)
SIMPLIFY_TOLERANCE = 0.0001

# HTTP timeout for county ZIP downloads (seconds)
DOWNLOAD_TIMEOUT = 120.0

# Polite delay between downloads (seconds)
DOWNLOAD_DELAY = 1.5

# Rate limit: polite for federal server
NFHL_CALLS_PER_SECOND = 0.5

# Default max concurrent workers
DEFAULT_MAX_WORKERS = 1

# HTTP status codes indicating no NFHL coverage
NO_COVERAGE_STATUS_CODES = {404}

# HTTP status codes that trigger retry
RETRY_STATUS_CODES = {500, 503}

# Max retry attempts per county download
MAX_DOWNLOAD_RETRIES = 3

# Retry backoff base (seconds)
RETRY_BACKOFF_BASE = 2.0

# ---------------------------------------------------------------------------
# ArcGIS REST API fallback (for counties where MSC returns HTML)
# ---------------------------------------------------------------------------

# FEMA NFHL MapServer base URL
NFHL_REST_BASE = (
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
)

# Layer IDs within the MapServer
FLOOD_ZONE_LAYER_ID = 28  # S_Fld_Haz_Ar
FIRM_PANEL_LAYER_ID = 3   # S_FIRM_Pan

# Fields to retrieve from each layer
REST_FLOOD_ZONE_FIELDS = "FLD_ZONE,ZONE_SUBTY,DFIRM_ID"
REST_FIRM_PANEL_FIELDS = "FIRM_PAN,EFF_DATE,DFIRM_ID,PANEL_TYP"

# ObjectID batch size for REST queries
REST_OBJECTID_BATCH_SIZE = 25

# Timeout for REST API queries (seconds)
REST_QUERY_TIMEOUT = 60.0

# TIGERweb county boundary service (spatial fallback for DFIRM_ID mismatch)
TIGERWEB_COUNTY_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "tigerWMS_Current/MapServer/82/query"
)


class _HtmlFallback:
    """Sentinel: MSC returned HTML instead of ZIP — REST fallback needed."""

    pass


class FEMANFHLIngester(BaseIngester):
    """Ingest FEMA NFHL flood zone data via MSC county ZIP downloads.

    Downloads pre-built countywide NFHL ZIP packages from the FEMA Map
    Service Center portal. Each ZIP contains complete NFHL shapefiles
    for a single county. For counties where MSC returns HTML instead of
    a ZIP (detected via Content-Type header), falls back to ArcGIS REST
    MapServer queries on layer 28 (flood zones) and layer 3 (panels).

    Both paths produce the same output schema: GeoParquet per county with
    columns geometry, flood_zone, zone_subtype, dfirm_id, state_fips,
    county_fips. Metadata sidecars include ``fetch_method`` indicating
    which path was used ("msc" or "arcgis_rest").
    """

    source_name = "fema_nfhl"
    confidence = "A"
    attribution = "proxy"
    calls_per_second = NFHL_CALLS_PER_SECOND

    # Schema validation uses the zone output columns
    required_columns: dict[str, type] = {
        "geometry": object,
        "flood_zone": str,
        "zone_subtype": str,
        "dfirm_id": str,
        "state_fips": str,
        "county_fips": str,
    }

    def __init__(
        self,
        states: list[str] | None = None,
        counties: list[str] | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        """Initialize the FEMA NFHL ingester.

        Args:
            states: Optional list of 2-digit state FIPS codes to process.
                When None, processes all states.
            counties: Optional list of 5-digit county FIPS codes to process.
                When provided, only these counties are processed and the
                states parameter is ignored.
            max_workers: Number of concurrent download workers (default 1).
        """
        super().__init__()
        self._states = states
        self._counties = counties
        self._max_workers = max_workers
        self._file_logger: logging.Logger | None = None

    # -- File logger -----------------------------------------------------------

    def _get_file_logger(self) -> logging.Logger:
        """Return a dedicated file logger for run stats (append mode).

        Creates or re-creates the file handler if the target log path
        has changed (e.g., when RAW_DIR is redirected in tests).
        """
        log_path = self.cache_dir() / "fema_nfhl_run.log"

        if self._file_logger is not None:
            # Check if existing handler still points to the right path
            for h in self._file_logger.handlers:
                if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path:
                    return self._file_logger
            # Path changed — remove old handlers
            for h in list(self._file_logger.handlers):
                h.close()
                self._file_logger.removeHandler(h)

        self._file_logger = logging.getLogger(f"{__name__}.runlog.{id(self)}")
        self._file_logger.setLevel(logging.INFO)
        self._file_logger.propagate = False

        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        self._file_logger.addHandler(handler)

        return self._file_logger

    # -- County FIPS list ------------------------------------------------------

    def _fetch_county_fips_list(self) -> pd.DataFrame:
        """Download and parse the Census county FIPS list.

        Returns:
            DataFrame with columns: state_fips, county_fips, fips5, county_name, classfp.
        """
        cache_path = self.cache_dir() / "national_county.txt"

        if cache_path.exists():
            logger.info("FEMA_NFHL: loading cached county FIPS list from %s", cache_path)
            text = cache_path.read_text()
        else:
            logger.info("FEMA_NFHL: downloading county FIPS list from Census")
            resp = self.api_get(CENSUS_COUNTY_FIPS_URL)
            text = resp.text
            cache_path.write_text(text)

        rows = []
        for line in text.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            # Actual Census format: state_abbr, state_fips, county_fips, county_name, classfp
            # Handle both 4-column (as documented) and 5-column (actual) layouts
            if len(parts) >= 5 and len(parts[1].strip()) == 2 and parts[1].strip().isdigit():
                # 5-column: state_abbr, state_fips(2), county_fips(3), name, classfp
                state_fp = parts[1].strip()
                county_fp = parts[2].strip()
                county_name = parts[3].strip()
                classfp = parts[4].strip()
            else:
                # 4-column: state_fips(2), county_fips(3), name, classfp
                state_fp = parts[0].strip()
                county_fp = parts[1].strip()
                county_name = parts[2].strip()
                classfp = parts[3].strip()
            rows.append({
                "state_fips": state_fp,
                "county_fips_3": county_fp,
                "county_name": county_name,
                "classfp": classfp,
            })

        df = pd.DataFrame(rows)
        df["fips5"] = df["state_fips"] + df["county_fips_3"]

        # Filter to specific counties if provided (takes precedence over states)
        if self._counties is not None:
            df = df[df["fips5"].isin(self._counties)].copy()
            logger.info(
                "FEMA_NFHL: filtered to %d specific counties",
                len(df),
            )
        elif self._states is not None:
            df = df[df["state_fips"].isin(self._states)].copy()
            logger.info(
                "FEMA_NFHL: filtered to %d counties in %d states",
                len(df),
                len(self._states),
            )

        return df

    # -- Download URL ----------------------------------------------------------

    @staticmethod
    def _build_download_url(fips5: str) -> str:
        """Construct the MSC download URL for a county FIPS.

        Args:
            fips5: 5-digit county FIPS code.

        Returns:
            Full download URL string.
        """
        return MSC_DOWNLOAD_URL.format(fips5=fips5)

    # -- Cache check -----------------------------------------------------------

    def _is_county_cached(self, fips5: str) -> bool:
        """Check if a county's flood zone parquet already exists."""
        return (self.cache_dir() / f"fema_nfhl_{fips5}.parquet").exists()

    # -- ZIP download with retry -----------------------------------------------

    def _download_county_zip(self, fips5: str) -> bytes | None:
        """Download the county NFHL ZIP from MSC with retry.

        Args:
            fips5: 5-digit county FIPS code.

        Returns:
            ZIP file bytes, or None if download failed or county has no coverage.
        """
        url = self._build_download_url(fips5)

        for attempt in range(MAX_DOWNLOAD_RETRIES):
            try:
                resp = self.client.get(
                    url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
                )

                if resp.status_code in NO_COVERAGE_STATUS_CODES:
                    logger.info(
                        "FEMA_NFHL: county %s has no NFHL coverage (HTTP %d)",
                        fips5, resp.status_code,
                    )
                    return None

                if resp.status_code in RETRY_STATUS_CODES:
                    if attempt < MAX_DOWNLOAD_RETRIES - 1:
                        wait = RETRY_BACKOFF_BASE ** attempt
                        logger.warning(
                            "FEMA_NFHL: HTTP %d for county %s (attempt %d/%d), "
                            "retrying in %.1fs",
                            resp.status_code, fips5, attempt + 1,
                            MAX_DOWNLOAD_RETRIES, wait,
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.error(
                            "FEMA_NFHL: HTTP %d for county %s after %d attempts",
                            resp.status_code, fips5, MAX_DOWNLOAD_RETRIES,
                        )
                        return None

                resp.raise_for_status()

                # Check Content-Type — MSC returns HTML for some counties
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type.lower():
                    logger.info(
                        "FEMA_NFHL: MSC returned HTML for county %s "
                        "(Content-Type: %s), REST fallback needed",
                        fips5,
                        content_type,
                    )
                    return _HtmlFallback()

                return resp.content

            except httpx.TimeoutException:
                if attempt < MAX_DOWNLOAD_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "FEMA_NFHL: timeout for county %s (attempt %d/%d), "
                        "retrying in %.1fs",
                        fips5, attempt + 1, MAX_DOWNLOAD_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "FEMA_NFHL: timeout for county %s after %d attempts",
                        fips5, MAX_DOWNLOAD_RETRIES,
                    )
                    return None

            except httpx.TransportError as exc:
                if attempt < MAX_DOWNLOAD_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "FEMA_NFHL: transport error for county %s (attempt %d/%d): %s",
                        fips5, attempt + 1, MAX_DOWNLOAD_RETRIES, exc,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "FEMA_NFHL: transport error for county %s after %d attempts: %s",
                        fips5, MAX_DOWNLOAD_RETRIES, exc,
                    )
                    return None

        return None

    # -- ZIP extraction --------------------------------------------------------

    @staticmethod
    def _find_shapefile_members(
        zf: zipfile.ZipFile, layer_name: str
    ) -> list[str]:
        """Find shapefile component paths for a given layer inside a ZIP.

        Handles both flat and nested layouts. Looks for files ending with
        the layer name plus each shapefile extension.

        Args:
            zf: Opened ZipFile object.
            layer_name: Layer name (e.g., "S_FLD_HAZ_AR").

        Returns:
            List of matching member paths within the ZIP.
        """
        targets = [f"{layer_name}{ext}" for ext in SHAPEFILE_EXTENSIONS]
        members = []
        for name in zf.namelist():
            basename = os.path.basename(name)
            if basename in targets:
                members.append(name)
        return members

    @staticmethod
    def _extract_shapefile(
        zf: zipfile.ZipFile,
        members: list[str],
        dest_dir: Path,
    ) -> Path | None:
        """Selectively extract shapefile components to a destination directory.

        Extracts members to dest_dir, flattening any subdirectory nesting.

        Args:
            zf: Opened ZipFile object.
            members: Paths within the ZIP to extract.
            dest_dir: Directory to write extracted files.

        Returns:
            Path to the extracted .shp file, or None if .shp not found.
        """
        shp_path = None
        for member in members:
            basename = os.path.basename(member)
            target_path = dest_dir / basename
            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            if basename.endswith(".shp"):
                shp_path = target_path
        return shp_path

    # -- Shapefile parsing -----------------------------------------------------

    @staticmethod
    def _parse_flood_zones(
        shp_path: Path, fips5: str
    ) -> gpd.GeoDataFrame | None:
        """Read and strip the S_FLD_HAZ_AR shapefile.

        Args:
            shp_path: Path to the .shp file.
            fips5: 5-digit county FIPS code.

        Returns:
            GeoDataFrame with CCI schema, or None if empty/error.
        """
        try:
            gdf = gpd.read_file(shp_path)
        except Exception:
            logger.warning(
                "FEMA_NFHL: failed to read flood zone shapefile for %s",
                fips5, exc_info=True,
            )
            return None

        if gdf.empty:
            logger.info("FEMA_NFHL: empty flood zone shapefile for %s", fips5)
            return None

        # Keep only CCI-needed columns
        keep_cols = {"geometry", "FLD_ZONE", "ZONE_SUBTY", "DFIRM_ID"}
        available = set(gdf.columns) & keep_cols
        if "FLD_ZONE" not in available:
            logger.warning(
                "FEMA_NFHL: missing FLD_ZONE column for %s, columns: %s",
                fips5, list(gdf.columns),
            )
            return None

        result = gdf[list(available)].copy()

        # Rename to CCI schema
        result = result.rename(columns={
            "FLD_ZONE": "flood_zone",
            "ZONE_SUBTY": "zone_subtype",
            "DFIRM_ID": "dfirm_id",
        })

        # Handle missing columns
        if "zone_subtype" not in result.columns:
            result["zone_subtype"] = ""
        if "dfirm_id" not in result.columns:
            result["dfirm_id"] = ""

        # Replace NaN in zone_subtype with empty string
        result["zone_subtype"] = result["zone_subtype"].fillna("").astype(str)
        result["dfirm_id"] = result["dfirm_id"].fillna("").astype(str)
        result["flood_zone"] = result["flood_zone"].astype(str)

        # Add FIPS columns
        result["state_fips"] = fips5[:2]
        result["county_fips"] = fips5

        # Simplify geometry, then fix any self-intersections introduced by simplification
        result["geometry"] = result.geometry.simplify(tolerance=SIMPLIFY_TOLERANCE)
        result["geometry"] = result.geometry.make_valid()

        return result

    @staticmethod
    def _parse_firm_panels(
        shp_path: Path, fips5: str
    ) -> pd.DataFrame | None:
        """Read and strip the S_FIRM_PAN shapefile.

        Args:
            shp_path: Path to the .shp file.
            fips5: 5-digit county FIPS code.

        Returns:
            DataFrame with panel metadata, or None if empty/error.
        """
        try:
            gdf = gpd.read_file(shp_path)
        except Exception:
            logger.warning(
                "FEMA_NFHL: failed to read panel shapefile for %s",
                fips5, exc_info=True,
            )
            return None

        if gdf.empty:
            logger.info("FEMA_NFHL: empty panel shapefile for %s", fips5)
            return None

        # Rename to CCI schema
        rename_map = {
            "DFIRM_ID": "dfirm_id",
            "EFF_DATE": "effective_date",
            "FIRM_PAN": "panel_id",
            "PANEL_TYP": "panel_type",
        }

        df = pd.DataFrame()
        for src, dst in rename_map.items():
            if src in gdf.columns:
                df[dst] = gdf[src]
            else:
                df[dst] = ""

        # Parse effective_date to date type
        if "effective_date" in df.columns:
            df["effective_date"] = pd.to_datetime(
                df["effective_date"], errors="coerce"
            ).dt.date

        # Ensure string types
        for col in ("dfirm_id", "panel_id", "panel_type"):
            df[col] = df[col].fillna("").astype(str)

        # Add FIPS columns
        df["state_fips"] = fips5[:2]
        df["county_fips"] = fips5

        return df

    # -- ArcGIS REST API fallback ----------------------------------------------

    def _rest_query_object_ids(
        self, layer_id: int, where: str
    ) -> list[int]:
        """Query ArcGIS REST for ObjectIDs matching a WHERE clause.

        Uses returnIdsOnly=true for efficiency.

        Args:
            layer_id: MapServer layer number (28 for zones, 3 for panels).
            where: SQL WHERE clause (e.g., "DFIRM_ID LIKE '11001%'").

        Returns:
            List of matching ObjectIDs, or empty list.
        """
        url = f"{NFHL_REST_BASE}/{layer_id}/query"
        params = {
            "where": where,
            "returnIdsOnly": "true",
            "f": "json",
        }
        self.rate_limit()
        resp = self.client.get(url, params=params, timeout=REST_QUERY_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            logger.warning(
                "FEMA_NFHL REST: query error for layer %d: %s",
                layer_id,
                data["error"],
            )
            return []

        oids = data.get("objectIds") or []
        logger.info(
            "FEMA_NFHL REST: layer %d returned %d ObjectIDs for WHERE=%s",
            layer_id,
            len(oids),
            where,
        )
        return oids

    def _fetch_county_envelope(self, fips5: str) -> dict | None:
        """Fetch county bounding box from TIGERweb for spatial queries.

        When DFIRM_ID-based queries return no results (DFIRM_ID uses
        FEMA community numbers, not county FIPS codes), this provides
        the county envelope for a spatial intersection query instead.

        Args:
            fips5: 5-digit county FIPS code (GEOID).

        Returns:
            Envelope dict with xmin, ymin, xmax, ymax keys, or None.
        """
        params = {
            "where": f"GEOID='{fips5}'",
            "returnExtentOnly": "true",
            "outSR": "4269",
            "f": "json",
        }
        self.rate_limit()
        try:
            resp = self.client.get(
                TIGERWEB_COUNTY_URL, params=params, timeout=REST_QUERY_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "FEMA_NFHL REST: TIGERweb envelope query failed for %s: %s",
                fips5,
                exc,
            )
            return None

        extent = data.get("extent")
        if not extent or "xmin" not in extent:
            logger.warning(
                "FEMA_NFHL REST: TIGERweb returned no extent for %s", fips5
            )
            return None

        logger.info(
            "FEMA_NFHL REST: TIGERweb envelope for %s: "
            "[%.4f, %.4f, %.4f, %.4f]",
            fips5,
            extent["xmin"],
            extent["ymin"],
            extent["xmax"],
            extent["ymax"],
        )
        return extent

    def _rest_query_object_ids_spatial(
        self, layer_id: int, envelope: dict
    ) -> list[int]:
        """Query ArcGIS REST for ObjectIDs using a spatial envelope.

        Uses the county bounding box to find flood zones via spatial
        intersection rather than DFIRM_ID matching.

        Args:
            layer_id: MapServer layer number (28 for zones, 3 for panels).
            envelope: Dict with xmin, ymin, xmax, ymax keys.

        Returns:
            List of matching ObjectIDs, or empty list.
        """
        url = f"{NFHL_REST_BASE}/{layer_id}/query"
        geometry = (
            f"{envelope['xmin']},{envelope['ymin']},"
            f"{envelope['xmax']},{envelope['ymax']}"
        )
        params = {
            "geometry": geometry,
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": "4269",
            "returnIdsOnly": "true",
            "f": "json",
        }
        self.rate_limit()
        try:
            resp = self.client.get(
                url, params=params, timeout=REST_QUERY_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "FEMA_NFHL REST: spatial query error for layer %d: %s",
                layer_id,
                exc,
            )
            return []

        if "error" in data:
            logger.warning(
                "FEMA_NFHL REST: spatial query error for layer %d: %s",
                layer_id,
                data["error"],
            )
            return []

        oids = data.get("objectIds") or []
        logger.info(
            "FEMA_NFHL REST: spatial query on layer %d returned %d ObjectIDs",
            layer_id,
            len(oids),
        )
        return oids

    def _rest_fetch_features_batch(
        self,
        layer_id: int,
        object_ids: list[int],
        out_fields: str,
        return_geometry: bool = True,
    ) -> dict:
        """Fetch features by ObjectID batch from ArcGIS REST.

        Uses POST to avoid URL length limits. Returns GeoJSON.
        Retries on server errors with exponential backoff.

        Args:
            layer_id: MapServer layer number.
            object_ids: ObjectIDs to fetch.
            out_fields: Comma-separated field names.
            return_geometry: Whether to include geometry in response.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        url = f"{NFHL_REST_BASE}/{layer_id}/query"
        payload = {
            "objectIds": ",".join(str(oid) for oid in object_ids),
            "outFields": out_fields,
            "returnGeometry": "true" if return_geometry else "false",
            "outSR": "4269",
            "f": "geojson",
        }

        for attempt in range(MAX_DOWNLOAD_RETRIES):
            self.rate_limit()
            try:
                resp = self.client.post(
                    url, data=payload, timeout=REST_QUERY_TIMEOUT
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                if attempt < MAX_DOWNLOAD_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "FEMA_NFHL REST: batch fetch error for layer %d "
                        "(attempt %d/%d, retrying in %.1fs): %s",
                        layer_id,
                        attempt + 1,
                        MAX_DOWNLOAD_RETRIES,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                    continue
                logger.warning(
                    "FEMA_NFHL REST: batch fetch failed after %d attempts "
                    "for layer %d: %s",
                    MAX_DOWNLOAD_RETRIES,
                    layer_id,
                    exc,
                )
                return {"type": "FeatureCollection", "features": []}

            # Check for ArcGIS error responses
            if "error" in data:
                logger.warning(
                    "FEMA_NFHL REST: feature fetch error for layer %d: %s",
                    layer_id,
                    data["error"],
                )
                return {"type": "FeatureCollection", "features": []}

            return data

        return {"type": "FeatureCollection", "features": []}

    def _fetch_flood_zones_rest(self, fips5: str) -> gpd.GeoDataFrame | None:
        """Fetch flood zones for a county via ArcGIS REST API.

        Queries layer 28 (S_Fld_Haz_Ar). Tries DFIRM_ID-based lookup
        first; if that returns nothing (DFIRM_ID uses FEMA community
        numbers, not always matching the 5-digit county FIPS), falls back
        to a spatial intersection query using the county's bounding box
        from TIGERweb.

        Args:
            fips5: 5-digit county FIPS code.

        Returns:
            GeoDataFrame with CCI schema, or None if no data.
        """
        # Strategy 1: DFIRM_ID prefix match (fast, works for most counties)
        where = f"DFIRM_ID LIKE '{fips5}%'"
        oids = self._rest_query_object_ids(FLOOD_ZONE_LAYER_ID, where)

        # Strategy 2: spatial query via TIGERweb county envelope
        if not oids:
            logger.info(
                "FEMA_NFHL REST: DFIRM_ID query returned nothing for %s, "
                "trying spatial query via TIGERweb envelope",
                fips5,
            )
            envelope = self._fetch_county_envelope(fips5)
            if envelope:
                oids = self._rest_query_object_ids_spatial(
                    FLOOD_ZONE_LAYER_ID, envelope
                )

        if not oids:
            logger.info(
                "FEMA_NFHL REST: no flood zone ObjectIDs for county %s "
                "(DFIRM_ID and spatial queries both empty)",
                fips5,
            )
            return None

        # Batch fetch in groups of REST_OBJECTID_BATCH_SIZE
        all_features: list[dict] = []
        for i in range(0, len(oids), REST_OBJECTID_BATCH_SIZE):
            batch = oids[i : i + REST_OBJECTID_BATCH_SIZE]
            geojson = self._rest_fetch_features_batch(
                FLOOD_ZONE_LAYER_ID, batch, REST_FLOOD_ZONE_FIELDS
            )
            all_features.extend(geojson.get("features", []))

        if not all_features:
            return None

        # Convert to GeoDataFrame
        fc = {"type": "FeatureCollection", "features": all_features}
        gdf = gpd.GeoDataFrame.from_features(fc, crs="EPSG:4269")

        if gdf.empty:
            return None

        # Rename to CCI schema
        rename_map = {
            "FLD_ZONE": "flood_zone",
            "ZONE_SUBTY": "zone_subtype",
            "DFIRM_ID": "dfirm_id",
        }
        gdf = gdf.rename(
            columns={k: v for k, v in rename_map.items() if k in gdf.columns}
        )

        # Ensure all columns exist
        if "zone_subtype" not in gdf.columns:
            gdf["zone_subtype"] = ""
        if "dfirm_id" not in gdf.columns:
            gdf["dfirm_id"] = ""
        if "flood_zone" not in gdf.columns:
            logger.warning(
                "FEMA_NFHL REST: no FLD_ZONE in response for %s", fips5
            )
            return None

        # Clean types
        gdf["zone_subtype"] = gdf["zone_subtype"].fillna("").astype(str)
        gdf["dfirm_id"] = gdf["dfirm_id"].fillna("").astype(str)
        gdf["flood_zone"] = gdf["flood_zone"].astype(str)

        # Add FIPS
        gdf["state_fips"] = fips5[:2]
        gdf["county_fips"] = fips5

        # Simplify geometry and fix self-intersections
        gdf["geometry"] = gdf.geometry.simplify(tolerance=SIMPLIFY_TOLERANCE)
        gdf["geometry"] = gdf.geometry.make_valid()

        # Keep only CCI schema columns
        keep = [
            "geometry",
            "flood_zone",
            "zone_subtype",
            "dfirm_id",
            "state_fips",
            "county_fips",
        ]
        gdf = gdf[[c for c in keep if c in gdf.columns]]

        return gdf

    def _fetch_firm_panels_rest(self, fips5: str) -> pd.DataFrame | None:
        """Fetch FIRM panels for a county via ArcGIS REST API.

        Queries layer 3 (S_FIRM_Pan). Tries DFIRM_ID-based lookup first;
        if empty, falls back to spatial query via TIGERweb county envelope.

        Args:
            fips5: 5-digit county FIPS code.

        Returns:
            DataFrame with panel metadata, or None if no data.
        """
        where = f"DFIRM_ID LIKE '{fips5}%'"
        oids = self._rest_query_object_ids(FIRM_PANEL_LAYER_ID, where)

        # Spatial fallback if DFIRM_ID query is empty
        if not oids:
            envelope = self._fetch_county_envelope(fips5)
            if envelope:
                oids = self._rest_query_object_ids_spatial(
                    FIRM_PANEL_LAYER_ID, envelope
                )

        if not oids:
            return None

        # Batch fetch (no geometry needed for panels)
        all_features: list[dict] = []
        for i in range(0, len(oids), REST_OBJECTID_BATCH_SIZE):
            batch = oids[i : i + REST_OBJECTID_BATCH_SIZE]
            geojson = self._rest_fetch_features_batch(
                FIRM_PANEL_LAYER_ID,
                batch,
                REST_FIRM_PANEL_FIELDS,
                return_geometry=False,
            )
            all_features.extend(geojson.get("features", []))

        if not all_features:
            return None

        # Extract properties
        rows = [feat.get("properties", {}) for feat in all_features]
        df = pd.DataFrame(rows)

        if df.empty:
            return None

        # Rename columns
        rename_map = {
            "DFIRM_ID": "dfirm_id",
            "EFF_DATE": "effective_date",
            "FIRM_PAN": "panel_id",
            "PANEL_TYP": "panel_type",
        }
        for src, dst in rename_map.items():
            if src in df.columns:
                df = df.rename(columns={src: dst})
            elif dst not in df.columns:
                df[dst] = ""

        # Parse effective_date (ArcGIS may return epoch ms or string)
        if "effective_date" in df.columns:
            eff = df["effective_date"]
            numeric_eff = pd.to_numeric(eff, errors="coerce")
            if numeric_eff.notna().any():
                df["effective_date"] = pd.to_datetime(
                    numeric_eff, unit="ms", errors="coerce"
                ).dt.date
            else:
                df["effective_date"] = pd.to_datetime(
                    eff, errors="coerce"
                ).dt.date

        # String types
        for col in ("dfirm_id", "panel_id", "panel_type"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Add FIPS
        df["state_fips"] = fips5[:2]
        df["county_fips"] = fips5

        # Keep only expected columns
        keep = [
            "dfirm_id",
            "effective_date",
            "panel_id",
            "panel_type",
            "state_fips",
            "county_fips",
        ]
        df = df[[c for c in keep if c in df.columns]]

        return df

    def _process_county_rest(
        self,
        fips5: str,
        county_name: str,
        result_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a county using the ArcGIS REST API fallback.

        Called when MSC returns HTML instead of a ZIP archive.
        Produces the same output schema as the MSC path.

        Args:
            fips5: 5-digit county FIPS code.
            county_name: County name for logging.
            result_info: Mutable result dict to update.

        Returns:
            Updated result_info dict.
        """
        result_info["fetch_method"] = "arcgis_rest"

        try:
            zones_gdf = self._fetch_flood_zones_rest(fips5)

            if zones_gdf is None or zones_gdf.empty:
                logger.warning(
                    "FEMA_NFHL REST: no flood zone data for county %s (%s)",
                    fips5,
                    county_name,
                )
                result_info["status"] = "empty"
                result_info["warnings"].append(
                    "REST fallback: no flood zone data"
                )
                return result_info

            panels_df = self._fetch_firm_panels_rest(fips5)

            # Save flood zones GeoParquet
            zone_path = self.cache_dir() / f"fema_nfhl_{fips5}.parquet"
            zones_gdf.to_parquet(zone_path, index=False)

            # Write zone metadata
            zone_meta = {
                "source": "FEMA_NFHL",
                "confidence": self.confidence,
                "attribution": self.attribution,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "data_vintage": f"NFHL county {fips5} ({county_name})",
                "row_count": len(zones_gdf),
                "fetch_method": "arcgis_rest",
            }
            zone_meta_path = (
                self.cache_dir() / f"fema_nfhl_{fips5}_metadata.json"
            )
            zone_meta_path.write_text(json.dumps(zone_meta, indent=2))

            result_info["row_count"] = len(zones_gdf)
            result_info["parquet_size"] = zone_path.stat().st_size

            # Save panels
            if panels_df is not None and not panels_df.empty:
                panel_path = (
                    self.cache_dir() / f"fema_nfhl_panels_{fips5}.parquet"
                )
                panels_df.to_parquet(panel_path, index=False)

                panel_meta = {
                    "source": "FEMA_NFHL",
                    "confidence": self.confidence,
                    "attribution": self.attribution,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "data_vintage": (
                        f"NFHL panels county {fips5} ({county_name})"
                    ),
                    "row_count": len(panels_df),
                    "fetch_method": "arcgis_rest",
                }
                panel_meta_path = (
                    self.cache_dir()
                    / f"fema_nfhl_panels_{fips5}_metadata.json"
                )
                panel_meta_path.write_text(json.dumps(panel_meta, indent=2))

                result_info["panel_count"] = len(panels_df)

            result_info["status"] = "success"

            logger.info(
                "FEMA_NFHL: county %s (%s) [REST] — %d zones, %d panels, "
                "%.1f KB",
                fips5,
                county_name,
                result_info["row_count"],
                result_info["panel_count"],
                result_info["parquet_size"] / 1024,
            )

        except Exception as exc:
            logger.error(
                "FEMA_NFHL: REST fallback failed for county %s (%s): %s",
                fips5,
                county_name,
                exc,
                exc_info=True,
            )
            result_info["status"] = "failed"
            result_info["warnings"].append(f"REST fallback error: {exc}")

        return result_info

    # -- Per-county processing -------------------------------------------------

    def _process_county(
        self,
        fips5: str,
        county_name: str,
    ) -> dict[str, Any]:
        """Download, extract, parse, and cache NFHL data for one county.

        Args:
            fips5: 5-digit county FIPS code.
            county_name: County name for logging.

        Returns:
            Dict with status info: fips5, status, row_count, panel_count, etc.
        """
        result_info: dict[str, Any] = {
            "fips5": fips5,
            "county_name": county_name,
            "status": "unknown",
            "fetch_method": "msc",
            "row_count": 0,
            "panel_count": 0,
            "parquet_size": 0,
            "download_time": 0.0,
            "warnings": [],
        }

        t0 = time.monotonic()
        tmp_dir = None

        try:
            # Download ZIP
            zip_bytes = self._download_county_zip(fips5)
            result_info["download_time"] = time.monotonic() - t0

            # Check if MSC returned HTML (needs REST fallback)
            if isinstance(zip_bytes, _HtmlFallback):
                logger.info(
                    "FEMA_NFHL: MSC returned HTML for %s (%s), "
                    "switching to REST fallback",
                    fips5,
                    county_name,
                )
                return self._process_county_rest(
                    fips5, county_name, result_info
                )

            if zip_bytes is None:
                result_info["status"] = "no_coverage"
                return result_info

            # Validate ZIP
            try:
                zf = zipfile.ZipFile(__import__("io").BytesIO(zip_bytes))
            except zipfile.BadZipFile:
                logger.warning(
                    "FEMA_NFHL: corrupted ZIP for county %s", fips5
                )
                result_info["status"] = "failed"
                result_info["warnings"].append("corrupted ZIP")
                return result_info

            tmp_dir = Path(tempfile.mkdtemp(prefix=f"nfhl_{fips5}_"))

            try:
                # Extract flood zone shapefile
                zone_members = self._find_shapefile_members(zf, FLOOD_ZONE_LAYER)
                zone_shp_path = None
                if zone_members:
                    zone_shp_path = self._extract_shapefile(
                        zf, zone_members, tmp_dir
                    )

                # Extract panel shapefile
                panel_members = self._find_shapefile_members(zf, FIRM_PANEL_LAYER)
                panel_shp_path = None
                if panel_members:
                    panel_shp_path = self._extract_shapefile(
                        zf, panel_members, tmp_dir
                    )

                zf.close()

                # Parse flood zones
                zones_gdf = None
                if zone_shp_path is not None:
                    zones_gdf = self._parse_flood_zones(zone_shp_path, fips5)

                if zones_gdf is None or zones_gdf.empty:
                    logger.warning(
                        "FEMA_NFHL: no flood zone data for county %s (%s)",
                        fips5, county_name,
                    )
                    result_info["status"] = "empty"
                    result_info["warnings"].append("no flood zone data")
                    return result_info

                # Parse panels
                panels_df = None
                if panel_shp_path is not None:
                    panels_df = self._parse_firm_panels(panel_shp_path, fips5)

                # Save flood zones GeoParquet
                zone_path = self.cache_dir() / f"fema_nfhl_{fips5}.parquet"
                zones_gdf.to_parquet(zone_path, index=False)

                # Write zone metadata
                zone_meta = {
                    "source": "FEMA_NFHL",
                    "confidence": self.confidence,
                    "attribution": self.attribution,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "data_vintage": f"NFHL county {fips5} ({county_name})",
                    "row_count": len(zones_gdf),
                }
                zone_meta_path = self.cache_dir() / f"fema_nfhl_{fips5}_metadata.json"
                zone_meta_path.write_text(json.dumps(zone_meta, indent=2))

                result_info["row_count"] = len(zones_gdf)
                result_info["parquet_size"] = zone_path.stat().st_size

                # Save panels
                if panels_df is not None and not panels_df.empty:
                    panel_path = self.cache_dir() / f"fema_nfhl_panels_{fips5}.parquet"
                    panels_df.to_parquet(panel_path, index=False)

                    panel_meta = {
                        "source": "FEMA_NFHL",
                        "confidence": self.confidence,
                        "attribution": self.attribution,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "data_vintage": f"NFHL panels county {fips5} ({county_name})",
                        "row_count": len(panels_df),
                    }
                    panel_meta_path = self.cache_dir() / f"fema_nfhl_panels_{fips5}_metadata.json"
                    panel_meta_path.write_text(json.dumps(panel_meta, indent=2))

                    result_info["panel_count"] = len(panels_df)

                result_info["status"] = "success"

                logger.info(
                    "FEMA_NFHL: county %s (%s) — %d zones, %d panels, %.1f KB",
                    fips5,
                    county_name,
                    result_info["row_count"],
                    result_info["panel_count"],
                    result_info["parquet_size"] / 1024,
                )

            finally:
                zf.close() if not zf.fp else None  # type: ignore[union-attr]

        except Exception as exc:
            logger.error(
                "FEMA_NFHL: unexpected error for county %s (%s): %s",
                fips5, county_name, exc, exc_info=True,
            )
            result_info["status"] = "failed"
            result_info["warnings"].append(str(exc))

        finally:
            # Clean up temp directory
            if tmp_dir is not None and tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return result_info

    # -- Run logging -----------------------------------------------------------

    def _log_county_result(self, info: dict[str, Any]) -> None:
        """Write per-county stats to fema_nfhl_run.log."""
        file_logger = self._get_file_logger()
        warnings_str = "; ".join(info["warnings"]) if info["warnings"] else "none"
        file_logger.info(
            "county=%s name=%s status=%s method=%s rows=%d panels=%d "
            "size_kb=%.1f download_sec=%.1f warnings=%s",
            info["fips5"],
            info["county_name"],
            info["status"],
            info.get("fetch_method", "msc"),
            info["row_count"],
            info["panel_count"],
            info["parquet_size"] / 1024,
            info["download_time"],
            warnings_str,
        )

    def _write_run_summary(self, results: list[dict[str, Any]]) -> None:
        """Write end-of-run summary JSON."""
        succeeded = [r for r in results if r["status"] == "success"]
        no_coverage = [r for r in results if r["status"] == "no_coverage"]
        failed = [r for r in results if r["status"] == "failed"]
        empty = [r for r in results if r["status"] == "empty"]

        rest_fallback = [
            r for r in results if r.get("fetch_method") == "arcgis_rest"
        ]

        summary = {
            "total_attempted": len(results),
            "total_succeeded": len(succeeded),
            "total_no_coverage": len(no_coverage),
            "total_failed": len(failed),
            "total_empty": len(empty),
            "total_rest_fallback": len(rest_fallback),
            "failed_fips": [r["fips5"] for r in failed],
            "no_coverage_fips": [r["fips5"] for r in no_coverage],
            "rest_fallback_fips": [r["fips5"] for r in rest_fallback],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        summary_path = self.cache_dir() / "fema_nfhl_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        logger.info(
            "FEMA_NFHL: run complete — %d attempted, %d succeeded, "
            "%d no coverage, %d failed, %d empty",
            len(results),
            len(succeeded),
            len(no_coverage),
            len(failed),
            len(empty),
        )

    # -- Main fetch ------------------------------------------------------------

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch NFHL flood zone data for all target counties.

        NFHL is a snapshot dataset, not a time series. The ``years``
        parameter is accepted for interface compatibility but ignored.

        Returns:
            Empty DataFrame (county-by-county parquets are the primary output).
            Individual county data is cached as per-county GeoParquet files.
        """
        # Get county FIPS list
        county_df = self._fetch_county_fips_list()
        logger.info(
            "FEMA_NFHL: processing %d counties", len(county_df)
        )

        # Determine which counties need downloading
        counties_to_process = []
        skipped = 0
        for _, row in county_df.iterrows():
            fips5 = row["fips5"]
            if self._is_county_cached(fips5):
                skipped += 1
                continue
            counties_to_process.append((fips5, row["county_name"]))

        logger.info(
            "FEMA_NFHL: %d counties to download (%d already cached)",
            len(counties_to_process), skipped,
        )

        # Process counties
        results: list[dict[str, Any]] = []

        if self._max_workers <= 1:
            # Sequential processing
            for fips5, county_name in counties_to_process:
                info = self._process_county(fips5, county_name)
                self._log_county_result(info)
                results.append(info)
                # Polite delay between downloads
                time.sleep(DOWNLOAD_DELAY)
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(self._process_county, fips5, name): fips5
                    for fips5, name in counties_to_process
                }
                for future in as_completed(futures):
                    try:
                        info = future.result()
                        self._log_county_result(info)
                        results.append(info)
                    except Exception as exc:
                        fips5 = futures[future]
                        logger.error(
                            "FEMA_NFHL: worker error for county %s: %s",
                            fips5, exc,
                        )
                        results.append({
                            "fips5": fips5,
                            "county_name": "",
                            "status": "failed",
                            "row_count": 0,
                            "panel_count": 0,
                            "parquet_size": 0,
                            "download_time": 0.0,
                            "warnings": [str(exc)],
                        })

        # Write run summary
        self._write_run_summary(results)

        # Return empty GeoDataFrame — per-county parquets are the primary output
        return gpd.GeoDataFrame(
            columns=list(self.required_columns.keys()),
            geometry="geometry",
        )
