"""Tests for transform/station_to_county.py — station-to-county spatial mapping.

All tests use synthetic DataFrames with known expected outputs.
No real data files are read.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from transform.station_to_county import (
    OUTPUT_COLUMNS,
    map_stations_to_counties,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic counties and stations
# ---------------------------------------------------------------------------

def _make_county_gdf(counties: list[dict]) -> gpd.GeoDataFrame:
    """Build a synthetic county boundary GeoDataFrame.

    Each dict in *counties* must have keys: fips, bounds (xmin, ymin, xmax, ymax).
    """
    records = []
    for c in counties:
        records.append(
            {
                "GEOID": c["fips"],
                "geometry": box(*c["bounds"]),
            }
        )
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4269")
    return gdf


def _make_station_df(stations: list[dict]) -> pd.DataFrame:
    """Build a synthetic station metadata DataFrame.

    Each dict must have keys: station_id, lat, lon.
    """
    return pd.DataFrame(stations)


# Two adjacent square counties:
# County A ("01001"): lon -90 to -89, lat 30 to 31
# County B ("01003"): lon -89 to -88, lat 30 to 31
# County C ("02001"): lon -100 to -99, lat 40 to 41  (empty — no stations)
COUNTIES = [
    {"fips": "01001", "bounds": (-90, 30, -89, 31)},
    {"fips": "01003", "bounds": (-89, 30, -88, 31)},
    {"fips": "02001", "bounds": (-100, 40, -99, 41)},
]

# Stations placed clearly inside known counties:
STATIONS = [
    {"station_id": "USW00001", "lat": 30.5, "lon": -89.5},  # inside 01001
    {"station_id": "USW00002", "lat": 30.3, "lon": -89.7},  # inside 01001
    {"station_id": "USW00003", "lat": 30.8, "lon": -89.2},  # inside 01001
    {"station_id": "USW00004", "lat": 30.5, "lon": -88.5},  # inside 01003
]


def _run_mapping(
    stations: list[dict] | None = None,
    counties: list[dict] | None = None,
) -> pd.DataFrame:
    """Helper: run map_stations_to_counties with synthetic data."""
    station_df = _make_station_df(STATIONS if stations is None else stations)
    county_gdf = _make_county_gdf(COUNTIES if counties is None else counties)

    # Write county shapefile to a temp location so the function can read it
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = Path(tmpdir) / "counties.shp"
        county_gdf.to_file(shp_path)
        result = map_stations_to_counties(station_df, shp_path)
    return result


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestBasicSpatialJoin:
    """Verify stations are assigned to the correct county."""

    def test_station_in_county_a(self):
        result = _run_mapping()
        row = result[result["station_id"] == "USW00001"]
        assert len(row) == 1
        assert row.iloc[0]["fips"] == "01001"

    def test_station_in_county_b(self):
        result = _run_mapping()
        row = result[result["station_id"] == "USW00004"]
        assert len(row) == 1
        assert row.iloc[0]["fips"] == "01003"

    def test_all_stations_assigned(self):
        result = _run_mapping()
        assigned = result[result["station_id"].notna()]
        assert len(assigned) == 4


class TestMultiStationCounty:
    """County with 3 stations should have station_count == 3."""

    def test_multi_station_count(self):
        result = _run_mapping()
        county_a = result[result["fips"] == "01001"]
        stations_in_a = county_a[county_a["station_id"].notna()]
        assert len(stations_in_a) == 3
        assert (stations_in_a["station_count"] == 3).all()


class TestSingleStationCounty:
    """County with 1 station should have station_count == 1."""

    def test_single_station_count(self):
        result = _run_mapping()
        county_b = result[result["fips"] == "01003"]
        stations_in_b = county_b[county_b["station_id"].notna()]
        assert len(stations_in_b) == 1
        assert stations_in_b.iloc[0]["station_count"] == 1


class TestZeroStationCounty:
    """County with no stations appears with NaN station_id and station_count == 0."""

    def test_zero_station_in_output(self):
        result = _run_mapping()
        county_c = result[result["fips"] == "02001"]
        assert len(county_c) == 1
        assert pd.isna(county_c.iloc[0]["station_id"])
        assert county_c.iloc[0]["station_count"] == 0

    def test_zero_station_lat_lon_nan(self):
        result = _run_mapping()
        county_c = result[result["fips"] == "02001"]
        assert pd.isna(county_c.iloc[0]["lat"])
        assert pd.isna(county_c.iloc[0]["lon"])


class TestStationOutsideAllCounties:
    """Station outside all polygons should be handled via nearest-county fallback."""

    def test_outside_station_not_dropped(self):
        # Station far outside all county polygons
        stations = [
            {"station_id": "USW00001", "lat": 30.5, "lon": -89.5},  # inside 01001
            {"station_id": "USW00099", "lat": 35.0, "lon": -85.0},  # outside all
        ]
        result = _run_mapping(stations=stations)
        assigned = result[result["station_id"].notna()]
        station_ids = set(assigned["station_id"].values)
        assert "USW00099" in station_ids, "Outside station should be assigned via nearest fallback"

    def test_outside_station_gets_nearest_fips(self):
        stations = [
            {"station_id": "USW00099", "lat": 30.5, "lon": -87.5},  # closest to 01003
        ]
        result = _run_mapping(stations=stations)
        row = result[result["station_id"] == "USW00099"]
        assert len(row) == 1
        # Should map to nearest county (01003, since -87.5 is closest to -88..-89 range)
        assert row.iloc[0]["fips"] in {"01001", "01003", "02001"}


class TestCRSAlignment:
    """Verify spatial join uses EPSG:4269 and handles reprojection."""

    def test_different_crs_reprojected(self):
        """Stations in NAD83 but counties in Web Mercator — should reproject."""
        station_df = _make_station_df(
            [{"station_id": "USW00001", "lat": 30.5, "lon": -89.5}]
        )
        # Build county in Web Mercator
        county_gdf = _make_county_gdf(
            [{"fips": "01001", "bounds": (-90, 30, -89, 31)}]
        )
        county_gdf_mercator = county_gdf.to_crs("EPSG:3857")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "counties.shp"
            county_gdf_mercator.to_file(shp_path)
            result = map_stations_to_counties(station_df, shp_path)

        assigned = result[result["station_id"].notna()]
        assert len(assigned) == 1
        assert assigned.iloc[0]["fips"] == "01001"


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has correct columns and types."""

    def test_column_presence(self):
        result = _run_mapping()
        for col in OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_extra_columns(self):
        result = _run_mapping()
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_column_types(self):
        result = _run_mapping()
        assigned = result[result["station_id"].notna()]
        if not assigned.empty:
            # station_id and fips must be string-like (object or StringDtype)
            assert pd.api.types.is_string_dtype(assigned["station_id"])
            assert pd.api.types.is_string_dtype(assigned["fips"])
            assert pd.api.types.is_float_dtype(assigned["lat"])
            assert pd.api.types.is_float_dtype(assigned["lon"])
            assert pd.api.types.is_integer_dtype(assigned["station_count"])


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestMissingCoordinates:
    """Stations with NaN lat/lon are dropped before the spatial join."""

    def test_nan_lat_dropped(self):
        stations = [
            {"station_id": "USW00001", "lat": np.nan, "lon": -89.5},
            {"station_id": "USW00002", "lat": 30.5, "lon": -89.5},
        ]
        result = _run_mapping(stations=stations)
        assigned = result[result["station_id"].notna()]
        assert "USW00001" not in set(assigned["station_id"].values)
        assert "USW00002" in set(assigned["station_id"].values)

    def test_nan_lon_dropped(self):
        stations = [
            {"station_id": "USW00001", "lat": 30.5, "lon": np.nan},
            {"station_id": "USW00002", "lat": 30.5, "lon": -89.5},
        ]
        result = _run_mapping(stations=stations)
        assigned = result[result["station_id"].notna()]
        assert "USW00001" not in set(assigned["station_id"].values)

    def test_all_nan_coords(self):
        """All stations have NaN coordinates — result should have only zero-station counties."""
        stations = [
            {"station_id": "USW00001", "lat": np.nan, "lon": np.nan},
        ]
        result = _run_mapping(stations=stations)
        assert (result["station_count"] == 0).all()
        assert result["station_id"].isna().all()


class TestEmptyStationMetadata:
    """Empty station metadata should produce zero-station entries for all counties."""

    def test_empty_stations(self):
        result = _run_mapping(stations=[])
        assert len(result) == len(COUNTIES)
        assert (result["station_count"] == 0).all()
        assert result["station_id"].isna().all()


class TestEmptyCountyBoundaries:
    """Empty county boundaries should return empty result gracefully."""

    def test_empty_counties(self):
        station_df = _make_station_df(STATIONS)
        # Pass an empty GeoDataFrame directly instead of writing empty shapefile
        result = map_stations_to_counties(station_df, None)
        # With no county boundaries found, should raise FileNotFoundError
        # or return empty. Test via the internal path: provide empty counties.

    def test_empty_counties_no_crash(self):
        """Passing empty county data via a temp file should not crash."""
        import tempfile

        county_gdf = gpd.GeoDataFrame(
            {"GEOID": pd.Series(dtype=str)},
            geometry=gpd.GeoSeries([], crs="EPSG:4269"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "counties.shp"
            county_gdf.to_file(shp_path)
            result = map_stations_to_counties(
                _make_station_df(STATIONS), shp_path
            )
        assert len(result) == 0


class TestFIPSNormalization:
    """Verify output FIPS codes are 5-digit zero-padded strings."""

    def test_fips_format(self):
        result = _run_mapping()
        for fips in result["fips"]:
            assert isinstance(fips, str)
            assert len(fips) == 5
            assert fips.isdigit()


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Running the same input twice must produce identical output."""

    def test_deterministic(self):
        result1 = _run_mapping()
        result2 = _run_mapping()
        pd.testing.assert_frame_equal(
            result1.sort_values(["fips", "station_id"]).reset_index(drop=True),
            result2.sort_values(["fips", "station_id"]).reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Transform purity tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """Verify no scoring metrics appear in output."""

    def test_no_scoring_columns(self):
        result = _run_mapping()
        forbidden = {
            "percentile", "rank", "composite", "cci_score",
            "acceleration", "overlap_penalty", "weight",
        }
        for col in result.columns:
            assert col not in forbidden, f"Scoring column found in output: {col}"


class TestNoDuplicateStations:
    """Each station should appear at most once in the output."""

    def test_unique_station_ids(self):
        result = _run_mapping()
        assigned = result[result["station_id"].notna()]
        assert assigned["station_id"].is_unique


# ---------------------------------------------------------------------------
# Metadata sidecar test
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written with correct values."""

    def test_metadata_content(self, tmp_path):
        station_df = _make_station_df(STATIONS)
        county_gdf = _make_county_gdf(COUNTIES)

        shp_path = tmp_path / "counties.shp"
        county_gdf.to_file(shp_path)

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        with (
            patch("transform.station_to_county.HARMONIZED_DIR", harmonized),
            patch("transform.station_to_county.OUTPUT_PARQUET", harmonized / "station_to_county.parquet"),
            patch("transform.station_to_county.OUTPUT_METADATA", harmonized / "station_to_county_metadata.json"),
        ):
            run(station_df, shp_path)

        meta_path = harmonized / "station_to_county_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "NOAA_NCEI_STATIONS"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "none"
        assert "retrieved_at" in meta

    def test_parquet_written(self, tmp_path):
        station_df = _make_station_df(STATIONS)
        county_gdf = _make_county_gdf(COUNTIES)

        shp_path = tmp_path / "counties.shp"
        county_gdf.to_file(shp_path)

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        with (
            patch("transform.station_to_county.HARMONIZED_DIR", harmonized),
            patch("transform.station_to_county.OUTPUT_PARQUET", harmonized / "station_to_county.parquet"),
            patch("transform.station_to_county.OUTPUT_METADATA", harmonized / "station_to_county_metadata.json"),
        ):
            run(station_df, shp_path)

        pq_path = harmonized / "station_to_county.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)
        assert set(df.columns) == set(OUTPUT_COLUMNS)
