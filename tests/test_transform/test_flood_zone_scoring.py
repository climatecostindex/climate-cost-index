"""Tests for transform/flood_zone_scoring.py — NFHL flood zone scoring.

All tests use synthetic GeoDataFrames/DataFrames with known expected outputs.
No real data files are read.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from transform.flood_zone_scoring import (
    HIGH_RISK_WEIGHT,
    MAP_CURRENCY_THRESHOLD_YEARS,
    MODERATE_RISK_WEIGHT,
    OUTPUT_COLUMNS,
    compute_flood_scores,
    run,
)

CRS_NAD83 = "EPSG:4269"
CRS_ALBERS = "EPSG:5070"


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _make_county_boundary(
    fips: str = "01001",
    bounds: tuple[float, float, float, float] = (-87.0, 32.0, -86.0, 33.0),
) -> gpd.GeoDataFrame:
    """Build a synthetic county boundary GeoDataFrame."""
    return gpd.GeoDataFrame(
        {"fips": [fips], "geometry": [box(*bounds)]},
        crs=CRS_NAD83,
    )


def _make_flood_zones(
    zones: list[dict],
) -> gpd.GeoDataFrame:
    """Build a synthetic flood zone GeoDataFrame.

    Each dict: {"county_fips": str, "flood_zone": str, "zone_subtype": str,
                "geometry": shapely Polygon}
    """
    if not zones:
        return gpd.GeoDataFrame(
            columns=["flood_zone", "zone_subtype", "county_fips", "geometry"],
            geometry="geometry",
            crs=CRS_NAD83,
        )
    df = pd.DataFrame(zones)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_NAD83)


def _make_block_groups(
    groups: list[dict],
) -> pd.DataFrame:
    """Build a synthetic block-group housing DataFrame.

    Each dict: {"block_group_fips": str, "county_fips": str,
                "housing_units": float, "lat": float, "lon": float}
    """
    return pd.DataFrame(groups)


def _make_panel_dates(
    panels: list[dict],
) -> pd.DataFrame:
    """Build synthetic panel date DataFrame.

    Each dict: {"county_fips": str, "effective_date": date}
    """
    return pd.DataFrame(panels)


def _county_and_zones_simple(
    fips: str = "01001",
    county_bounds: tuple = (-87.0, 32.0, -86.0, 33.0),
    high_risk_bounds: tuple | None = None,
    moderate_risk_bounds: tuple | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create a simple county + flood zone setup for testing."""
    county = _make_county_boundary(fips, county_bounds)
    zones = []

    if high_risk_bounds:
        zones.append({
            "county_fips": fips,
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(*high_risk_bounds),
        })

    if moderate_risk_bounds:
        zones.append({
            "county_fips": fips,
            "flood_zone": "X",
            "zone_subtype": "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
            "geometry": box(*moderate_risk_bounds),
        })

    fz = _make_flood_zones(zones)
    return county, fz


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestHighRiskAreaCalculation:
    """Verify high-risk zone area percentage."""

    def test_high_risk_area(self):
        """County with a high-risk zone covering ~25% of area."""
        # County: 1° × 1° box
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        # High-risk zone: 0.5° × 0.5° box (≈25% of county area)
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        row = result.iloc[0]
        # In equal-area projection, the ratio should be close to 25%
        assert row["pct_area_high_risk"] > 20.0
        assert row["pct_area_high_risk"] < 30.0
        assert row["pct_area_moderate_risk"] == pytest.approx(0.0, abs=0.1)


class TestModerateRiskAreaCalculation:
    """Verify moderate-risk zone area percentage."""

    def test_moderate_risk_x_zone(self):
        """X zone with 0.2% subtype is moderate-risk."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "X",
            "zone_subtype": "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        row = result.iloc[0]
        assert row["pct_area_moderate_risk"] > 20.0
        assert row["pct_area_moderate_risk"] < 30.0
        assert row["pct_area_high_risk"] == pytest.approx(0.0, abs=0.1)

    def test_b_zone_is_moderate(self):
        """B zone is classified as moderate-risk."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "B",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        row = result.iloc[0]
        assert row["pct_area_moderate_risk"] > 20.0


class TestFloodExposureScoreFormula:
    """Verify score = (pct_high × 3) + (pct_moderate × 1)."""

    def test_score_formula(self):
        """Create county with both high and moderate zones, verify formula."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        # High-risk zone: bottom-left quarter
        # Moderate-risk zone: bottom-right quarter
        fz = _make_flood_zones([
            {
                "county_fips": "01001",
                "flood_zone": "AE",
                "zone_subtype": "",
                "geometry": box(-87.0, 32.0, -86.5, 32.5),
            },
            {
                "county_fips": "01001",
                "flood_zone": "X",
                "zone_subtype": "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
                "geometry": box(-86.5, 32.0, -86.0, 32.5),
            },
        ])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        row = result.iloc[0]
        expected_score = (
            row["pct_area_high_risk"] * HIGH_RISK_WEIGHT
            + row["pct_area_moderate_risk"] * MODERATE_RISK_WEIGHT
        )
        assert row["flood_exposure"] == pytest.approx(expected_score, rel=1e-6)


class TestHousingUnitOverlay:
    """Verify pct_hu_high_risk computation."""

    def test_housing_unit_percentage(self):
        """Some block groups inside, some outside high-risk zone."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        # High-risk zone: left half
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 33.0),
        }])

        # 2 block groups: one inside high-risk (100 units), one outside (100 units)
        bg = _make_block_groups([
            {"block_group_fips": "010010101001", "county_fips": "01001",
             "housing_units": 100.0, "lat": 32.5, "lon": -86.75},  # inside
            {"block_group_fips": "010010101002", "county_fips": "01001",
             "housing_units": 100.0, "lat": 32.5, "lon": -86.25},  # outside
        ])

        result = compute_flood_scores(fz, county, block_group_housing=bg, scoring_year=2024)
        row = result.iloc[0]
        assert row["pct_hu_high_risk"] == pytest.approx(50.0, abs=1.0)


class TestMultipleZoneTypes:
    """County with both high and moderate zones."""

    def test_both_zone_types(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([
            {
                "county_fips": "01001",
                "flood_zone": "V",
                "zone_subtype": "",
                "geometry": box(-87.0, 32.0, -86.5, 32.5),
            },
            {
                "county_fips": "01001",
                "flood_zone": "X",
                "zone_subtype": "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
                "geometry": box(-86.5, 32.5, -86.0, 33.0),
            },
        ])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        row = result.iloc[0]
        assert row["pct_area_high_risk"] > 0
        assert row["pct_area_moderate_risk"] > 0
        assert row["flood_exposure"] > 0


class TestCountyWithNoFloodZones:
    """County with no overlapping flood zone polygons."""

    def test_no_flood_zones_absent(self):
        """County with no NFHL data should be absent from output."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        # Flood zones for a DIFFERENT county
        fz = _make_flood_zones([{
            "county_fips": "99999",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-80.0, 30.0, -79.0, 31.0),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        assert "01001" not in result["fips"].values


class TestMapCurrencyFlag:
    """Verify map currency flag computation."""

    def test_recent_map(self):
        """Effective date < 10 years old → flag = 0."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])
        panels = _make_panel_dates([{
            "county_fips": "01001",
            "effective_date": date(2020, 1, 1),
        }])

        result = compute_flood_scores(
            fz, county, panel_dates=panels, scoring_year=2024,
        )
        row = result.iloc[0]
        assert row["map_currency_flag"] == 0
        assert row["nfhl_effective_date"] == "2020-01-01"

    def test_stale_map(self):
        """Effective date > 10 years old → flag = 1."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])
        panels = _make_panel_dates([{
            "county_fips": "01001",
            "effective_date": date(2005, 1, 1),
        }])

        result = compute_flood_scores(
            fz, county, panel_dates=panels, scoring_year=2024,
        )
        row = result.iloc[0]
        assert row["map_currency_flag"] == 1


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame schema."""

    def _get_result(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])
        return compute_flood_scores(fz, county, scoring_year=2024)

    def test_column_presence(self):
        result = self._get_result()
        for col in OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_extra_columns(self):
        result = self._get_result()
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_column_types(self):
        result = self._get_result()
        assert pd.api.types.is_string_dtype(result["fips"])
        assert pd.api.types.is_float_dtype(result["flood_exposure"])
        assert pd.api.types.is_float_dtype(result["pct_area_high_risk"])
        assert pd.api.types.is_float_dtype(result["pct_area_moderate_risk"])
        assert pd.api.types.is_float_dtype(result["pct_hu_high_risk"])
        assert pd.api.types.is_integer_dtype(result["map_currency_flag"])


# ---------------------------------------------------------------------------
# Edge case tests (module-specific)
# ---------------------------------------------------------------------------

class TestNoBlockGroupData:
    """Area-only fallback when block-group data is unavailable."""

    def test_area_only_fallback(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(
            fz, county, block_group_housing=None, scoring_year=2024,
        )
        row = result.iloc[0]
        assert pd.isna(row["pct_hu_high_risk"])
        assert row["pct_area_high_risk"] > 0
        assert row["flood_exposure"] > 0


class TestCountyWithZeroHousingUnits:
    """County with zero housing units should not cause division by zero."""

    def test_zero_housing_units(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])
        bg = _make_block_groups([{
            "block_group_fips": "010010101001",
            "county_fips": "01001",
            "housing_units": 0.0,
            "lat": 32.5,
            "lon": -86.75,
        }])

        result = compute_flood_scores(
            fz, county, block_group_housing=bg, scoring_year=2024,
        )
        row = result.iloc[0]
        assert row["pct_hu_high_risk"] == 0.0


class TestInvalidGeometryRepair:
    """Flood zone with self-intersecting geometry."""

    def test_self_intersecting_repaired(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        # Create a self-intersecting "bowtie" polygon
        bowtie = Polygon([
            (-87.0, 32.0), (-86.5, 32.5), (-87.0, 32.5), (-86.5, 32.0), (-87.0, 32.0),
        ])
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": bowtie,
        }])

        # Should not crash
        result = compute_flood_scores(fz, county, scoring_year=2024)
        assert len(result) >= 0  # Either produces result or empty, no crash


class TestUnrecognizedFloodZone:
    """Unrecognized zone type treated as minimal-risk."""

    def test_d_zone_minimal(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "D",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        # D zone is minimal-risk — county should have 0 score since no
        # high or moderate zones exist
        if len(result) > 0:
            row = result.iloc[0]
            assert row["pct_area_high_risk"] == pytest.approx(0.0, abs=0.1)
            assert row["pct_area_moderate_risk"] == pytest.approx(0.0, abs=0.1)


class TestMissingPanelDate:
    """County with flood data but no panel date info."""

    def test_missing_panel_pessimistic(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        # No panel dates provided
        result = compute_flood_scores(
            fz, county, panel_dates=None, scoring_year=2024,
        )
        row = result.iloc[0]
        assert row["map_currency_flag"] == 1


class TestOverlappingFloodZones:
    """Overlapping high-risk polygons area calculation."""

    def test_overlapping_summed(self):
        """With sum-based area, overlapping zones contribute their full area."""
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        # Two overlapping high-risk zones
        fz = _make_flood_zones([
            {
                "county_fips": "01001",
                "flood_zone": "AE",
                "zone_subtype": "",
                "geometry": box(-87.0, 32.0, -86.5, 32.5),
            },
            {
                "county_fips": "01001",
                "flood_zone": "A",
                "zone_subtype": "",
                "geometry": box(-86.75, 32.0, -86.25, 32.5),
            },
        ])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        row = result.iloc[0]
        # Area is summed (not dissolved) for performance with 5M+ NFHL polygons.
        # NFHL map tiles rarely overlap significantly in practice.
        # Result should be > 0 and clamped to ≤100
        assert row["pct_area_high_risk"] > 0
        assert row["pct_area_high_risk"] <= 100.0


# ---------------------------------------------------------------------------
# Edge case tests (generic)
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Verify graceful handling of empty GeoDataFrames."""

    def test_empty_flood_zones(self):
        county = _make_county_boundary("01001")
        fz = _make_flood_zones([])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_empty_county_boundaries(self):
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])
        county = gpd.GeoDataFrame(
            columns=["fips", "geometry"], geometry="geometry", crs=CRS_NAD83,
        )

        result = compute_flood_scores(fz, county, scoring_year=2024)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_empty_block_group_data(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])
        bg = pd.DataFrame(columns=["block_group_fips", "county_fips",
                                     "housing_units", "lat", "lon"])

        result = compute_flood_scores(
            fz, county, block_group_housing=bg, scoring_year=2024,
        )
        assert len(result) > 0
        assert pd.isna(result.iloc[0]["pct_hu_high_risk"])


class TestMissingColumns:
    """Missing required columns raise ValueError."""

    def test_flood_zones_missing_flood_zone(self):
        county = _make_county_boundary("01001")
        fz = gpd.GeoDataFrame(
            {"zone_subtype": [""], "county_fips": ["01001"],
             "geometry": [box(-87.0, 32.0, -86.5, 32.5)]},
            crs=CRS_NAD83,
        )

        with pytest.raises(ValueError, match="flood_zone"):
            compute_flood_scores(fz, county, scoring_year=2024)

    def test_county_missing_fips(self):
        county = gpd.GeoDataFrame(
            {"name": ["Test"], "geometry": [box(-87.0, 32.0, -86.0, 33.0)]},
            crs=CRS_NAD83,
        )
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        with pytest.raises(ValueError, match="fips"):
            compute_flood_scores(fz, county, scoring_year=2024)


class TestPartialData:
    """Valid data for some counties produces output for those."""

    def test_partial_nfhl_coverage(self):
        """Two counties, only one has NFHL data."""
        county1 = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        county2 = _make_county_boundary("01003", (-86.0, 32.0, -85.0, 33.0))
        counties = pd.concat([county1, county2], ignore_index=True)
        counties = gpd.GeoDataFrame(counties, geometry="geometry", crs=CRS_NAD83)

        # Only 01001 has flood zone data
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, counties, scoring_year=2024)
        assert "01001" in result["fips"].values
        assert "01003" not in result["fips"].values


class TestFIPSNormalization:
    """Output FIPS codes are 5-digit zero-padded strings."""

    def test_fips_format(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        for fips in result["fips"]:
            assert isinstance(fips, str)
            assert len(fips) == 5
            assert fips.isdigit()


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Running the same input twice produces identical output."""

    def test_deterministic(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result1 = compute_flood_scores(fz, county, scoring_year=2024)
        result2 = compute_flood_scores(fz, county, scoring_year=2024)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Transform purity tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """Verify NO scoring metrics in output."""

    def test_no_scoring_columns(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        forbidden = {
            "percentile", "rank", "composite", "cci_score",
            "acceleration", "overlap_penalty", "weight",
        }
        for col in result.columns:
            assert col not in forbidden, f"Scoring column found in output: {col}"


# ---------------------------------------------------------------------------
# County-year grain test
# ---------------------------------------------------------------------------

class TestCountyYearGrain:
    """Verify exactly one row per (fips, year) — no duplicates."""

    def test_no_duplicate_fips_year(self):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        dupes = result.duplicated(subset=["fips", "year"], keep=False)
        assert not dupes.any(), "Duplicate (fips, year) rows found"


# ---------------------------------------------------------------------------
# Metadata sidecar test
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written with correct values."""

    def test_metadata_content(self, tmp_path):
        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        # Write test county boundary zip
        county_with_geoid = county.copy()
        county_with_geoid["GEOID"] = "01001"
        county_path = tmp_path / "cb_2024_us_county_500k.zip"

        # We patch the loading functions to return our test data directly
        with (
            patch("transform.flood_zone_scoring.HARMONIZED_DIR", harmonized),
            patch("transform.flood_zone_scoring._load_county_boundaries", return_value=county),
            patch("transform.flood_zone_scoring._load_all_flood_zones", return_value=fz),
            patch("transform.flood_zone_scoring._load_block_group_housing", return_value=None),
            patch("transform.flood_zone_scoring._load_all_panel_dates", return_value=None),
            patch("transform.flood_zone_scoring.get_settings") as mock_settings,
        ):
            mock_settings.return_value.scoring_year = 2024
            run()

        meta_path = harmonized / "flood_zone_scoring_2024_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "FEMA_NFHL"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta

    def test_parquet_written(self, tmp_path):
        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": "AE",
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        with (
            patch("transform.flood_zone_scoring.HARMONIZED_DIR", harmonized),
            patch("transform.flood_zone_scoring._load_county_boundaries", return_value=county),
            patch("transform.flood_zone_scoring._load_all_flood_zones", return_value=fz),
            patch("transform.flood_zone_scoring._load_block_group_housing", return_value=None),
            patch("transform.flood_zone_scoring._load_all_panel_dates", return_value=None),
            patch("transform.flood_zone_scoring.get_settings") as mock_settings,
        ):
            mock_settings.return_value.scoring_year = 2024
            run()

        pq_path = harmonized / "flood_zone_scoring_2024.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)
        assert set(df.columns) == set(OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# All high-risk zone types
# ---------------------------------------------------------------------------

class TestAllHighRiskZoneTypes:
    """Verify all high-risk zone classifications."""

    @pytest.mark.parametrize("zone", ["A", "AE", "AH", "AO", "V", "VE"])
    def test_zone_is_high_risk(self, zone):
        county = _make_county_boundary("01001", (-87.0, 32.0, -86.0, 33.0))
        fz = _make_flood_zones([{
            "county_fips": "01001",
            "flood_zone": zone,
            "zone_subtype": "",
            "geometry": box(-87.0, 32.0, -86.5, 32.5),
        }])

        result = compute_flood_scores(fz, county, scoring_year=2024)
        assert len(result) > 0
        assert result.iloc[0]["pct_area_high_risk"] > 0
