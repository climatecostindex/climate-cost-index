"""Tests for transform/air_quality_scoring.py — county-level AQI + smoke days.

All tests use synthetic DataFrames/GeoDataFrames with known expected outputs.
No real data files are read.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from transform.air_quality_scoring import (
    AQI_UNHEALTHY_THRESHOLD,
    AQI_VERY_UNHEALTHY_THRESHOLD,
    OUTPUT_COLUMNS,
    SPIKE_DETECTION_MULTIPLIER,
    SPIKE_DETECTION_WINDOW_DAYS,
    compute_air_quality_scores,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _make_daily_readings(
    monitor_id: str = "M001",
    year: int = 2020,
    n_days: int = 365,
    pm25_value: float = 10.0,
    aqi_value: float = 50.0,
    start_month: int = 1,
    start_day: int = 1,
) -> pd.DataFrame:
    """Build synthetic daily readings for one monitor."""
    dates = pd.date_range(
        start=date(year, start_month, start_day),
        periods=n_days,
        freq="D",
    )
    return pd.DataFrame({
        "monitor_id": monitor_id,
        "date": dates,
        "pm25_value": pm25_value,
        "aqi_value": aqi_value,
    })


def _make_monitor_county_map(mappings: list[dict]) -> pd.DataFrame:
    """Build a synthetic monitor-to-county mapping.

    Each dict: {"monitor_id": str, "fips": str}
    """
    return pd.DataFrame(mappings)


def _make_county_boundaries(
    fips_list: list[str],
    polygons: list[Polygon] | None = None,
) -> gpd.GeoDataFrame:
    """Build synthetic county boundary GeoDataFrame."""
    if polygons is None:
        # Default: non-overlapping 1x1 degree boxes along equator
        polygons = [
            box(i, 0, i + 1, 1) for i in range(len(fips_list))
        ]
    return gpd.GeoDataFrame(
        {"fips": fips_list, "geometry": polygons},
        crs="EPSG:4269",
    )


def _make_hms_plumes(
    dates: list,
    polygons: list[Polygon],
    densities: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Build synthetic HMS plume GeoDataFrame."""
    if densities is None:
        densities = ["medium"] * len(dates)
    return gpd.GeoDataFrame(
        {"date": dates, "geometry": polygons, "density": densities},
        crs="EPSG:4269",
    )


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestPM25AnnualAverage:
    """Verify PM2.5 annual average calculation."""

    def test_constant_pm25(self):
        """PM2.5 annual avg equals the constant daily value."""
        readings = _make_daily_readings(pm25_value=12.5, aqi_value=50.0)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        row = result[(result["fips"] == "01001") & (result["year"] == 2020)]

        assert len(row) == 1
        assert row.iloc[0]["pm25_annual_avg"] == pytest.approx(12.5, rel=1e-6)

    def test_varying_pm25(self):
        """PM2.5 annual avg is the mean of varying daily values."""
        readings = _make_daily_readings(n_days=4)
        readings["pm25_value"] = [5.0, 10.0, 15.0, 20.0]
        readings["aqi_value"] = [30.0, 40.0, 50.0, 60.0]
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        assert result.iloc[0]["pm25_annual_avg"] == pytest.approx(12.5, rel=1e-6)


class TestAQIUnhealthyDays:
    """Verify AQI unhealthy day counts."""

    def test_count_unhealthy_days(self):
        """Count days with AQI > 100."""
        readings = _make_daily_readings(n_days=10)
        # 3 days above 100, 2 days above 150
        readings["aqi_value"] = [50, 80, 101, 105, 151, 200, 90, 60, 40, 30]
        readings["pm25_value"] = 10.0
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        # AQI > 100: indices 2,3,4,5 = 4 days
        assert result.iloc[0]["aqi_unhealthy_days"] == 4

    def test_count_very_unhealthy_days(self):
        """Count days with AQI > 150."""
        readings = _make_daily_readings(n_days=10)
        readings["aqi_value"] = [50, 80, 101, 105, 151, 200, 90, 60, 40, 30]
        readings["pm25_value"] = 10.0
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        # AQI > 150: indices 4,5 = 2 days
        assert result.iloc[0]["aqi_very_unhealthy_days"] == 2

    def test_zero_unhealthy_days(self):
        """All AQI values below threshold => 0 unhealthy days."""
        readings = _make_daily_readings(aqi_value=50.0)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        assert result.iloc[0]["aqi_unhealthy_days"] == 0
        assert result.iloc[0]["aqi_very_unhealthy_days"] == 0


class TestCountyDailyAggregation:
    """Verify multi-monitor county-day aggregation."""

    def test_pm25_is_mean_of_monitors(self):
        """County daily PM2.5 = mean of monitors in the same county."""
        r1 = _make_daily_readings(monitor_id="M001", n_days=1, pm25_value=10.0, aqi_value=40.0)
        r2 = _make_daily_readings(monitor_id="M002", n_days=1, pm25_value=20.0, aqi_value=80.0)
        readings = pd.concat([r1, r2], ignore_index=True)
        mapping = _make_monitor_county_map([
            {"monitor_id": "M001", "fips": "01001"},
            {"monitor_id": "M002", "fips": "01001"},
        ])

        result = compute_air_quality_scores(readings, mapping)
        # Mean of 10 and 20
        assert result.iloc[0]["pm25_annual_avg"] == pytest.approx(15.0, rel=1e-6)

    def test_aqi_is_max_of_monitors(self):
        """County daily AQI = max across monitors."""
        r1 = _make_daily_readings(monitor_id="M001", n_days=5, pm25_value=10.0, aqi_value=40.0)
        r2 = _make_daily_readings(monitor_id="M002", n_days=5, pm25_value=10.0, aqi_value=110.0)
        readings = pd.concat([r1, r2], ignore_index=True)
        mapping = _make_monitor_county_map([
            {"monitor_id": "M001", "fips": "01001"},
            {"monitor_id": "M002", "fips": "01001"},
        ])

        result = compute_air_quality_scores(readings, mapping)
        # Max AQI is 110 for every day => all 5 days are unhealthy
        assert result.iloc[0]["aqi_unhealthy_days"] == 5


class TestSmokeDayHMS:
    """Verify HMS-based smoke day identification."""

    def test_hms_plume_with_elevated_pm25(self):
        """HMS plume intersecting county + elevated PM2.5 = smoke day."""
        # 60 days of baseline PM2.5 = 10, then 5 days with PM2.5 = 25
        n_baseline = 60
        n_elevated = 5
        readings = _make_daily_readings(
            n_days=n_baseline + n_elevated, pm25_value=10.0, aqi_value=50.0,
        )
        readings.loc[n_baseline:, "pm25_value"] = 25.0
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        # County polygon: box(0, 0, 1, 1)
        county_bounds = _make_county_boundaries(["01001"], [box(0, 0, 1, 1)])

        # HMS plumes covering the county for the 5 elevated days
        elevated_dates = readings["date"].iloc[n_baseline:].tolist()
        plume_poly = box(-1, -1, 2, 2)  # covers the county
        hms = _make_hms_plumes(
            dates=elevated_dates,
            polygons=[plume_poly] * n_elevated,
        )

        result = compute_air_quality_scores(readings, mapping, hms, county_bounds)
        row = result[result["fips"] == "01001"]
        assert len(row) == 1
        assert row.iloc[0]["smoke_days"] >= 1
        assert row.iloc[0]["smoke_day_method"] == "hms"

    def test_hms_plume_but_low_pm25(self):
        """HMS plume intersects county but PM2.5 is below baseline = NOT a smoke day."""
        # Constant PM2.5 = 10 for all days
        readings = _make_daily_readings(n_days=90, pm25_value=10.0, aqi_value=50.0)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        county_bounds = _make_county_boundaries(["01001"], [box(0, 0, 1, 1)])

        # HMS plumes for last 5 days — but PM2.5 stays at baseline
        plume_dates = readings["date"].iloc[-5:].tolist()
        hms = _make_hms_plumes(
            dates=plume_dates,
            polygons=[box(-1, -1, 2, 2)] * 5,
        )

        result = compute_air_quality_scores(readings, mapping, hms, county_bounds)
        row = result[result["fips"] == "01001"]
        # PM2.5 stays constant at baseline, so no smoke days
        assert row.iloc[0]["smoke_days"] == 0

    def test_hms_no_plume_intersection(self):
        """HMS data available but plume doesn't intersect county = NOT a smoke day."""
        # Elevated PM2.5 that would trigger spike detection
        readings = _make_daily_readings(n_days=60, pm25_value=10.0, aqi_value=50.0)
        readings.loc[55:, "pm25_value"] = 50.0  # spike
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        county_bounds = _make_county_boundaries(["01001"], [box(0, 0, 1, 1)])

        # HMS plumes far from the county
        plume_dates = readings["date"].iloc[55:].tolist()
        hms = _make_hms_plumes(
            dates=plume_dates,
            polygons=[box(100, 100, 101, 101)] * len(plume_dates),  # far away
        )

        result = compute_air_quality_scores(readings, mapping, hms, county_bounds)
        row = result[result["fips"] == "01001"]
        # HMS is authoritative — no plume intersection means no smoke day,
        # even though PM2.5 is elevated
        assert row.iloc[0]["smoke_days"] == 0


class TestSmokeDaySpike:
    """Verify spike-detection fallback for smoke days."""

    def test_spike_detection_no_hms(self):
        """Elevated PM2.5 with no HMS data = smoke day via spike detection."""
        # 45 days at baseline=10, then 5 days at 25 (> 1.5 × 10 = 15)
        n_baseline = 45
        n_spike = 5
        readings = _make_daily_readings(
            n_days=n_baseline + n_spike, pm25_value=10.0, aqi_value=50.0,
        )
        readings.loc[n_baseline:, "pm25_value"] = 25.0
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping, hms_plumes=None)
        row = result[result["fips"] == "01001"]
        assert row.iloc[0]["smoke_days"] >= 1
        assert row.iloc[0]["smoke_day_method"] == "spike_detection"

    def test_spike_not_used_when_hms_available(self):
        """When HMS data is available for a date, spike detection is NOT used."""
        # Set up PM2.5 spike on a date that has HMS data (but no plume intersection)
        readings = _make_daily_readings(n_days=60, pm25_value=10.0, aqi_value=50.0)
        readings.loc[55:, "pm25_value"] = 50.0  # big spike
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        county_bounds = _make_county_boundaries(["01001"], [box(0, 0, 1, 1)])

        # HMS data exists for the spike dates but plumes don't intersect
        spike_dates = readings["date"].iloc[55:].tolist()
        hms = _make_hms_plumes(
            dates=spike_dates,
            polygons=[box(100, 100, 101, 101)] * len(spike_dates),
        )

        result = compute_air_quality_scores(readings, mapping, hms, county_bounds)
        row = result[result["fips"] == "01001"]
        # Despite the PM2.5 spike, HMS says no plume intersection → not a smoke day
        assert row.iloc[0]["smoke_days"] == 0

    def test_zero_smoke_days(self):
        """No spikes and no HMS = zero smoke days."""
        readings = _make_daily_readings(n_days=90, pm25_value=10.0, aqi_value=50.0)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping, hms_plumes=None)
        row = result[result["fips"] == "01001"]
        assert row.iloc[0]["smoke_days"] == 0


class TestMultiCountyMultiYear:
    """Verify independent results per county per year."""

    def test_two_counties_two_years(self):
        """Two counties × two years produce 4 independent rows."""
        r1_2020 = _make_daily_readings(
            monitor_id="M001", year=2020, n_days=100, pm25_value=8.0, aqi_value=40.0,
        )
        r1_2021 = _make_daily_readings(
            monitor_id="M001", year=2021, n_days=100, pm25_value=12.0, aqi_value=60.0,
        )
        r2_2020 = _make_daily_readings(
            monitor_id="M002", year=2020, n_days=100, pm25_value=20.0, aqi_value=90.0,
        )
        r2_2021 = _make_daily_readings(
            monitor_id="M002", year=2021, n_days=100, pm25_value=25.0, aqi_value=120.0,
        )
        readings = pd.concat([r1_2020, r1_2021, r2_2020, r2_2021], ignore_index=True)
        mapping = _make_monitor_county_map([
            {"monitor_id": "M001", "fips": "01001"},
            {"monitor_id": "M002", "fips": "02002"},
        ])

        result = compute_air_quality_scores(readings, mapping)
        assert len(result) == 4

        # Check county 1, year 2020
        c1y20 = result[(result["fips"] == "01001") & (result["year"] == 2020)]
        assert c1y20.iloc[0]["pm25_annual_avg"] == pytest.approx(8.0, rel=1e-6)

        # Check county 2, year 2021
        c2y21 = result[(result["fips"] == "02002") & (result["year"] == 2021)]
        assert c2y21.iloc[0]["pm25_annual_avg"] == pytest.approx(25.0, rel=1e-6)
        # AQI 120 > 100 on all days
        assert c2y21.iloc[0]["aqi_unhealthy_days"] == 100


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame schema."""

    def test_column_presence(self):
        """Output contains all expected columns."""
        readings = _make_daily_readings(n_days=10)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_no_extra_columns(self):
        """Output contains ONLY the specified columns."""
        readings = _make_daily_readings(n_days=10)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_column_types(self):
        """Verify column data types."""
        readings = _make_daily_readings(n_days=10)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)

        assert pd.api.types.is_string_dtype(result["fips"])
        assert pd.api.types.is_integer_dtype(result["year"])
        assert pd.api.types.is_float_dtype(result["pm25_annual_avg"])
        assert pd.api.types.is_integer_dtype(result["aqi_unhealthy_days"])
        assert pd.api.types.is_integer_dtype(result["aqi_very_unhealthy_days"])
        assert pd.api.types.is_integer_dtype(result["smoke_days"])
        assert pd.api.types.is_string_dtype(result["smoke_day_method"])


# ---------------------------------------------------------------------------
# Edge case tests (module-specific)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Module-specific edge cases."""

    def test_no_hms_data(self):
        """Pass hms_plumes=None — all smoke days use spike detection."""
        readings = _make_daily_readings(n_days=50, pm25_value=10.0, aqi_value=50.0)
        # Add a spike
        readings.loc[45:, "pm25_value"] = 25.0
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping, hms_plumes=None)
        row = result[result["fips"] == "01001"]
        assert row.iloc[0]["smoke_day_method"] == "spike_detection"

    def test_county_with_no_monitors(self):
        """Counties without monitors should be absent from output."""
        readings = _make_daily_readings(monitor_id="M001", n_days=10)
        # M001 maps to 01001 — county 02002 has no monitor
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        assert "02002" not in result["fips"].values

    def test_monitor_not_in_mapping(self):
        """Monitor in readings but not in mapping => excluded, no crash."""
        r1 = _make_daily_readings(monitor_id="M001", n_days=10)
        r2 = _make_daily_readings(monitor_id="M999", n_days=10)  # not mapped
        readings = pd.concat([r1, r2], ignore_index=True)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        assert len(result) == 1
        assert result.iloc[0]["fips"] == "01001"

    def test_negative_pm25_treated_as_nan(self):
        """Negative PM2.5 values should be treated as NaN."""
        readings = _make_daily_readings(n_days=5, pm25_value=10.0, aqi_value=50.0)
        readings.loc[0, "pm25_value"] = -5.0  # negative => NaN
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        # Mean should be over the 4 valid values: (10+10+10+10)/4 = 10
        assert result.iloc[0]["pm25_annual_avg"] == pytest.approx(10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge cases (generic)
# ---------------------------------------------------------------------------

class TestGenericEdgeCases:
    """Generic edge case handling."""

    def test_empty_readings(self):
        """Empty readings DataFrame returns empty result."""
        empty = pd.DataFrame(columns=["monitor_id", "date", "pm25_value", "aqi_value"])
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(empty, mapping)
        assert result.empty
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_empty_monitor_map(self):
        """Empty monitor mapping returns empty result."""
        readings = _make_daily_readings(n_days=10)
        empty_map = pd.DataFrame(columns=["monitor_id", "fips"])

        result = compute_air_quality_scores(readings, empty_map)
        assert result.empty

    def test_missing_readings_columns(self):
        """Missing required columns raises ValueError."""
        bad_df = pd.DataFrame({"monitor_id": ["M001"], "date": [date(2020, 1, 1)]})
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        with pytest.raises(ValueError, match="missing columns"):
            compute_air_quality_scores(bad_df, mapping)

    def test_missing_map_columns(self):
        """Missing required columns in mapping raises ValueError."""
        readings = _make_daily_readings(n_days=1)
        bad_map = pd.DataFrame({"monitor_id": ["M001"]})  # missing fips

        with pytest.raises(ValueError, match="missing columns"):
            compute_air_quality_scores(readings, bad_map)

    def test_partial_data_produces_partial_output(self):
        """Counties with data produce output even if others have none."""
        r1 = _make_daily_readings(monitor_id="M001", n_days=10, pm25_value=10.0, aqi_value=50.0)
        mapping = _make_monitor_county_map([
            {"monitor_id": "M001", "fips": "01001"},
            {"monitor_id": "M002", "fips": "02002"},  # M002 has no readings
        ])

        result = compute_air_quality_scores(r1, mapping)
        assert "01001" in result["fips"].values
        assert "02002" not in result["fips"].values

    def test_fips_normalization(self):
        """FIPS codes should be 5-digit zero-padded strings."""
        readings = _make_daily_readings(n_days=10)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "1001"}])

        result = compute_air_quality_scores(readings, mapping)
        assert result.iloc[0]["fips"] == "01001"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify reproducibility."""

    def test_same_input_same_output(self):
        """Same input produces identical output on two runs."""
        readings = _make_daily_readings(n_days=100, pm25_value=15.0, aqi_value=65.0)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        r1 = compute_air_quality_scores(readings, mapping)
        r2 = compute_air_quality_scores(readings, mapping)

        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Standard transform tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """Verify no Phase 3 scoring metrics in output."""

    def test_no_scoring_metrics(self):
        """Output should NOT contain percentiles, ranks, or composite scores."""
        readings = _make_daily_readings(n_days=100)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        scoring_cols = {"percentile", "rank", "score", "weight", "penalty", "acceleration"}
        for col in result.columns:
            for keyword in scoring_cols:
                assert keyword not in col.lower(), (
                    f"Column '{col}' appears to be a scoring metric"
                )


class TestMetadataSidecar:
    """Verify metadata JSON sidecar."""

    def test_metadata_values(self, tmp_path):
        """Metadata contains correct source, confidence, attribution."""
        from transform.air_quality_scoring import _write_metadata

        meta_path = tmp_path / "test_metadata.json"
        _write_metadata(meta_path, 2020)

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "EPA_AQS"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta
        assert meta["data_vintage"] == "2020"


class TestCountyYearGrain:
    """Verify exactly one row per (fips, year)."""

    def test_unique_county_year(self):
        """No duplicate (fips, year) rows."""
        readings = _make_daily_readings(n_days=365)
        mapping = _make_monitor_county_map([{"monitor_id": "M001", "fips": "01001"}])

        result = compute_air_quality_scores(readings, mapping)
        dupes = result.duplicated(subset=["fips", "year"], keep=False)
        assert not dupes.any(), "Duplicate (fips, year) rows found"

    def test_multi_monitor_single_county_year(self):
        """Two monitors in same county still produce one row per year."""
        r1 = _make_daily_readings(monitor_id="M001", n_days=100, pm25_value=10.0, aqi_value=50.0)
        r2 = _make_daily_readings(monitor_id="M002", n_days=100, pm25_value=20.0, aqi_value=80.0)
        readings = pd.concat([r1, r2], ignore_index=True)
        mapping = _make_monitor_county_map([
            {"monitor_id": "M001", "fips": "01001"},
            {"monitor_id": "M002", "fips": "01001"},
        ])

        result = compute_air_quality_scores(readings, mapping)
        assert len(result) == 1
        assert result.iloc[0]["fips"] == "01001"
