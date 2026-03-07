"""Tests for transform/degree_days.py — HDD/CDD computation + anomalies.

All tests use synthetic DataFrames with known expected outputs.
No real data files are read.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from transform.degree_days import (
    BASE_TEMP_C,
    MIN_DAYS_PER_YEAR,
    OUTPUT_COLUMNS,
    compute_degree_days,
    run,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _make_daily_obs(
    station_id: str = "USW00001",
    year: int = 2020,
    n_days: int = 365,
    tmax: float = 25.0,
    tmin: float = 15.0,
    start_month: int = 1,
    start_day: int = 1,
    q_flag_tmax: str = "",
    q_flag_tmin: str = "",
) -> pd.DataFrame:
    """Build synthetic daily observations for a single station.

    Generates n_days of data starting from {year}-{start_month}-{start_day}.
    All days have the same tmax/tmin unless overridden per-row.
    """
    dates = pd.date_range(
        start=date(year, start_month, start_day),
        periods=n_days,
        freq="D",
    )
    return pd.DataFrame({
        "station_id": station_id,
        "date": dates,
        "tmax": tmax,
        "tmin": tmin,
        "q_flag_tmax": q_flag_tmax,
        "q_flag_tmin": q_flag_tmin,
    })


def _make_station_county_map(mappings: list[dict]) -> pd.DataFrame:
    """Build a synthetic station-to-county mapping.

    Each dict: {"station_id": str, "fips": str}
    """
    return pd.DataFrame(mappings)


def _make_normals(
    station_id: str = "USW00001",
    normal_tmax: float = 20.0,
    normal_tmin: float = 10.0,
    months: list[int] | None = None,
) -> pd.DataFrame:
    """Build synthetic 12-month normals for a station."""
    if months is None:
        months = list(range(1, 13))
    return pd.DataFrame({
        "station_id": station_id,
        "month": months,
        "normal_tmax": normal_tmax,
        "normal_tmin": normal_tmin,
    })


def _full_year_obs(
    station_id: str = "USW00001",
    year: int = 2020,
    tmax: float = 25.0,
    tmin: float = 15.0,
) -> pd.DataFrame:
    """Full 365-day observation set for a station-year."""
    return _make_daily_obs(station_id=station_id, year=year, n_days=365,
                           tmax=tmax, tmin=tmin)


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestDailyDegreeDay:
    """Verify daily HDD/CDD calculation."""

    def test_hdd_only_day(self):
        """When avg_temp < BASE_TEMP_C: HDD > 0 and CDD = 0."""
        # avg = (10 + 0) / 2 = 5°C; HDD = 18.333 - 5 = 13.333; CDD = 0
        obs = _make_daily_obs(tmax=10.0, tmin=0.0, n_days=365)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result[(result["fips"] == "01001") & (result["year"] == 2020)]
        assert len(row) == 1

        expected_daily_hdd = BASE_TEMP_C - 5.0
        assert row.iloc[0]["hdd_annual"] == pytest.approx(expected_daily_hdd * 365, rel=1e-6)
        assert row.iloc[0]["cdd_annual"] == pytest.approx(0.0, abs=1e-6)

    def test_cdd_only_day(self):
        """When avg_temp > BASE_TEMP_C: CDD > 0 and HDD = 0."""
        # avg = (35 + 25) / 2 = 30°C; CDD = 30 - 18.333 = 11.667; HDD = 0
        obs = _make_daily_obs(tmax=35.0, tmin=25.0, n_days=365)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result[(result["fips"] == "01001") & (result["year"] == 2020)]

        expected_daily_cdd = 30.0 - BASE_TEMP_C
        assert row.iloc[0]["cdd_annual"] == pytest.approx(expected_daily_cdd * 365, rel=1e-6)
        assert row.iloc[0]["hdd_annual"] == pytest.approx(0.0, abs=1e-6)

    def test_neutral_day(self):
        """When avg_temp ≈ BASE_TEMP_C: both HDD and CDD ≈ 0."""
        # avg = (20.333 + 16.333) / 2 = 18.333°C = BASE_TEMP_C
        tmax = BASE_TEMP_C + 2.0
        tmin = BASE_TEMP_C - 2.0
        obs = _make_daily_obs(tmax=tmax, tmin=tmin, n_days=365)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result[(result["fips"] == "01001") & (result["year"] == 2020)]

        assert row.iloc[0]["hdd_annual"] == pytest.approx(0.0, abs=1e-6)
        assert row.iloc[0]["cdd_annual"] == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self):
        """Verify against hand-calculated values."""
        # avg = (25 + 15) / 2 = 20°C
        # HDD = max(0, 18.333 - 20) = 0
        # CDD = max(0, 20 - 18.333) = 1.667
        obs = _make_daily_obs(tmax=25.0, tmin=15.0, n_days=365)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        expected_cdd = (20.0 - BASE_TEMP_C) * 365
        assert row["cdd_annual"] == pytest.approx(expected_cdd, rel=1e-6)
        assert row["hdd_annual"] == pytest.approx(0.0, abs=1e-6)


class TestAnnualAggregation:
    """Verify annual aggregation (sum of daily values)."""

    def test_annual_sum(self):
        """hdd_annual / cdd_annual should be sum of 365 daily values."""
        # avg = (10 + 0) / 2 = 5°C, daily HDD = 13.333
        obs = _make_daily_obs(tmax=10.0, tmin=0.0, n_days=365)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        expected = (BASE_TEMP_C - 5.0) * 365
        assert result.iloc[0]["hdd_annual"] == pytest.approx(expected, rel=1e-6)


class TestCountyAveraging:
    """Verify county-level averaging across stations."""

    def test_two_stations_averaged(self):
        """County result should be arithmetic mean of station values."""
        # Station 1: avg=5°C → daily HDD=13.333, CDD=0
        # Station 2: avg=30°C → daily HDD=0, CDD=11.667
        obs1 = _full_year_obs("USW00001", tmax=10.0, tmin=0.0)
        obs2 = _full_year_obs("USW00002", tmax=35.0, tmin=25.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([
            {"station_id": "USW00001", "fips": "01001"},
            {"station_id": "USW00002", "fips": "01001"},
        ])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        station1_hdd = (BASE_TEMP_C - 5.0) * 365
        station2_hdd = 0.0
        expected_hdd = (station1_hdd + station2_hdd) / 2
        assert row["hdd_annual"] == pytest.approx(expected_hdd, rel=1e-6)

        station1_cdd = 0.0
        station2_cdd = (30.0 - BASE_TEMP_C) * 365
        expected_cdd = (station1_cdd + station2_cdd) / 2
        assert row["cdd_annual"] == pytest.approx(expected_cdd, rel=1e-6)


class TestNormalAnomaly:
    """Verify anomaly computation: observed - normal."""

    def test_anomaly_calculation(self):
        """Anomaly = observed annual - normal annual."""
        # Observed: avg = 20°C → CDD = (20 - 18.333) * 365 = 608.33, HDD = 0
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        # Normal: avg = 15°C → normal HDD per month = (18.333 - 15) * days_in_month
        normals = _make_normals(normal_tmax=20.0, normal_tmin=10.0)

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        # Observed: avg = 20°C → HDD = 0, CDD = (20 - 18.333) * 365
        observed_hdd = 0.0
        observed_cdd = (20.0 - BASE_TEMP_C) * 365

        # Normal: avg = 15°C → HDD = (18.333 - 15) * sum(days) = 3.333 * 365
        # CDD = 0
        normal_hdd = (BASE_TEMP_C - 15.0) * 365
        normal_cdd = 0.0

        expected_hdd_anomaly = observed_hdd - normal_hdd
        expected_cdd_anomaly = observed_cdd - normal_cdd

        assert row["hdd_anomaly"] == pytest.approx(expected_hdd_anomaly, rel=1e-4)
        assert row["cdd_anomaly"] == pytest.approx(expected_cdd_anomaly, rel=1e-4)

    def test_warmer_year_anomalies(self):
        """Warmer year: negative HDD anomaly, positive CDD anomaly."""
        # Normal: avg = 15°C (cool, produces HDD)
        # Observed: avg = 25°C (warm, produces CDD but not HDD)
        obs = _full_year_obs(tmax=30.0, tmin=20.0)  # avg = 25
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals(normal_tmax=20.0, normal_tmin=10.0)  # avg = 15

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        # HDD anomaly: observed HDD (0) - normal HDD (positive) → negative
        assert row["hdd_anomaly"] < 0
        # CDD anomaly: observed CDD (positive) - normal CDD (0) → positive
        assert row["cdd_anomaly"] > 0

    def test_cooler_year_anomalies(self):
        """Cooler year: positive HDD anomaly, negative CDD anomaly."""
        # Normal: avg = 25°C (warm, produces CDD)
        # Observed: avg = 10°C (cool, produces HDD)
        obs = _full_year_obs(tmax=15.0, tmin=5.0)  # avg = 10
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals(normal_tmax=30.0, normal_tmin=20.0)  # avg = 25

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        # HDD anomaly: observed HDD (positive) - normal HDD (0) → positive
        assert row["hdd_anomaly"] > 0
        # CDD anomaly: observed CDD (0) - normal CDD (positive) → negative
        assert row["cdd_anomaly"] < 0


class TestMultiYearOutput:
    """Verify separate rows for each year per county."""

    def test_two_years(self):
        obs1 = _full_year_obs(year=2019, tmax=25.0, tmin=15.0)
        obs2 = _full_year_obs(year=2020, tmax=30.0, tmin=20.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 2
        assert set(result["year"]) == {2019, 2020}
        assert result["fips"].unique().tolist() == ["01001"]


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame schema."""

    def _get_result(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()
        return compute_degree_days(obs, mapping, normals)

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
        assert pd.api.types.is_integer_dtype(result["year"]) or result["year"].dtype == np.int64
        assert pd.api.types.is_float_dtype(result["hdd_annual"])
        assert pd.api.types.is_float_dtype(result["cdd_annual"])
        assert pd.api.types.is_float_dtype(result["hdd_anomaly"])
        assert pd.api.types.is_float_dtype(result["cdd_anomaly"])


# ---------------------------------------------------------------------------
# Edge case tests (module-specific)
# ---------------------------------------------------------------------------

class TestMissingTemps:
    """Rows with NaN tmax or tmin are dropped before computation."""

    def test_nan_tmax_dropped(self):
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        # Set first 30 rows to NaN tmax
        obs.loc[:29, "tmax"] = np.nan
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        # Should have 335 days contributing (365 - 30 = 335 ≥ threshold)
        expected_cdd = (20.0 - BASE_TEMP_C) * 335
        assert row["cdd_annual"] == pytest.approx(expected_cdd, rel=1e-6)

    def test_nan_tmin_dropped(self):
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        obs.loc[:29, "tmin"] = np.nan
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]
        expected_cdd = (20.0 - BASE_TEMP_C) * 335
        assert row["cdd_annual"] == pytest.approx(expected_cdd, rel=1e-6)


class TestQualityFlags:
    """Quality-flagged observations are excluded."""

    def test_flagged_tmax_excluded(self):
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        # Flag first 10 rows
        obs.loc[:9, "q_flag_tmax"] = "G"
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]
        # 355 valid days
        expected_cdd = (20.0 - BASE_TEMP_C) * 355
        assert row["cdd_annual"] == pytest.approx(expected_cdd, rel=1e-6)

    def test_flagged_tmin_excluded(self):
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        obs.loc[:9, "q_flag_tmin"] = "X"
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]
        expected_cdd = (20.0 - BASE_TEMP_C) * 355
        assert row["cdd_annual"] == pytest.approx(expected_cdd, rel=1e-6)


class TestCompletenessThreshold:
    """Station-years with <335 valid days are excluded."""

    def test_below_threshold_excluded(self):
        """Station with 300 days should be excluded."""
        obs = _make_daily_obs(n_days=300, tmax=25.0, tmin=15.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 0

    def test_at_threshold_included(self):
        """Station with exactly 335 days should be included."""
        obs = _make_daily_obs(n_days=335, tmax=25.0, tmin=15.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 1

    def test_above_threshold_included(self):
        """Station with 365 days should be included."""
        obs = _make_daily_obs(n_days=365, tmax=25.0, tmin=15.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 1


class TestStationWithNoNormals:
    """Station with data but no normals: HDD/CDD computed, anomalies NaN."""

    def test_no_normals_nan_anomaly(self):
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        # Empty normals
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        assert row["hdd_annual"] == pytest.approx(0.0, abs=1e-6)
        assert row["cdd_annual"] > 0
        assert pd.isna(row["hdd_anomaly"])
        assert pd.isna(row["cdd_anomaly"])


class TestIncompleteNormals:
    """Station with <12 months of normals excluded from anomaly but keeps HDD/CDD."""

    def test_incomplete_normals_nan_anomaly(self):
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        # Only 6 months of normals
        normals = _make_normals(months=[1, 2, 3, 4, 5, 6])

        result = compute_degree_days(obs, mapping, normals)
        row = result.iloc[0]

        # HDD/CDD should be computed
        assert row["cdd_annual"] > 0
        # Anomalies should be NaN (incomplete normals)
        assert pd.isna(row["hdd_anomaly"])
        assert pd.isna(row["cdd_anomaly"])


class TestCountyWithNoStations:
    """Counties with no station mapping are absent from output."""

    def test_unmapped_county_absent(self):
        obs = _full_year_obs(tmax=25.0, tmin=15.0)
        # Map to 01001, but no mapping for 02001
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert "02001" not in result["fips"].values


class TestUnmappedStation:
    """Station in observations but not in station-to-county mapping."""

    def test_unmapped_station_excluded(self):
        obs = _full_year_obs(station_id="USW00099", tmax=25.0, tmin=15.0)
        # Mapping has a different station
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 0


class TestOneStationFailsOnePassesSameCounty:
    """One station fails completeness, another passes — county still in output."""

    def test_partial_county(self):
        # Station 1: 340 days (passes)
        obs1 = _make_daily_obs("USW00001", n_days=340, tmax=25.0, tmin=15.0)
        # Station 2: 100 days (fails)
        obs2 = _make_daily_obs("USW00002", n_days=100, tmax=30.0, tmin=20.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([
            {"station_id": "USW00001", "fips": "01001"},
            {"station_id": "USW00002", "fips": "01001"},
        ])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 1
        assert result.iloc[0]["fips"] == "01001"

        # Should use only station 1's data (avg=20, CDD per day = 20-18.333)
        expected_cdd = (20.0 - BASE_TEMP_C) * 340
        assert result.iloc[0]["cdd_annual"] == pytest.approx(expected_cdd, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge case tests (generic)
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Verify graceful handling of empty DataFrames."""

    def test_empty_daily_obs(self):
        obs = pd.DataFrame(columns=["station_id", "date", "tmax", "tmin",
                                     "q_flag_tmax", "q_flag_tmin"])
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_empty_station_county_map(self):
        obs = _full_year_obs()
        mapping = pd.DataFrame(columns=["station_id", "fips"])
        normals = _make_normals()

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_empty_normals(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["hdd_anomaly"])
        assert pd.isna(result.iloc[0]["cdd_anomaly"])


class TestMissingColumns:
    """Missing required columns raise ValueError."""

    def test_obs_missing_tmax(self):
        obs = pd.DataFrame({
            "station_id": ["USW00001"],
            "date": [date(2020, 1, 1)],
            "tmin": [15.0],
            "q_flag_tmax": [""],
            "q_flag_tmin": [""],
        })
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()

        with pytest.raises(ValueError, match="tmax"):
            compute_degree_days(obs, mapping, normals)

    def test_normals_missing_normal_tmin(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame({
            "station_id": ["USW00001"],
            "month": [1],
            "normal_tmax": [20.0],
        })

        with pytest.raises(ValueError, match="normal_tmin"):
            compute_degree_days(obs, mapping, normals)

    def test_mapping_missing_fips(self):
        obs = _full_year_obs()
        mapping = pd.DataFrame({"station_id": ["USW00001"]})
        normals = _make_normals()

        with pytest.raises(ValueError, match="fips"):
            compute_degree_days(obs, mapping, normals)


class TestPartialDataProducesPartialOutput:
    """Valid data for some years/counties produces output for those; others absent."""

    def test_one_year_valid_one_invalid(self):
        # Year 2019: 365 days (valid)
        obs1 = _full_year_obs(year=2019, tmax=25.0, tmin=15.0)
        # Year 2020: 100 days (invalid — below threshold)
        obs2 = _make_daily_obs(year=2020, n_days=100, tmax=25.0, tmin=15.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
        assert len(result) == 1
        assert result.iloc[0]["year"] == 2019


class TestFIPSNormalization:
    """Output FIPS codes are 5-digit zero-padded strings."""

    def test_fips_format(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = pd.DataFrame(columns=["station_id", "month", "normal_tmax", "normal_tmin"])

        result = compute_degree_days(obs, mapping, normals)
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
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()

        result1 = compute_degree_days(obs, mapping, normals)
        result2 = compute_degree_days(obs, mapping, normals)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Transform purity tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """Verify NO scoring metrics in output."""

    def test_no_scoring_columns(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()

        result = compute_degree_days(obs, mapping, normals)
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
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()

        result = compute_degree_days(obs, mapping, normals)
        dupes = result.duplicated(subset=["fips", "year"], keep=False)
        assert not dupes.any(), "Duplicate (fips, year) rows found"


# ---------------------------------------------------------------------------
# Metadata sidecar test
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written with correct values."""

    def test_metadata_content(self, tmp_path):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        with (
            patch("transform.degree_days.HARMONIZED_DIR", harmonized),
            patch("transform.degree_days.OBS_COMBINED_PATH", tmp_path / "obs_combined.parquet"),
            patch("transform.degree_days.STATION_COUNTY_PATH", tmp_path / "stc.parquet"),
            patch("transform.degree_days.NORMALS_PATH", tmp_path / "normals.parquet"),
        ):
            # Write source data so run() can load it
            obs.to_parquet(tmp_path / "obs_combined.parquet", index=False)
            mapping.to_parquet(tmp_path / "stc.parquet", index=False)
            normals.to_parquet(tmp_path / "normals.parquet", index=False)

            run()

        meta_path = harmonized / "degree_days_2020_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "NOAA_GHCN_DAILY"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta

    def test_parquet_written(self, tmp_path):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        normals = _make_normals()

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        with (
            patch("transform.degree_days.HARMONIZED_DIR", harmonized),
            patch("transform.degree_days.OBS_COMBINED_PATH", tmp_path / "obs_combined.parquet"),
            patch("transform.degree_days.STATION_COUNTY_PATH", tmp_path / "stc.parquet"),
            patch("transform.degree_days.NORMALS_PATH", tmp_path / "normals.parquet"),
        ):
            obs.to_parquet(tmp_path / "obs_combined.parquet", index=False)
            mapping.to_parquet(tmp_path / "stc.parquet", index=False)
            normals.to_parquet(tmp_path / "normals.parquet", index=False)

            run()

        pq_path = harmonized / "degree_days_2020.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)
        assert set(df.columns) == set(OUTPUT_COLUMNS)
