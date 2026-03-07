"""Tests for transform/extreme_heat.py — Extreme heat day counts.

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

from transform.extreme_heat import (
    MIN_DAYS_PER_YEAR,
    OUTPUT_COLUMNS,
    THRESHOLD_95F_C,
    THRESHOLD_100F_C,
    compute_extreme_heat_days,
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
    start_month: int = 1,
    start_day: int = 1,
    q_flag_tmax: str = "",
) -> pd.DataFrame:
    """Build synthetic daily observations for a single station."""
    dates = pd.date_range(
        start=date(year, start_month, start_day),
        periods=n_days,
        freq="D",
    )
    return pd.DataFrame({
        "station_id": station_id,
        "date": dates,
        "tmax": tmax,
        "q_flag_tmax": q_flag_tmax,
    })


def _make_station_county_map(mappings: list[dict]) -> pd.DataFrame:
    """Build a synthetic station-to-county mapping."""
    return pd.DataFrame(mappings)


def _full_year_obs(
    station_id: str = "USW00001",
    year: int = 2020,
    tmax: float = 25.0,
) -> pd.DataFrame:
    """Full 365-day observation set for a station-year."""
    return _make_daily_obs(station_id=station_id, year=year, n_days=365, tmax=tmax)


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestDailyThresholdExceedance:
    """Verify daily threshold exceedance detection."""

    def test_exceeds_95f(self):
        """tmax = 36.0°C → exceeds 95°F (35.0°C)."""
        obs = _full_year_obs(tmax=36.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == 365.0
        assert row["days_above_100f"] == 0.0

    def test_does_not_exceed_95f(self):
        """tmax = 34.0°C → does NOT exceed 95°F (35.0°C)."""
        obs = _full_year_obs(tmax=34.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == 0.0
        assert row["days_above_100f"] == 0.0

    def test_exceeds_100f(self):
        """tmax = 38.0°C → exceeds 100°F (37.778°C)."""
        obs = _full_year_obs(tmax=38.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_100f"] == 365.0

    def test_does_not_exceed_100f(self):
        """tmax = 37.0°C → does NOT exceed 100°F (37.778°C)."""
        obs = _full_year_obs(tmax=37.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_100f"] == 0.0

    def test_exceeds_both_thresholds(self):
        """tmax > 37.778°C counts toward BOTH 95°F and 100°F."""
        obs = _full_year_obs(tmax=40.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == 365.0
        assert row["days_above_100f"] == 365.0

    def test_exceeds_neither_threshold(self):
        """tmax < 35.0°C → counts toward neither metric."""
        obs = _full_year_obs(tmax=30.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == 0.0
        assert row["days_above_100f"] == 0.0

    def test_between_thresholds(self):
        """35.0°C < tmax < 37.778°C → counts toward 95°F but NOT 100°F."""
        obs = _full_year_obs(tmax=36.5)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == 365.0
        assert row["days_above_100f"] == 0.0


class TestAnnualAggregation:
    """Verify annual aggregation (sum of daily exceedances)."""

    def test_known_hot_day_count(self):
        """Create a year with a known number of hot days."""
        # 365 days: first 30 at 40°C (both thresholds), rest at 20°C (neither)
        obs_hot = _make_daily_obs(n_days=30, tmax=40.0)
        obs_cool = _make_daily_obs(n_days=335, tmax=20.0, start_month=1, start_day=31)
        obs = pd.concat([obs_hot, obs_cool], ignore_index=True)

        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == 30.0
        assert row["days_above_100f"] == 30.0


class TestCountyAveraging:
    """Verify county-level averaging across stations."""

    def test_two_stations_averaged(self):
        """County result should be arithmetic mean of station values."""
        # Station 1: all 365 days at 40°C → 365 days above both
        obs1 = _full_year_obs("USW00001", tmax=40.0)
        # Station 2: all 365 days at 20°C → 0 days above either
        obs2 = _full_year_obs("USW00002", tmax=20.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([
            {"station_id": "USW00001", "fips": "01001"},
            {"station_id": "USW00002", "fips": "01001"},
        ])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == pytest.approx(365.0 / 2)
        assert row["days_above_100f"] == pytest.approx(365.0 / 2)


class TestMultiYearOutput:
    """Verify separate rows for each year per county."""

    def test_two_years(self):
        obs1 = _full_year_obs(year=2019, tmax=36.0)
        obs2 = _full_year_obs(year=2020, tmax=40.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 2
        assert set(result["year"]) == {2019, 2020}
        assert result["fips"].unique().tolist() == ["01001"]

        # 2019: tmax=36 → exceeds 95F, not 100F
        row_2019 = result[result["year"] == 2019].iloc[0]
        assert row_2019["days_above_95f"] == 365.0
        assert row_2019["days_above_100f"] == 0.0

        # 2020: tmax=40 → exceeds both
        row_2020 = result[result["year"] == 2020].iloc[0]
        assert row_2020["days_above_95f"] == 365.0
        assert row_2020["days_above_100f"] == 365.0


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame schema."""

    def _get_result(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])
        return compute_extreme_heat_days(obs, mapping)

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
        assert pd.api.types.is_float_dtype(result["days_above_95f"])
        assert pd.api.types.is_float_dtype(result["days_above_100f"])


# ---------------------------------------------------------------------------
# Edge case tests (module-specific)
# ---------------------------------------------------------------------------

class TestMissingTmax:
    """Rows with NaN tmax are dropped before computation."""

    def test_nan_tmax_dropped(self):
        obs = _full_year_obs(tmax=36.0)
        # Set first 30 rows to NaN tmax
        obs.loc[:29, "tmax"] = np.nan
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        # 335 valid days at 36°C → all exceed 95F, none exceed 100F
        assert row["days_above_95f"] == 335.0
        assert row["days_above_100f"] == 0.0


class TestQualityFlags:
    """Quality-flagged observations are excluded."""

    def test_flagged_tmax_excluded(self):
        obs = _full_year_obs(tmax=36.0)
        # Flag first 10 rows
        obs.loc[:9, "q_flag_tmax"] = "G"
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        # 355 valid days at 36°C → all exceed 95F
        assert row["days_above_95f"] == 355.0


class TestCompletenessThreshold:
    """Station-years with <335 valid days are excluded."""

    def test_below_threshold_excluded(self):
        """Station with 300 days should be excluded."""
        obs = _make_daily_obs(n_days=300, tmax=36.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 0

    def test_at_threshold_included(self):
        """Station with exactly 335 days should be included."""
        obs = _make_daily_obs(n_days=335, tmax=36.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 1

    def test_above_threshold_included(self):
        """Station with 365 days should be included."""
        obs = _make_daily_obs(n_days=365, tmax=36.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 1


class TestCountyWithNoStations:
    """Counties with no station mapping are absent from output."""

    def test_unmapped_county_absent(self):
        obs = _full_year_obs(tmax=36.0)
        # Map to 01001, but no mapping for 02001
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert "02001" not in result["fips"].values


class TestUnmappedStation:
    """Station in observations but not in station-to-county mapping."""

    def test_unmapped_station_excluded(self):
        obs = _full_year_obs(station_id="USW00099", tmax=36.0)
        # Mapping has a different station
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 0


class TestOneStationFailsOnePassesSameCounty:
    """One station fails completeness, another passes — county still in output."""

    def test_partial_county(self):
        # Station 1: 340 days at 40°C (passes)
        obs1 = _make_daily_obs("USW00001", n_days=340, tmax=40.0)
        # Station 2: 100 days at 40°C (fails)
        obs2 = _make_daily_obs("USW00002", n_days=100, tmax=40.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([
            {"station_id": "USW00001", "fips": "01001"},
            {"station_id": "USW00002", "fips": "01001"},
        ])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 1
        assert result.iloc[0]["fips"] == "01001"

        # Should use only station 1's data (340 days at 40°C → all exceed both)
        assert result.iloc[0]["days_above_95f"] == 340.0
        assert result.iloc[0]["days_above_100f"] == 340.0


class TestZeroHotDays:
    """A full year where tmax never exceeds either threshold."""

    def test_zero_hot_days(self):
        obs = _full_year_obs(tmax=20.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        row = result.iloc[0]
        assert row["days_above_95f"] == 0.0
        assert row["days_above_100f"] == 0.0


# ---------------------------------------------------------------------------
# Edge case tests (generic)
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Verify graceful handling of empty DataFrames."""

    def test_empty_daily_obs(self):
        obs = pd.DataFrame(columns=["station_id", "date", "tmax", "q_flag_tmax"])
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_empty_station_county_map(self):
        obs = _full_year_obs()
        mapping = pd.DataFrame(columns=["station_id", "fips"])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)


class TestMissingColumns:
    """Missing required columns raise ValueError."""

    def test_obs_missing_tmax(self):
        obs = pd.DataFrame({
            "station_id": ["USW00001"],
            "date": [date(2020, 1, 1)],
            "q_flag_tmax": [""],
        })
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        with pytest.raises(ValueError, match="tmax"):
            compute_extreme_heat_days(obs, mapping)

    def test_obs_missing_date(self):
        obs = pd.DataFrame({
            "station_id": ["USW00001"],
            "tmax": [36.0],
            "q_flag_tmax": [""],
        })
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        with pytest.raises(ValueError, match="date"):
            compute_extreme_heat_days(obs, mapping)

    def test_mapping_missing_fips(self):
        obs = _full_year_obs()
        mapping = pd.DataFrame({"station_id": ["USW00001"]})

        with pytest.raises(ValueError, match="fips"):
            compute_extreme_heat_days(obs, mapping)


class TestMissingInputFile:
    """Missing raw data files raise FileNotFoundError."""

    def test_missing_observations(self, tmp_path):
        with (
            patch("transform.extreme_heat.OBS_COMBINED_PATH", tmp_path / "no_such_file.parquet"),
            patch("transform.extreme_heat.OBS_DIR", tmp_path),
        ):
            with pytest.raises(FileNotFoundError):
                run()

    def test_missing_station_county(self, tmp_path):
        obs = _full_year_obs()
        obs_path = tmp_path / "obs_combined.parquet"
        obs.to_parquet(obs_path, index=False)

        with (
            patch("transform.extreme_heat.OBS_COMBINED_PATH", obs_path),
            patch("transform.extreme_heat.STATION_COUNTY_PATH", tmp_path / "no_such_file.parquet"),
        ):
            with pytest.raises(FileNotFoundError):
                run()


class TestPartialDataProducesPartialOutput:
    """Valid data for some years produces output for those; others absent."""

    def test_one_year_valid_one_invalid(self):
        # Year 2019: 365 days (valid)
        obs1 = _full_year_obs(year=2019, tmax=36.0)
        # Year 2020: 100 days (invalid — below threshold)
        obs2 = _make_daily_obs(year=2020, n_days=100, tmax=36.0)
        obs = pd.concat([obs1, obs2], ignore_index=True)

        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
        assert len(result) == 1
        assert result.iloc[0]["year"] == 2019


class TestFIPSNormalization:
    """Output FIPS codes are 5-digit zero-padded strings."""

    def test_fips_format(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
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
        obs = _full_year_obs(tmax=36.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result1 = compute_extreme_heat_days(obs, mapping)
        result2 = compute_extreme_heat_days(obs, mapping)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Transform purity tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """Verify NO scoring metrics in output."""

    def test_no_scoring_columns(self):
        obs = _full_year_obs()
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        result = compute_extreme_heat_days(obs, mapping)
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

        result = compute_extreme_heat_days(obs, mapping)
        dupes = result.duplicated(subset=["fips", "year"], keep=False)
        assert not dupes.any(), "Duplicate (fips, year) rows found"


# ---------------------------------------------------------------------------
# Metadata sidecar test
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written with correct values."""

    def test_metadata_content(self, tmp_path):
        obs = _full_year_obs(tmax=36.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        with (
            patch("transform.extreme_heat.HARMONIZED_DIR", harmonized),
            patch("transform.extreme_heat.OBS_COMBINED_PATH", tmp_path / "obs_combined.parquet"),
            patch("transform.extreme_heat.STATION_COUNTY_PATH", tmp_path / "stc.parquet"),
        ):
            obs.to_parquet(tmp_path / "obs_combined.parquet", index=False)
            mapping.to_parquet(tmp_path / "stc.parquet", index=False)

            run()

        meta_path = harmonized / "extreme_heat_2020_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "NOAA_GHCN_DAILY"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta

    def test_parquet_written(self, tmp_path):
        obs = _full_year_obs(tmax=36.0)
        mapping = _make_station_county_map([{"station_id": "USW00001", "fips": "01001"}])

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        with (
            patch("transform.extreme_heat.HARMONIZED_DIR", harmonized),
            patch("transform.extreme_heat.OBS_COMBINED_PATH", tmp_path / "obs_combined.parquet"),
            patch("transform.extreme_heat.STATION_COUNTY_PATH", tmp_path / "stc.parquet"),
        ):
            obs.to_parquet(tmp_path / "obs_combined.parquet", index=False)
            mapping.to_parquet(tmp_path / "stc.parquet", index=False)

            run()

        pq_path = harmonized / "extreme_heat_2020.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)
        assert set(df.columns) == set(OUTPUT_COLUMNS)
