"""Tests for transform/health_burden.py — Health burden index from ED visits.

All tests use synthetic DataFrames with known expected outputs.
No real data files are read.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from transform.health_burden import (
    OUTPUT_COLUMNS,
    compute_health_burden,
    run,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _make_ed_visits(
    records: list[dict],
) -> pd.DataFrame:
    """Build synthetic ED visit DataFrame.

    Each dict should have keys like:
        fips, state_fips, year, heat_ed_visits, geo_resolution
    Optional: population
    """
    df = pd.DataFrame(records)
    # Ensure required columns exist
    for col in ["year", "heat_ed_visits", "geo_resolution"]:
        if col not in df.columns:
            df[col] = None
    return df


def _make_population(
    records: list[dict],
) -> pd.DataFrame:
    """Build synthetic population DataFrame.

    Each dict: {"fips": str, "year": int, "population": int}
    """
    return pd.DataFrame(records)


def _county_ed(fips: str, year: int, visits: float, population: float | None = None) -> dict:
    """Shorthand for a county-level ED record."""
    rec = {
        "fips": fips,
        "state_fips": fips[:2],
        "year": year,
        "heat_ed_visits": visits,
        "geo_resolution": "county",
    }
    if population is not None:
        rec["population"] = population
    return rec


def _state_ed(state_fips: str, year: int, visits: float) -> dict:
    """Shorthand for a state-level ED record."""
    return {
        "fips": state_fips,
        "state_fips": state_fips,
        "year": year,
        "heat_ed_visits": visits,
        "geo_resolution": "state",
    }


def _pop(fips: str, year: int, population: int) -> dict:
    """Shorthand for a population record."""
    return {"fips": fips, "year": year, "population": population}


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestPerCapitaRateCountyLevel:
    """Per-capita rate — county-level data."""

    def test_basic_rate(self):
        """50 visits / 100,000 population → rate = 50.0."""
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        result = compute_health_burden(ed, pop)
        row = result[(result["fips"] == "01001") & (result["year"] == 2020)]
        assert len(row) == 1
        assert row.iloc[0]["heat_ed_rate_per_100k"] == pytest.approx(50.0)

    def test_large_population(self):
        """200 visits / 1,000,000 population → rate = 20.0."""
        ed = _make_ed_visits([_county_ed("06037", 2020, 200)])
        pop = _make_population([_pop("06037", 2020, 1_000_000)])

        result = compute_health_burden(ed, pop)
        row = result[result["fips"] == "06037"]
        assert row.iloc[0]["heat_ed_rate_per_100k"] == pytest.approx(20.0)


class TestStateLevelDisaggregation:
    """State-level ED data disaggregated to counties."""

    def test_uniform_rate_to_counties(self):
        """State-level data → all counties in state get same rate."""
        # State 01 has 100 visits, state population = 200,000
        ed = _make_ed_visits([_state_ed("01", 2020, 100)])
        pop = _make_population([
            _pop("01001", 2020, 50_000),
            _pop("01003", 2020, 80_000),
            _pop("01005", 2020, 70_000),
        ])

        result = compute_health_burden(ed, pop)
        assert len(result) == 3

        # State rate = 100 / 200,000 × 100,000 = 50.0
        expected_rate = (100 / 200_000) * 100_000
        for _, row in result.iterrows():
            assert row["heat_ed_rate_per_100k"] == pytest.approx(expected_rate)


class TestCountyLevelPrecedence:
    """County-level data takes precedence over state-level."""

    def test_county_over_state(self):
        """When both exist for same county-year, county-level wins."""
        ed = _make_ed_visits([
            _county_ed("01001", 2020, 75),
            _state_ed("01", 2020, 100),
        ])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("01003", 2020, 100_000),
        ])

        result = compute_health_burden(ed, pop)
        county_01001 = result[result["fips"] == "01001"]
        # County-level rate: 75 / 100,000 × 100,000 = 75.0
        assert county_01001.iloc[0]["heat_ed_rate_per_100k"] == pytest.approx(75.0)

        # 01003 should get the state rate: 100 / 200,000 × 100,000 = 50.0
        county_01003 = result[result["fips"] == "01003"]
        assert len(county_01003) == 1
        assert county_01003.iloc[0]["heat_ed_rate_per_100k"] == pytest.approx(50.0)


class TestHealthBurdenIndexEqualsRate:
    """health_burden_index == heat_ed_rate_per_100k in v1."""

    def test_index_equals_rate(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 42)])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        result = compute_health_burden(ed, pop)
        for _, row in result.iterrows():
            assert row["health_burden_index"] == row["heat_ed_rate_per_100k"]


class TestMultiYearOutput:
    """Separate rows per year."""

    def test_two_years(self):
        ed = _make_ed_visits([
            _county_ed("01001", 2019, 30),
            _county_ed("01001", 2020, 50),
        ])
        pop = _make_population([
            _pop("01001", 2019, 100_000),
            _pop("01001", 2020, 100_000),
        ])

        result = compute_health_burden(ed, pop)
        assert len(result) == 2
        assert set(result["year"]) == {2019, 2020}


class TestMultiStateOutput:
    """Counties in different states get their respective state rates."""

    def test_two_states(self):
        ed = _make_ed_visits([
            _state_ed("01", 2020, 100),   # state pop = 100k → rate = 100
            _state_ed("06", 2020, 200),   # state pop = 500k → rate = 40
        ])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("06037", 2020, 300_000),
            _pop("06073", 2020, 200_000),
        ])

        result = compute_health_burden(ed, pop)

        state01 = result[result["fips"] == "01001"]
        assert state01.iloc[0]["heat_ed_rate_per_100k"] == pytest.approx(100.0)

        state06 = result[result["fips"].str.startswith("06")]
        expected_06 = (200 / 500_000) * 100_000  # = 40.0
        for _, row in state06.iterrows():
            assert row["heat_ed_rate_per_100k"] == pytest.approx(expected_06)


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame schema."""

    def _get_result(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([_pop("01001", 2020, 100_000)])
        return compute_health_burden(ed, pop)

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
        assert pd.api.types.is_float_dtype(result["heat_ed_rate_per_100k"])
        assert pd.api.types.is_float_dtype(result["health_burden_index"])


# ---------------------------------------------------------------------------
# Edge case tests (module-specific)
# ---------------------------------------------------------------------------

class TestZeroPopulation:
    """County with population = 0 → rate is NaN, not division-by-zero."""

    def test_zero_pop_nan_rate(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([_pop("01001", 2020, 0)])

        result = compute_health_burden(ed, pop)
        # County with zero population should be excluded (NaN rate dropped)
        assert len(result) == 0


class TestNanEdVisits:
    """Record with NaN heat_ed_visits is excluded."""

    def test_nan_visits_excluded(self):
        ed = _make_ed_visits([
            _county_ed("01001", 2020, float("nan")),
            _county_ed("01003", 2020, 50),
        ])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("01003", 2020, 100_000),
        ])

        result = compute_health_burden(ed, pop)
        assert "01001" not in result["fips"].values
        assert "01003" in result["fips"].values


class TestStateWithNoCountiesInCensus:
    """State-level EPHT data but no matching counties in Census → graceful."""

    def test_no_matching_counties(self):
        ed = _make_ed_visits([_state_ed("99", 2020, 100)])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        result = compute_health_burden(ed, pop)
        # State 99 has no counties in pop → no disaggregation output
        assert "99" not in result["fips"].str[:2].values


class TestCountyLevelOnly:
    """All data is county-level — works correctly."""

    def test_county_only(self):
        ed = _make_ed_visits([
            _county_ed("01001", 2020, 50),
            _county_ed("01003", 2020, 30),
        ])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("01003", 2020, 100_000),
        ])

        result = compute_health_burden(ed, pop)
        assert len(result) == 2
        assert set(result["fips"]) == {"01001", "01003"}


class TestStateLevelOnly:
    """All data is state-level — works correctly."""

    def test_state_only(self):
        ed = _make_ed_visits([_state_ed("01", 2020, 200)])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("01003", 2020, 100_000),
        ])

        result = compute_health_burden(ed, pop)
        assert len(result) == 2
        expected_rate = (200 / 200_000) * 100_000  # 100.0
        for _, row in result.iterrows():
            assert row["heat_ed_rate_per_100k"] == pytest.approx(expected_rate)


class TestNonReportingState:
    """Counties in states without EPHT data are absent from output."""

    def test_non_reporting(self):
        # Only state 01 reports; state 06 does not
        ed = _make_ed_visits([_state_ed("01", 2020, 100)])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("06037", 2020, 500_000),
        ])

        result = compute_health_burden(ed, pop)
        assert "01001" in result["fips"].values
        assert "06037" not in result["fips"].values


# ---------------------------------------------------------------------------
# Edge case tests (generic)
# ---------------------------------------------------------------------------

class TestEmptyInputData:
    """Graceful handling of empty DataFrames."""

    def test_empty_ed_visits(self):
        ed = _make_ed_visits([])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        result = compute_health_burden(ed, pop)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_empty_population(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([])

        result = compute_health_burden(ed, pop)
        assert len(result) == 0
        assert set(result.columns) == set(OUTPUT_COLUMNS)


class TestMissingInputFile:
    """FileNotFoundError for both EPHT and Census ACS."""

    def test_missing_epht(self, tmp_path):
        fake_epht = tmp_path / "cdc_epht" / "cdc_epht_all.parquet"
        fake_epht_dir = tmp_path / "cdc_epht"
        with (
            patch("transform.health_burden.EPHT_COMBINED_PATH", fake_epht),
            patch("transform.health_burden.EPHT_DIR", fake_epht_dir),
        ):
            with pytest.raises(FileNotFoundError):
                from transform.health_burden import _load_ed_visits
                _load_ed_visits()

    def test_missing_acs(self, tmp_path):
        fake_acs = tmp_path / "census_acs" / "census_acs_all.parquet"
        fake_acs_dir = tmp_path / "census_acs"
        with (
            patch("transform.health_burden.ACS_COMBINED_PATH", fake_acs),
            patch("transform.health_burden.ACS_DIR", fake_acs_dir),
        ):
            with pytest.raises(FileNotFoundError):
                from transform.health_burden import _load_population
                _load_population()


class TestMissingRequiredColumns:
    """ValueError with missing column names."""

    def test_ed_missing_heat_ed_visits(self):
        ed = pd.DataFrame({
            "fips": ["01001"],
            "year": [2020],
            "geo_resolution": ["county"],
        })
        pop = _make_population([_pop("01001", 2020, 100_000)])

        with pytest.raises(ValueError, match="heat_ed_visits"):
            compute_health_burden(ed, pop)

    def test_pop_missing_population(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = pd.DataFrame({
            "fips": ["01001"],
            "year": [2020],
        })

        with pytest.raises(ValueError, match="population"):
            compute_health_burden(ed, pop)


class TestPartialDataProducesPartialOutput:
    """Partial data → partial output."""

    def test_some_counties_have_data(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("01003", 2020, 100_000),
        ])

        result = compute_health_burden(ed, pop)
        assert "01001" in result["fips"].values
        assert "01003" not in result["fips"].values


class TestFIPSNormalization:
    """5-digit zero-padded FIPS strings."""

    def test_fips_format(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        result = compute_health_burden(ed, pop)
        for fips in result["fips"]:
            assert isinstance(fips, str)
            assert len(fips) == 5
            assert fips.isdigit()


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Same input → identical output."""

    def test_deterministic(self):
        ed = _make_ed_visits([
            _county_ed("01001", 2020, 50),
            _state_ed("06", 2020, 200),
        ])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("06037", 2020, 300_000),
            _pop("06073", 2020, 200_000),
        ])

        result1 = compute_health_burden(ed, pop)
        result2 = compute_health_burden(ed, pop)

        # Sort both for stable comparison
        result1 = result1.sort_values(["fips", "year"]).reset_index(drop=True)
        result2 = result2.sort_values(["fips", "year"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Transform purity tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """Verify NO scoring metrics in output."""

    def test_no_scoring_columns(self):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        result = compute_health_burden(ed, pop)
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
    """Exactly one row per (fips, year) — no duplicates."""

    def test_no_duplicate_fips_year(self):
        ed = _make_ed_visits([
            _county_ed("01001", 2020, 50),
            _state_ed("01", 2020, 200),
        ])
        pop = _make_population([
            _pop("01001", 2020, 100_000),
            _pop("01003", 2020, 100_000),
        ])

        result = compute_health_burden(ed, pop)
        dupes = result.duplicated(subset=["fips", "year"], keep=False)
        assert not dupes.any(), "Duplicate (fips, year) rows found"


# ---------------------------------------------------------------------------
# Metadata sidecar test
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written with correct values."""

    def test_metadata_content(self, tmp_path):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        epht_path = tmp_path / "cdc_epht" / "cdc_epht_all.parquet"
        epht_path.parent.mkdir(parents=True)
        ed.to_parquet(epht_path, index=False)

        acs_path = tmp_path / "census_acs" / "census_acs_all.parquet"
        acs_path.parent.mkdir(parents=True)
        pop.to_parquet(acs_path, index=False)

        with (
            patch("transform.health_burden.HARMONIZED_DIR", harmonized),
            patch("transform.health_burden.EPHT_COMBINED_PATH", epht_path),
            patch("transform.health_burden.ACS_COMBINED_PATH", acs_path),
        ):
            run()

        meta_path = harmonized / "health_burden_2020_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == "CDC_EPHT"
        assert meta["confidence"] == "B"
        assert meta["attribution"] == "proxy"
        assert "retrieved_at" in meta

    def test_parquet_written(self, tmp_path):
        ed = _make_ed_visits([_county_ed("01001", 2020, 50)])
        pop = _make_population([_pop("01001", 2020, 100_000)])

        harmonized = tmp_path / "harmonized"
        harmonized.mkdir()

        epht_path = tmp_path / "cdc_epht" / "cdc_epht_all.parquet"
        epht_path.parent.mkdir(parents=True)
        ed.to_parquet(epht_path, index=False)

        acs_path = tmp_path / "census_acs" / "census_acs_all.parquet"
        acs_path.parent.mkdir(parents=True)
        pop.to_parquet(acs_path, index=False)

        with (
            patch("transform.health_burden.HARMONIZED_DIR", harmonized),
            patch("transform.health_burden.EPHT_COMBINED_PATH", epht_path),
            patch("transform.health_burden.ACS_COMBINED_PATH", acs_path),
        ):
            run()

        pq_path = harmonized / "health_burden_2020.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)
        assert set(df.columns) == set(OUTPUT_COLUMNS)
