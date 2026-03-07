"""Tests for transform/drought_scoring.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from transform.drought_scoring import (
    METADATA_ATTRIBUTION,
    METADATA_CONFIDENCE,
    METADATA_SOURCE,
    OUTPUT_COLUMNS,
    SEVERITY_WEIGHTS,
    _empty_output,
    _write_metadata,
    compute_drought_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_weekly(
    fips: str,
    dates: list[str],
    d0: float = 0.0,
    d1: float = 0.0,
    d2: float = 0.0,
    d3: float = 0.0,
    d4: float = 0.0,
    none: float | None = None,
) -> pd.DataFrame:
    """Create a weekly drought DataFrame for a single county with uniform values."""
    if none is None:
        none = 100.0 - d0 - d1 - d2 - d3 - d4
    rows = []
    for dt in dates:
        rows.append({
            "fips": fips,
            "date": dt,
            "d0_pct": d0,
            "d1_pct": d1,
            "d2_pct": d2,
            "d3_pct": d3,
            "d4_pct": d4,
            "none_pct": none,
        })
    return pd.DataFrame(rows)


def _week_dates(year: int, n: int = 52) -> list[str]:
    """Generate n weekly date strings within a given year."""
    start = pd.Timestamp(f"{year}-01-07")
    return [str((start + pd.Timedelta(weeks=i)).date()) for i in range(n)]


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------
class TestWeeklySeverity:
    """Test the weekly severity calculation."""

    def test_known_severity(self):
        """d0=20, d1=10, d2=5 → weekly_severity = (20*1+10*2+5*3)/100 = 0.55"""
        dates = _week_dates(2020, n=1)
        df = _make_weekly("01001", dates, d0=20.0, d1=10.0, d2=5.0)
        result = compute_drought_scores(df, scoring_year=2020)

        assert len(result) == 1
        assert np.isclose(result.iloc[0]["drought_score"], 0.55)

    def test_annual_integral(self):
        """Multiple weeks sum correctly."""
        dates = _week_dates(2020, n=4)
        # Each week: severity = (50*1)/100 = 0.5
        df = _make_weekly("01001", dates, d0=50.0)
        result = compute_drought_scores(df, scoring_year=2020)

        assert np.isclose(result.iloc[0]["drought_score"], 0.5 * 4)


class TestMaxSeverity:
    """Test max_severity determination."""

    def test_d4_reached(self):
        """When D4 is non-zero, max_severity = 4."""
        dates = _week_dates(2020, n=2)
        df = _make_weekly("01001", dates, d4=10.0)
        result = compute_drought_scores(df, scoring_year=2020)
        assert result.iloc[0]["max_severity"] == 4

    def test_only_d0(self):
        """When only D0 is non-zero, max_severity = 0."""
        dates = _week_dates(2020, n=2)
        df = _make_weekly("01001", dates, d0=30.0)
        result = compute_drought_scores(df, scoring_year=2020)
        assert result.iloc[0]["max_severity"] == 0

    def test_no_drought(self):
        """All weeks none_pct=100 → max_severity = -1."""
        dates = _week_dates(2020, n=3)
        df = _make_weekly("01001", dates)  # all zeros, none=100
        result = compute_drought_scores(df, scoring_year=2020)
        assert result.iloc[0]["max_severity"] == -1


class TestWeeksInDrought:
    """Test weeks_in_drought counting."""

    def test_count(self):
        """Mix of drought and non-drought weeks."""
        drought_dates = _week_dates(2020, n=3)
        no_drought_dates = [str((pd.Timestamp("2020-04-01") + pd.Timedelta(weeks=i)).date()) for i in range(2)]
        df_drought = _make_weekly("01001", drought_dates, d0=20.0)
        df_none = _make_weekly("01001", no_drought_dates)
        df = pd.concat([df_drought, df_none], ignore_index=True)

        result = compute_drought_scores(df, scoring_year=2020)
        assert result.iloc[0]["weeks_in_drought"] == 3


class TestPctAreaAvg:
    """Test average percentage of county area in drought."""

    def test_average(self):
        """Known drought areas across weeks."""
        dates = _week_dates(2020, n=4)
        # Week 1-2: 40% in drought (none=60), Week 3-4: 20% in drought (none=80)
        rows = []
        for dt in dates[:2]:
            rows.append({"fips": "01001", "date": dt, "d0_pct": 40.0,
                         "d1_pct": 0, "d2_pct": 0, "d3_pct": 0, "d4_pct": 0, "none_pct": 60.0})
        for dt in dates[2:]:
            rows.append({"fips": "01001", "date": dt, "d0_pct": 20.0,
                         "d1_pct": 0, "d2_pct": 0, "d3_pct": 0, "d4_pct": 0, "none_pct": 80.0})
        df = pd.DataFrame(rows)

        result = compute_drought_scores(df, scoring_year=2020)
        # avg drought area = (40+40+20+20)/4 = 30
        assert np.isclose(result.iloc[0]["pct_area_avg"], 30.0)


class TestMultiCounty:
    """Test multi-county and multi-year scenarios."""

    def test_two_counties_same_year(self):
        """Two counties with different intensities get independent scores."""
        dates = _week_dates(2020, n=2)
        df1 = _make_weekly("01001", dates, d0=50.0)
        df2 = _make_weekly("06001", dates, d2=30.0)
        df = pd.concat([df1, df2], ignore_index=True)

        result = compute_drought_scores(df, scoring_year=2020)
        assert len(result) == 2

        r1 = result[result["fips"] == "01001"].iloc[0]
        r2 = result[result["fips"] == "06001"].iloc[0]

        # County 1: 2 weeks * (50*1)/100 = 1.0
        assert np.isclose(r1["drought_score"], 1.0)
        # County 2: 2 weeks * (30*3)/100 = 1.8
        assert np.isclose(r2["drought_score"], 1.8)

    def test_multi_year(self):
        """Data spanning 2 years produces separate rows."""
        dates_2019 = _week_dates(2019, n=2)
        dates_2020 = _week_dates(2020, n=3)
        df = _make_weekly("01001", dates_2019 + dates_2020, d0=10.0)

        r2019 = compute_drought_scores(df, scoring_year=2019)
        r2020 = compute_drought_scores(df, scoring_year=2020)

        assert len(r2019) == 1
        assert r2019.iloc[0]["year"] == 2019
        assert len(r2020) == 1
        assert r2020.iloc[0]["year"] == 2020

        # 2019: 2 weeks * 0.1 = 0.2, 2020: 3 weeks * 0.1 = 0.3
        assert np.isclose(r2019.iloc[0]["drought_score"], 0.2)
        assert np.isclose(r2020.iloc[0]["drought_score"], 0.3)


class TestZeroDrought:
    """Test county with no drought at all."""

    def test_zero_drought_present_in_output(self):
        """County with all none_pct=100 should appear with drought_score=0."""
        dates = _week_dates(2020, n=5)
        df = _make_weekly("01001", dates)  # all zeros
        result = compute_drought_scores(df, scoring_year=2020)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["fips"] == "01001"
        assert row["drought_score"] == 0.0
        assert row["weeks_in_drought"] == 0
        assert row["pct_area_avg"] == 0.0
        assert row["max_severity"] == -1


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------
class TestOutputSchema:
    """Test output DataFrame schema."""

    def test_column_presence(self):
        """Output contains all expected columns."""
        dates = _week_dates(2020, n=2)
        df = _make_weekly("01001", dates, d0=10.0)
        result = compute_drought_scores(df, scoring_year=2020)
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_column_types(self):
        """Output column types are correct."""
        dates = _week_dates(2020, n=2)
        df = _make_weekly("01001", dates, d0=10.0)
        result = compute_drought_scores(df, scoring_year=2020)

        assert pd.api.types.is_string_dtype(result["fips"])
        assert result["year"].dtype in [np.int64, np.int32, int]
        assert result["drought_score"].dtype == np.float64
        assert result["max_severity"].dtype in [np.int64, np.int32, int]
        assert result["weeks_in_drought"].dtype in [np.int64, np.int32, int]
        assert result["pct_area_avg"].dtype == np.float64

    def test_no_extra_columns(self):
        """Output contains ONLY the specified columns."""
        dates = _week_dates(2020, n=2)
        df = _make_weekly("01001", dates, d1=15.0)
        result = compute_drought_scores(df, scoring_year=2020)
        assert set(result.columns) == set(OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Edge case tests (module-specific)
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases specific to drought scoring."""

    def test_pct_not_summing_to_100(self):
        """Percentages summing to 95 or 105 — should not crash."""
        dates = _week_dates(2020, n=1)
        # Sum = 95 (within tolerance)
        df = _make_weekly("01001", dates, d0=20.0, none=75.0)
        result = compute_drought_scores(df, scoring_year=2020)
        assert len(result) == 1

    def test_negative_pct_clamped(self):
        """Negative d0_pct should be clamped to 0."""
        dates = _week_dates(2020, n=1)
        df = _make_weekly("01001", dates, d0=-5.0, none=105.0)
        result = compute_drought_scores(df, scoring_year=2020)
        assert len(result) == 1
        # After clamping d0 to 0: severity = 0
        assert result.iloc[0]["drought_score"] == 0.0

    def test_incomplete_year_still_computed(self):
        """County with only 30 weeks still produces output."""
        dates = _week_dates(2020, n=30)
        df = _make_weekly("01001", dates, d0=10.0)
        result = compute_drought_scores(df, scoring_year=2020)
        assert len(result) == 1
        assert np.isclose(result.iloc[0]["drought_score"], 0.1 * 30)

    def test_county_absent_from_year(self):
        """County not in raw data for a year → absent from output."""
        dates = _week_dates(2020, n=5)
        df = _make_weekly("01001", dates, d0=10.0)
        result = compute_drought_scores(df, scoring_year=2019)
        assert len(result) == 0

    def test_all_d_levels_nonzero(self):
        """All D0–D4 non-zero simultaneously computes correctly."""
        dates = _week_dates(2020, n=1)
        df = _make_weekly("01001", dates, d0=10.0, d1=10.0, d2=10.0, d3=10.0, d4=10.0)
        result = compute_drought_scores(df, scoring_year=2020)
        # severity = (10*1 + 10*2 + 10*3 + 10*4 + 10*5)/100 = 1.5
        assert np.isclose(result.iloc[0]["drought_score"], 1.5)

    def test_single_week(self):
        """Single week of data computes correctly."""
        dates = _week_dates(2020, n=1)
        df = _make_weekly("01001", dates, d1=25.0)
        result = compute_drought_scores(df, scoring_year=2020)
        # severity = (25*2)/100 = 0.5
        assert np.isclose(result.iloc[0]["drought_score"], 0.5)


# ---------------------------------------------------------------------------
# Edge cases (generic)
# ---------------------------------------------------------------------------
class TestGenericEdgeCases:
    """Test generic edge cases: empty input, missing files, missing columns."""

    def test_empty_input(self):
        """Empty DataFrame returns empty output with correct schema."""
        df = pd.DataFrame(columns=["fips", "date", "d0_pct", "d1_pct", "d2_pct",
                                    "d3_pct", "d4_pct", "none_pct"])
        result = compute_drought_scores(df, scoring_year=2020)
        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_missing_columns_raises(self):
        """Missing required columns raise ValueError."""
        df = pd.DataFrame({"fips": ["01001"], "date": ["2020-01-07"]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_drought_scores(df, scoring_year=2020)

    def test_missing_file_raises(self):
        """Missing raw data file raises FileNotFoundError."""
        from transform.drought_scoring import _load_weekly_drought
        with patch("transform.drought_scoring.DROUGHT_COMBINED_PATH",
                   Path("/nonexistent/path.parquet")):
            with patch("transform.drought_scoring.DROUGHT_DIR",
                       Path("/nonexistent")):
                with pytest.raises(FileNotFoundError):
                    _load_weekly_drought()

    def test_partial_data_produces_partial_output(self):
        """Counties with data produce output; missing counties don't."""
        dates = _week_dates(2020, n=3)
        df = _make_weekly("01001", dates, d0=10.0)
        # Only county 01001 has data, 06001 does not
        result = compute_drought_scores(df, scoring_year=2020)
        assert len(result) == 1
        assert result.iloc[0]["fips"] == "01001"

    def test_fips_normalization(self):
        """FIPS codes are 5-digit zero-padded strings."""
        dates = _week_dates(2020, n=1)
        df = _make_weekly("1001", dates, d0=10.0)  # Missing leading zero
        result = compute_drought_scores(df, scoring_year=2020)
        assert result.iloc[0]["fips"] == "01001"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    """Test that computation is deterministic."""

    def test_reproducibility(self):
        """Same input produces identical output."""
        dates = _week_dates(2020, n=10)
        df = _make_weekly("01001", dates, d0=20.0, d2=5.0)

        r1 = compute_drought_scores(df.copy(), scoring_year=2020)
        r2 = compute_drought_scores(df.copy(), scoring_year=2020)

        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Transform purity and metadata
# ---------------------------------------------------------------------------
class TestTransformPurity:
    """Verify no scoring metrics appear in output."""

    def test_no_percentiles_or_ranks(self):
        """Output should not contain percentiles, ranks, or composite scores."""
        dates = _week_dates(2020, n=5)
        df = _make_weekly("01001", dates, d1=15.0)
        result = compute_drought_scores(df, scoring_year=2020)

        forbidden = {"percentile", "rank", "composite", "acceleration", "overlap", "penalty"}
        for col in result.columns:
            assert not any(f in col.lower() for f in forbidden), f"Scoring metric found in column: {col}"

    def test_county_year_grain(self):
        """Output has exactly one row per (fips, year) — no duplicates."""
        dates = _week_dates(2020, n=10)
        df1 = _make_weekly("01001", dates, d0=10.0)
        df2 = _make_weekly("06001", dates, d2=5.0)
        df = pd.concat([df1, df2], ignore_index=True)

        result = compute_drought_scores(df, scoring_year=2020)
        assert len(result) == result.groupby(["fips", "year"]).ngroups


class TestMetadata:
    """Test metadata sidecar writing."""

    def test_metadata_content(self, tmp_path):
        """Metadata JSON has correct source/confidence/attribution."""
        meta_path = tmp_path / "test_metadata.json"
        _write_metadata(meta_path, 2020)

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == METADATA_SOURCE
        assert meta["confidence"] == METADATA_CONFIDENCE
        assert meta["attribution"] == METADATA_ATTRIBUTION
        assert meta["data_vintage"] == "2020"
        assert "retrieved_at" in meta
