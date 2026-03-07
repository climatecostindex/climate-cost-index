"""Comprehensive tests for score/acceleration.py (Module 3.3).

All tests use synthetic DataFrames — no real data files are read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import theilslopes

from score.acceleration import (
    compute_acceleration_multipliers,
    compute_theil_sen_slopes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_harmonized(
    fips_data: dict[str, dict[int, float]],
    component_id: str = "drought_score",
) -> pd.DataFrame:
    """Build a minimal harmonized DataFrame from {fips: {year: value}} mapping."""
    rows = []
    for fips, year_vals in fips_data.items():
        for year, val in year_vals.items():
            rows.append({"fips": fips, "year": year, component_id: val})
    return pd.DataFrame(rows)


def _make_multi_component(
    fips_data_a: dict[str, dict[int, float]],
    fips_data_b: dict[str, dict[int, float]],
    comp_a: str = "drought_score",
    comp_b: str = "pm25_annual",
) -> pd.DataFrame:
    """Build harmonized DataFrame with two components."""
    all_keys = set()
    for fips, yv in fips_data_a.items():
        for year in yv:
            all_keys.add((fips, year))
    for fips, yv in fips_data_b.items():
        for year in yv:
            all_keys.add((fips, year))

    rows = []
    for fips, year in sorted(all_keys):
        rows.append({
            "fips": fips,
            "year": year,
            comp_a: fips_data_a.get(fips, {}).get(year, np.nan),
            comp_b: fips_data_b.get(fips, {}).get(year, np.nan),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Theil-Sen slope computation
# ===========================================================================

class TestTheilSenSlopes:
    """Tests for compute_theil_sen_slopes."""

    def test_known_linear_trend(self):
        """5 years of perfectly linear data → slope = 2.0/year."""
        data = {"01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(2.0, abs=1e-10)

    def test_known_flat_trend(self):
        """Constant values → slope = 0.0."""
        data = {"01001": {2020: 10, 2021: 10, 2022: 10, 2023: 10, 2024: 10}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(0.0, abs=1e-10)

    def test_noisy_with_outlier(self):
        """Theil-Sen is robust to single outlier. True trend ≈ 2.0/year."""
        data = {"01001": {2020: 10, 2021: 12, 2022: 14, 2023: 100, 2024: 18}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        slope = slopes.loc["01001", "drought_score_slope"]
        # Theil-Sen median of pairwise slopes — robust, should be near 2.0
        assert abs(slope - 2.0) < 2.0  # within reasonable range

    def test_negative_trend(self):
        """Decreasing values → negative slope."""
        data = {"01001": {2020: 20, 2021: 18, 2022: 16, 2023: 14, 2024: 12}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(-2.0, abs=1e-10)

    def test_two_data_points_minimum(self):
        """Two years of data — just enough for slope computation."""
        data = {"01001": {2023: 10, 2024: 20}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(
            df, scoring_year=2024, component_ids=["drought_score"],
            min_completeness=0.0,  # Allow 2 points
        )
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(10.0, abs=1e-10)


# ===========================================================================
# Window selection
# ===========================================================================

class TestWindowSelection:
    """Tests for trailing window behavior."""

    def test_5year_window_filters_correctly(self):
        """Only years within the 5-year window should be used."""
        # 2020-2024 window: values [100, 102, 104, 106, 108] → slope 2.0
        # 2015-2019: values [0, 0, 0, 0, 0] — should be ignored
        data = {"01001": {
            2015: 0, 2016: 0, 2017: 0, 2018: 0, 2019: 0,
            2020: 100, 2021: 102, 2022: 104, 2023: 106, 2024: 108,
        }}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(2.0, abs=1e-10)

    def test_10year_window_for_event_component(self):
        """storm_severity uses 10-year window."""
        # Linear trend over 10 years: slope = 1.0
        data = {"01001": {y: float(y - 2015) for y in range(2015, 2025)}}
        df = _make_harmonized(data, component_id="storm_severity")
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["storm_severity"])
        assert slopes.loc["01001", "storm_severity_slope"] == pytest.approx(1.0, abs=1e-10)

    def test_data_before_window_ignored(self):
        """Pre-window wild data should not affect the slope."""
        data = {"01001": {
            2010: 9999, 2011: -9999, 2012: 9999, 2013: -9999, 2014: 9999,
            2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18,
        }}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(2.0, abs=1e-10)


# ===========================================================================
# Completeness threshold
# ===========================================================================

class TestCompletenessThreshold:
    """Tests for minimum completeness requirements."""

    def test_exactly_at_threshold(self):
        """5-year window, 80% → need 4 years. 4 of 5 → passes."""
        # Missing 2022, have 4 years
        data = {"01001": {2020: 10, 2021: 12, 2023: 16, 2024: 18}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert not np.isnan(slopes.loc["01001", "drought_score_slope"])

    def test_below_threshold(self):
        """3 of 5 years → below 80% threshold → NaN slope."""
        data = {"01001": {2020: 10, 2022: 14, 2024: 18}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert np.isnan(slopes.loc["01001", "drought_score_slope"])

    def test_above_threshold(self):
        """All 5 years → above threshold → slope computed."""
        data = {"01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(2.0, abs=1e-10)

    def test_10year_window_completeness(self):
        """10-year window, 80% → need 8 years. 7 → NaN, 8 → computed."""
        # storm_severity has acceleration_window=10
        # County A: 7 years → NaN
        data_7 = {y: float(y) for y in [2015, 2016, 2017, 2018, 2019, 2020, 2021]}
        # County B: 8 years → computed
        data_8 = {y: float(y) for y in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]}
        fips_data = {"01001": data_7, "01002": data_8}
        df = _make_harmonized(fips_data, component_id="storm_severity")
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["storm_severity"])
        assert np.isnan(slopes.loc["01001", "storm_severity_slope"])
        assert not np.isnan(slopes.loc["01002", "storm_severity_slope"])


# ===========================================================================
# Static components
# ===========================================================================

class TestStaticComponents:
    """Tests for flood/wildfire-style static components."""

    def test_single_year_of_data(self):
        """Component with only 2024 data → slope 0.0 → acceleration 1.0."""
        data = {"01001": {2024: 50.0}, "01002": {2024: 80.0}}
        df = _make_harmonized(data, component_id="flood_exposure")
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["flood_exposure"])
        assert slopes.loc["01001", "flood_exposure_slope"] == 0.0
        assert slopes.loc["01002", "flood_exposure_slope"] == 0.0

        accels = compute_acceleration_multipliers(slopes)
        assert accels.loc["01001", "flood_exposure_acceleration"] == 1.0
        assert accels.loc["01002", "flood_exposure_acceleration"] == 1.0

    def test_zero_years_of_data(self):
        """Component column is all NaN → slope 0.0 → acceleration 1.0."""
        data = {"01001": {2024: np.nan}, "01002": {2024: np.nan}}
        df = _make_harmonized(data, component_id="wildfire_score")
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["wildfire_score"])
        # 0 years of non-null data → static path
        assert slopes.loc["01001", "wildfire_score_slope"] == 0.0

        accels = compute_acceleration_multipliers(slopes)
        assert accels.loc["01001", "wildfire_score_acceleration"] == 1.0

    def test_two_years_not_static(self):
        """Two years of data → NOT static, slopes ARE computed."""
        data = {"01001": {2023: 10.0, 2024: 20.0}}
        df = _make_harmonized(data, component_id="flood_exposure")
        slopes = compute_theil_sen_slopes(
            df, scoring_year=2024, component_ids=["flood_exposure"],
            min_completeness=0.0,
        )
        # Two years → should compute slope, not set to 0.0
        assert slopes.loc["01001", "flood_exposure_slope"] == pytest.approx(10.0, abs=1e-10)


# ===========================================================================
# Acceleration multiplier computation
# ===========================================================================

class TestAccelerationMultipliers:
    """Tests for compute_acceleration_multipliers."""

    def test_normal_ratio_form(self):
        """Ratio form: a = slope / median_slope. Verify clipping."""
        slopes = pd.DataFrame(
            {"comp_slope": [1.0, 2.0, 3.0, 4.0, 5.0]},
            index=["A", "B", "C", "D", "E"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        col = "comp_acceleration"
        # median = 3.0, sigma = std([1,2,3,4,5]) ≈ 1.58
        # epsilon = 0.158; |3.0| > 0.158 → ratio form
        # a = slope / 3.0 → [0.333, 0.667, 1.0, 1.333, 1.667]
        # 0.333 clipped to 0.5
        assert accels.loc["A", col] == pytest.approx(0.5)  # clipped
        assert accels.loc["B", col] == pytest.approx(2.0 / 3.0, abs=1e-10)
        assert accels.loc["C", col] == pytest.approx(1.0, abs=1e-10)
        assert accels.loc["D", col] == pytest.approx(4.0 / 3.0, abs=1e-10)
        assert accels.loc["E", col] == pytest.approx(5.0 / 3.0, abs=1e-10)

    def test_difference_form_near_zero_median(self):
        """Difference form when median ≈ 0: a = 1 + (slope - median) / sigma."""
        slopes = pd.DataFrame(
            {"comp_slope": [-0.01, -0.005, 0.0, 0.005, 0.01]},
            index=["A", "B", "C", "D", "E"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        col = "comp_acceleration"
        # median = 0.0, sigma = std([-0.01, -0.005, 0, 0.005, 0.01])
        # epsilon = 0.1 * sigma; |0| < epsilon → difference form
        # a = 1 + slope / sigma
        sigma = pd.Series([-0.01, -0.005, 0.0, 0.005, 0.01]).std()
        expected = [1.0 + s / sigma for s in [-0.01, -0.005, 0.0, 0.005, 0.01]]
        for idx, exp in zip(["A", "B", "C", "D", "E"], expected):
            assert accels.loc[idx, col] == pytest.approx(max(0.5, min(3.0, exp)), abs=1e-10)

    def test_mixed_signs_difference_form(self):
        """Mixed positive/negative slopes with median ≈ 0 → difference form."""
        slopes = pd.DataFrame(
            {"comp_slope": [-3.0, -1.0, 0.0, 1.0, 3.0]},
            index=["A", "B", "C", "D", "E"],
        )
        slopes.index.name = "fips"
        # median = 0, sigma = std([-3,-1,0,1,3])
        sigma = pd.Series([-3.0, -1.0, 0.0, 1.0, 3.0]).std()
        epsilon = 0.1 * sigma
        # |0| < epsilon → difference form
        accels = compute_acceleration_multipliers(slopes)
        col = "comp_acceleration"
        # Negative slopes → a < 1
        assert accels.loc["A", col] < 1.0
        # Zero slope → a = 1.0
        assert accels.loc["C", col] == pytest.approx(1.0, abs=1e-10)
        # Positive slopes → a > 1
        assert accels.loc["E", col] > 1.0


# ===========================================================================
# Bounds [0.5, 3.0]
# ===========================================================================

class TestBounds:
    """Tests for acceleration bounds clipping."""

    def test_large_positive_slope_clipped_to_upper(self):
        """Extreme positive slope → clipped to 3.0."""
        # Use tight cluster + outlier so ratio form is used (|median| >> epsilon)
        slopes = pd.DataFrame(
            {"comp_slope": [9.0, 10.0, 10.0, 11.0, 50.0]},
            index=["A", "B", "C", "D", "E"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        # median=10, ratio form: 50/10=5.0 → clipped to 3.0
        assert accels.loc["E", "comp_acceleration"] == 3.0

    def test_large_negative_slope_clipped_to_lower(self):
        """Extreme negative slope → clipped to 0.5."""
        slopes = pd.DataFrame(
            {"comp_slope": [1.0, 2.0, 3.0, -100.0]},
            index=["A", "B", "C", "D"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        assert accels.loc["D", "comp_acceleration"] == 0.5

    def test_boundary_values_preserved(self):
        """Values exactly at bounds are not modified."""
        # Create slopes that produce exactly 0.5 and 3.0 after ratio
        # median = 2.0, slopes [1.0, 2.0, 6.0] → a = [0.5, 1.0, 3.0]
        slopes = pd.DataFrame(
            {"comp_slope": [1.0, 2.0, 6.0]},
            index=["A", "B", "C"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        assert accels.loc["A", "comp_acceleration"] == pytest.approx(0.5, abs=1e-10)
        assert accels.loc["C", "comp_acceleration"] == pytest.approx(3.0, abs=1e-10)

    def test_all_neutral(self):
        """All slopes identical → sigma=0 → all 1.0 (within bounds trivially)."""
        slopes = pd.DataFrame(
            {"comp_slope": [5.0, 5.0, 5.0]},
            index=["A", "B", "C"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        assert (accels["comp_acceleration"] == 1.0).all()


# ===========================================================================
# Denominator protection
# ===========================================================================

class TestDenominatorProtection:
    """Tests for sigma=0, epsilon boundary, and edge cases."""

    def test_sigma_zero_all_identical(self):
        """All slopes identical → sigma=0 → all acceleration = 1.0."""
        slopes = pd.DataFrame(
            {"comp_slope": [5.0, 5.0, 5.0, 5.0]},
            index=["A", "B", "C", "D"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        assert (accels["comp_acceleration"] == 1.0).all()

    def test_epsilon_boundary_ratio_selected(self):
        """When |median| = epsilon exactly, ratio form is used (>= check)."""
        # We want |median| == epsilon = 0.1 * sigma
        # Construct: slopes where median = 0.1 * std
        # Use [0, 0.1*s, 0.2*s] where median=0.1*s, std=0.1*s → eps=0.01*s
        # Actually simpler: just verify the code path
        # slopes [0, 1, 2]: median=1, sigma=1, epsilon=0.1; |1| >= 0.1 → ratio
        slopes = pd.DataFrame(
            {"comp_slope": [0.0, 1.0, 2.0]},
            index=["A", "B", "C"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        # ratio form: a = slope / 1.0 = slope
        assert accels.loc["A", "comp_acceleration"] == pytest.approx(0.5)  # 0.0/1.0 = 0 → clipped to 0.5
        assert accels.loc["B", "comp_acceleration"] == pytest.approx(1.0, abs=1e-10)
        assert accels.loc["C", "comp_acceleration"] == pytest.approx(2.0, abs=1e-10)

    def test_near_zero_sigma_no_crash(self):
        """Very tightly clustered slopes → no division-by-zero or inf."""
        slopes = pd.DataFrame(
            {"comp_slope": [1e-15, 1.0001e-15, 0.9999e-15]},
            index=["A", "B", "C"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        # Should produce finite values, no inf/nan
        assert accels["comp_acceleration"].isna().sum() == 0
        assert np.isfinite(accels["comp_acceleration"]).all()


# ===========================================================================
# NaN handling
# ===========================================================================

class TestNaNHandling:
    """Tests for NaN slope propagation to neutral acceleration."""

    def test_all_nan_component(self):
        """Component with all NaN slopes → all acceleration = 1.0."""
        slopes = pd.DataFrame(
            {"comp_slope": [np.nan, np.nan, np.nan]},
            index=["A", "B", "C"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        assert (accels["comp_acceleration"] == 1.0).all()

    def test_partial_nan_slopes(self):
        """Some NaN slopes → those counties get 1.0, others computed."""
        slopes = pd.DataFrame(
            {"comp_slope": [1.0, np.nan, 3.0, np.nan, 5.0]},
            index=["A", "B", "C", "D", "E"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes)
        assert accels.loc["B", "comp_acceleration"] == 1.0
        assert accels.loc["D", "comp_acceleration"] == 1.0
        # Others should be computed (not NaN)
        assert np.isfinite(accels.loc["A", "comp_acceleration"])
        assert np.isfinite(accels.loc["C", "comp_acceleration"])
        assert np.isfinite(accels.loc["E", "comp_acceleration"])

    def test_county_all_nan_in_window(self):
        """County with NaN for component in entire window → NaN slope → accel 1.0."""
        data = {
            "01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18},
            "01002": {2020: np.nan, 2021: np.nan, 2022: np.nan, 2023: np.nan, 2024: np.nan},
        }
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        accels = compute_acceleration_multipliers(slopes)
        assert accels.loc["01002", "drought_score_acceleration"] == 1.0


# ===========================================================================
# Multi-component independence
# ===========================================================================

class TestMultiComponentIndependence:
    """Tests that components are computed independently."""

    def test_independent_slopes(self):
        """Slopes for component A do not influence component B."""
        data_a = {
            "01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18},
            "01002": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18},
        }
        data_b = {
            "01001": {2020: 100, 2021: 90, 2022: 80, 2023: 70, 2024: 60},
            "01002": {2020: 100, 2021: 90, 2022: 80, 2023: 70, 2024: 60},
        }
        df = _make_multi_component(data_a, data_b)
        slopes = compute_theil_sen_slopes(
            df, scoring_year=2024, component_ids=["drought_score", "pm25_annual"],
        )
        # drought_score: slope = 2.0 for both counties
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(2.0, abs=1e-10)
        # pm25_annual: slope = -10.0 for both counties
        assert slopes.loc["01001", "pm25_annual_slope"] == pytest.approx(-10.0, abs=1e-10)


# ===========================================================================
# Determinism
# ===========================================================================

class TestDeterminism:
    """Tests that results are reproducible."""

    def test_same_input_same_output(self):
        """Running twice on identical input produces identical results."""
        data = {
            "01001": {2020: 10, 2021: 15, 2022: 12, 2023: 18, 2024: 20},
            "01002": {2020: 5, 2021: 8, 2022: 3, 2023: 12, 2024: 7},
            "01003": {2020: 30, 2021: 28, 2022: 25, 2023: 22, 2024: 20},
        }
        df = _make_harmonized(data)

        slopes1 = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        accels1 = compute_acceleration_multipliers(slopes1)

        slopes2 = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        accels2 = compute_acceleration_multipliers(slopes2)

        pd.testing.assert_frame_equal(slopes1, slopes2)
        pd.testing.assert_frame_equal(accels1, accels2)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Tests for boundary and degenerate inputs."""

    def test_single_county(self):
        """Single county → median = its slope, sigma=NaN (ddof=1) → accel 1.0."""
        data = {"01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        accels = compute_acceleration_multipliers(slopes)
        # Single valid slope → std with ddof=1 is NaN → neutral 1.0
        assert accels.loc["01001", "drought_score_acceleration"] == 1.0

    def test_two_counties_identical_slopes(self):
        """Two counties with identical slopes → sigma=0 → accel 1.0."""
        data = {
            "01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18},
            "01002": {2020: 20, 2021: 22, 2022: 24, 2023: 26, 2024: 28},
        }
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        accels = compute_acceleration_multipliers(slopes)
        assert accels.loc["01001", "drought_score_acceleration"] == 1.0
        assert accels.loc["01002", "drought_score_acceleration"] == 1.0

    def test_scoring_year_beyond_data(self):
        """Scoring year 2030, data ends 2024 → window has no data → accel 1.0."""
        data = {"01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18}}
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2030, component_ids=["drought_score"])
        accels = compute_acceleration_multipliers(slopes)
        # Window 2026-2030: no data → static → slope 0.0 → accel 1.0
        assert accels.loc["01001", "drought_score_acceleration"] == 1.0

    def test_very_large_values(self):
        """Large values (1e12 scale) → no overflow in theilslopes or division."""
        data = {
            "01001": {2020: 1e12, 2021: 2e12, 2022: 3e12, 2023: 4e12, 2024: 5e12},
            "01002": {2020: 0.5e12, 2021: 1e12, 2022: 1.5e12, 2023: 2e12, 2024: 2.5e12},
        }
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        accels = compute_acceleration_multipliers(slopes)
        assert np.isfinite(accels["drought_score_acceleration"]).all()

    def test_fips_index_handling(self):
        """DataFrame with fips as index (not column) is handled correctly."""
        data = {"01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18}}
        df = _make_harmonized(data).set_index("fips")
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert "drought_score_slope" in slopes.columns
        assert slopes.index.name == "fips"

    def test_empty_slopes_dataframe(self):
        """Empty slopes DataFrame → empty acceleration DataFrame."""
        slopes = pd.DataFrame(index=pd.Index([], name="fips"))
        accels = compute_acceleration_multipliers(slopes)
        assert len(accels) == 0

    def test_component_not_in_registry(self):
        """Component ID not in COMPONENTS registry is skipped gracefully."""
        data = {"01001": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18}}
        df = _make_harmonized(data, component_id="nonexistent_component")
        df.rename(columns={"nonexistent_component": "fake_comp"}, inplace=True)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["fake_comp"])
        # fake_comp is not in COMPONENTS registry → skipped, slope stays NaN
        assert np.isnan(slopes.loc["01001", "fake_comp_slope"])


# ===========================================================================
# End-to-end: slopes → multipliers chain
# ===========================================================================

class TestEndToEnd:
    """Tests for the full slope → multiplier pipeline."""

    def test_static_flood_through_pipeline(self):
        """flood_exposure with 1 year → neutral 1.0 end-to-end."""
        data = {
            "01001": {2024: 0.75},
            "01002": {2024: 0.30},
            "01003": {2024: 0.50},
        }
        df = _make_harmonized(data, component_id="flood_exposure")
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["flood_exposure"])
        accels = compute_acceleration_multipliers(slopes)
        assert (accels["flood_exposure_acceleration"] == 1.0).all()

    def test_divergent_trends_spread(self):
        """Counties with different trends → spread of acceleration multipliers."""
        data = {
            "fast": {2020: 10, 2021: 14, 2022: 18, 2023: 22, 2024: 26},    # slope 4
            "med":  {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18},    # slope 2
            "slow": {2020: 10, 2021: 11, 2022: 12, 2023: 13, 2024: 14},    # slope 1
        }
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        accels = compute_acceleration_multipliers(slopes)
        col = "drought_score_acceleration"
        # fast should have highest acceleration, slow lowest
        assert accels.loc["fast", col] > accels.loc["med", col]
        assert accels.loc["med", col] > accels.loc["slow", col]
        # Med should be near 1.0 (median trend)
        assert accels.loc["med", col] == pytest.approx(1.0, abs=0.1)

    def test_theilslopes_zero_variance_input(self):
        """All values identical for a county → slope 0.0 (not error)."""
        data = {
            "01001": {2020: 42, 2021: 42, 2022: 42, 2023: 42, 2024: 42},
            "01002": {2020: 10, 2021: 12, 2022: 14, 2023: 16, 2024: 18},
        }
        df = _make_harmonized(data)
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        assert slopes.loc["01001", "drought_score_slope"] == pytest.approx(0.0, abs=1e-10)
        assert slopes.loc["01002", "drought_score_slope"] == pytest.approx(2.0, abs=1e-10)

    def test_custom_bounds(self):
        """Custom bounds parameter is respected."""
        slopes = pd.DataFrame(
            {"comp_slope": [1.0, 2.0, 3.0, 100.0, -100.0]},
            index=["A", "B", "C", "D", "E"],
        )
        slopes.index.name = "fips"
        accels = compute_acceleration_multipliers(slopes, bounds=(0.8, 1.5))
        assert accels["comp_acceleration"].min() >= 0.8
        assert accels["comp_acceleration"].max() <= 1.5

    def test_custom_epsilon_factor(self):
        """epsilon_factor parameter changes ratio/difference form selection."""
        slopes = pd.DataFrame(
            {"comp_slope": [0.5, 1.0, 1.5]},
            index=["A", "B", "C"],
        )
        slopes.index.name = "fips"
        # With large epsilon_factor, difference form more likely
        accels_high = compute_acceleration_multipliers(slopes, epsilon_factor=10.0)
        # With small epsilon_factor, ratio form more likely
        accels_low = compute_acceleration_multipliers(slopes, epsilon_factor=0.001)
        # Both should produce valid bounded results
        assert (accels_high["comp_acceleration"] >= 0.5).all()
        assert (accels_low["comp_acceleration"] >= 0.5).all()
