"""Tests for sensitivity/window_sensitivity.py."""

import pandas as pd
import pytest

from sensitivity.window_sensitivity import run_window_sensitivity


class TestWindowSensitivity:
    """Tests for acceleration window sweep."""

    def test_four_windows(self, synthetic_harmonized):
        """Output has exactly 4 rows."""
        result = run_window_sensitivity(synthetic_harmonized)
        assert len(result) == 4

    def test_window_values(self, synthetic_harmonized):
        """Window years are [3, 5, 7, 10]."""
        result = run_window_sensitivity(synthetic_harmonized)
        assert list(result["window_years"]) == [3, 5, 7, 10]

    def test_output_columns(self, synthetic_harmonized):
        """Verify expected columns exist."""
        result = run_window_sensitivity(synthetic_harmonized)
        expected = {
            "window_years", "spearman_r_vs_primary", "max_rank_shift",
            "median_acceleration", "pct_at_lower_bound", "pct_at_upper_bound",
        }
        assert expected.issubset(set(result.columns))

    def test_default_window_high_correlation(self, synthetic_harmonized):
        """Window=5 (default for most components) should have high Spearman r."""
        result = run_window_sensitivity(synthetic_harmonized)
        row_5 = result[result["window_years"] == 5].iloc[0]
        # Most continuous components default to 5yr, so should correlate well
        assert row_5["spearman_r_vs_primary"] > 0.8

    def test_median_acceleration_reasonable(self, synthetic_harmonized):
        """Median acceleration should be near 1.0."""
        result = run_window_sensitivity(synthetic_harmonized)
        for _, row in result.iterrows():
            assert 0.5 <= row["median_acceleration"] <= 3.0

    def test_pct_at_bounds_valid(self, synthetic_harmonized):
        """Percentages at bounds should be between 0 and 100."""
        result = run_window_sensitivity(synthetic_harmonized)
        for _, row in result.iterrows():
            assert 0 <= row["pct_at_lower_bound"] <= 100
            assert 0 <= row["pct_at_upper_bound"] <= 100
