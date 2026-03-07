"""Tests for sensitivity/correlation_threshold.py."""

import pandas as pd
import pytest

from sensitivity.correlation_threshold import run_correlation_threshold_sweep


class TestCorrelationThreshold:
    """Tests for overlap correlation threshold sweep."""

    def test_three_thresholds(self, synthetic_harmonized):
        """Output has exactly 3 rows."""
        result = run_correlation_threshold_sweep(synthetic_harmonized)
        assert len(result) == 3

    def test_threshold_values(self, synthetic_harmonized):
        """Threshold column contains [0.5, 0.6, 0.7]."""
        result = run_correlation_threshold_sweep(synthetic_harmonized)
        assert list(result["threshold"]) == [0.5, 0.6, 0.7]

    def test_output_columns(self, synthetic_harmonized):
        """Verify expected columns exist."""
        result = run_correlation_threshold_sweep(synthetic_harmonized)
        expected = {
            "threshold", "n_pairs_flagged", "spearman_r_vs_primary",
            "max_rank_shift", "n_shifted_gt_10", "penalties",
        }
        assert expected.issubset(set(result.columns))

    def test_primary_threshold_matches(self, synthetic_harmonized):
        """Threshold=0.6 (primary default) should give Spearman r=1.0."""
        result = run_correlation_threshold_sweep(synthetic_harmonized)
        row_06 = result[result["threshold"] == 0.6].iloc[0]
        assert row_06["spearman_r_vs_primary"] == pytest.approx(1.0)

    def test_lower_threshold_more_pairs(self, synthetic_harmonized):
        """Lower threshold flags >= as many component pairs."""
        result = run_correlation_threshold_sweep(synthetic_harmonized)
        n_05 = result[result["threshold"] == 0.5].iloc[0]["n_pairs_flagged"]
        n_07 = result[result["threshold"] == 0.7].iloc[0]["n_pairs_flagged"]
        assert n_05 >= n_07

    def test_determinism(self, synthetic_harmonized):
        """Same input produces identical output."""
        r1 = run_correlation_threshold_sweep(synthetic_harmonized)
        r2 = run_correlation_threshold_sweep(synthetic_harmonized)
        pd.testing.assert_frame_equal(
            r1.drop(columns=["penalties"]),
            r2.drop(columns=["penalties"]),
        )
