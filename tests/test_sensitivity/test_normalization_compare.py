"""Tests for sensitivity/normalization_compare.py."""

import numpy as np
import pandas as pd
import pytest

from sensitivity.normalization_compare import run_normalization_comparison


class TestNormalizationComparison:
    """Tests for percentile vs z-score normalization."""

    def test_two_rows(self, synthetic_harmonized):
        """Output has exactly 2 rows."""
        result = run_normalization_comparison(synthetic_harmonized)
        assert len(result) == 2

    def test_normalization_names(self, synthetic_harmonized):
        """Rows are 'percentile' and 'z_score'."""
        result = run_normalization_comparison(synthetic_harmonized)
        assert set(result["normalization"]) == {"percentile", "z_score"}

    def test_output_columns(self, synthetic_harmonized):
        """Verify expected columns exist."""
        result = run_normalization_comparison(synthetic_harmonized)
        expected = {
            "normalization", "spearman_r_vs_primary", "kendall_tau",
            "max_rank_shift", "n_shifted_gt_10",
        }
        assert expected.issubset(set(result.columns))

    def test_percentile_row_perfect_match(self, synthetic_harmonized):
        """Percentile row has Spearman r = 1.0 (matches itself)."""
        result = run_normalization_comparison(synthetic_harmonized)
        pct_row = result[result["normalization"] == "percentile"].iloc[0]
        assert pct_row["spearman_r_vs_primary"] == pytest.approx(1.0)

    def test_zscore_rank_similarity(self, synthetic_harmonized):
        """Z-score should produce reasonably similar rankings for uniform data."""
        result = run_normalization_comparison(synthetic_harmonized)
        z_row = result[result["normalization"] == "z_score"].iloc[0]
        # For well-behaved synthetic data, expect decent correlation
        assert z_row["spearman_r_vs_primary"] > 0.5

    def test_handles_zero_variance(self, synthetic_harmonized, component_ids):
        """Gracefully handles a component with zero variance."""
        # Set one component to constant value
        modified = synthetic_harmonized.copy()
        modified["flood_exposure"] = 42.0
        # Should not raise
        result = run_normalization_comparison(modified)
        assert len(result) == 2
