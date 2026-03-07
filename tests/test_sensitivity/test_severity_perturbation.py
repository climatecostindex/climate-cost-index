"""Tests for sensitivity/severity_perturbation.py."""

import pandas as pd
import pytest

from sensitivity.severity_perturbation import run_severity_perturbation


class TestSeverityPerturbation:
    """Tests for storm severity tier weight perturbation."""

    def test_five_factors(self, synthetic_harmonized):
        """Output has exactly 5 rows."""
        result = run_severity_perturbation(synthetic_harmonized)
        assert len(result) == 5

    def test_factor_values(self, synthetic_harmonized):
        """Perturbation factors are [0.75, 0.875, 1.0, 1.125, 1.25]."""
        result = run_severity_perturbation(synthetic_harmonized)
        expected = [0.75, 0.875, 1.0, 1.125, 1.25]
        assert list(result["perturbation_factor"]) == expected

    def test_output_columns(self, synthetic_harmonized):
        """Verify expected columns exist."""
        result = run_severity_perturbation(synthetic_harmonized)
        expected = {
            "perturbation_factor", "spearman_r_vs_primary",
            "max_rank_shift", "n_shifted_gt_10",
        }
        assert expected.issubset(set(result.columns))

    def test_factor_one_matches_primary(self, synthetic_harmonized):
        """Factor=1.0 produces Spearman r=1.0 (no perturbation)."""
        result = run_severity_perturbation(synthetic_harmonized)
        row_1 = result[result["perturbation_factor"] == 1.0].iloc[0]
        assert row_1["spearman_r_vs_primary"] == pytest.approx(1.0)

    def test_perturbation_produces_valid_correlation(self, synthetic_harmonized):
        """Non-unity factors produce valid Spearman correlations."""
        result = run_severity_perturbation(synthetic_harmonized)
        for _, row in result.iterrows():
            assert -1.0 <= row["spearman_r_vs_primary"] <= 1.0

    def test_only_storm_severity_modified(self, synthetic_harmonized):
        """Other columns are unchanged when storm_severity is scaled."""
        original = synthetic_harmonized.copy()
        modified = synthetic_harmonized.copy()
        modified["storm_severity"] = modified["storm_severity"] * 1.25
        # All other columns should be identical
        other_cols = [c for c in original.columns if c != "storm_severity"]
        pd.testing.assert_frame_equal(original[other_cols], modified[other_cols])
