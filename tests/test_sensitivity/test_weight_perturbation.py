"""Tests for sensitivity/weight_perturbation.py."""

import numpy as np
import pandas as pd
import pytest

from sensitivity.weight_perturbation import _perturb_weights, run_weight_perturbation


class TestPerturbWeights:
    """Unit tests for the weight perturbation helper."""

    def test_weights_sum_to_one(self):
        """Perturbed + re-normalized weights sum to 1.0."""
        rng = np.random.default_rng(42)
        base = {"a": 0.3, "b": 0.5, "c": 0.2}
        for _ in range(100):
            perturbed = _perturb_weights(base, 0.30, rng)
            assert sum(perturbed.values()) == pytest.approx(1.0, abs=1e-10)

    def test_all_positive(self):
        """All perturbed weights remain positive."""
        rng = np.random.default_rng(42)
        base = {"a": 0.3, "b": 0.5, "c": 0.2}
        for _ in range(100):
            perturbed = _perturb_weights(base, 0.30, rng)
            assert all(v > 0 for v in perturbed.values())

    def test_zero_perturbation_preserves_weights(self):
        """With 0% perturbation, weights are unchanged."""
        rng = np.random.default_rng(42)
        base = {"a": 0.3, "b": 0.5, "c": 0.2}
        perturbed = _perturb_weights(base, 0.0, rng)
        for k in base:
            assert perturbed[k] == pytest.approx(base[k])


class TestRunWeightPerturbation:
    """Integration tests for the full Monte Carlo run."""

    def test_seed_reproducibility(self, synthetic_harmonized, base_weights):
        """Same seed produces identical results."""
        df1, s1 = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5, seed=42,
        )
        df2, s2 = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5, seed=42,
        )
        pd.testing.assert_frame_equal(df1, df2)
        assert s1 == s2

    def test_different_seed_different_results(self, synthetic_harmonized, base_weights):
        """Different seeds produce different results."""
        df1, _ = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5, seed=42,
        )
        df2, _ = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5, seed=99,
        )
        # Mean ranks should differ
        assert not np.allclose(df1["mean_rank"].values, df2["mean_rank"].values)

    def test_zero_perturbation_perfect_stability(self, synthetic_harmonized, base_weights):
        """With 0% perturbation, all iterations match primary → rank_std = 0."""
        df, summary = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5,
            perturbation_pct=0.0,
        )
        assert (df["rank_std"] == 0).all()
        assert summary["spearman_distribution"]["mean"] == pytest.approx(1.0)

    def test_output_schema(self, synthetic_harmonized, base_weights):
        """Verify all expected columns present in detail_df."""
        df, summary = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5,
        )
        expected_cols = {
            "fips", "primary_rank", "mean_rank", "rank_std",
            "rank_p05", "rank_p95", "max_rank_shift", "is_flagged",
        }
        assert expected_cols == set(df.columns)

    def test_output_shape(self, synthetic_harmonized, base_weights):
        """Output has one row per county in scoring universe."""
        df, _ = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5,
        )
        # Should have rows (number depends on universe, but > 0)
        assert len(df) > 0

    def test_summary_spearman_distribution(self, synthetic_harmonized, base_weights):
        """Summary contains Spearman distribution percentiles."""
        _, summary = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=10,
        )
        sp = summary["spearman_distribution"]
        assert "p05" in sp
        assert "p50" in sp
        assert "p95" in sp
        assert "mean" in sp
        # All values should be valid correlations
        assert -1.0 <= sp["p05"] <= 1.0
        assert -1.0 <= sp["p95"] <= 1.0

    def test_is_flagged_type(self, synthetic_harmonized, base_weights):
        """is_flagged column contains booleans."""
        df, _ = run_weight_perturbation(
            synthetic_harmonized, base_weights, n_iterations=5,
        )
        assert df["is_flagged"].dtype == bool
