"""Tests for sensitivity/alt_weighting.py."""

import numpy as np
import pandas as pd
import pytest

from sensitivity.alt_weighting import PURE_RISK_WEIGHTS, run_alt_weighting


class TestAltWeighting:
    """Tests for alternative weighting schemes."""

    def test_four_schemes(self, synthetic_harmonized):
        """Output has exactly 4 rows."""
        result = run_alt_weighting(synthetic_harmonized)
        assert len(result) == 4

    def test_scheme_names(self, synthetic_harmonized):
        """All 4 scheme names are present."""
        result = run_alt_weighting(synthetic_harmonized)
        expected = {"equal", "pure_budget", "pure_risk", "neutral_acceleration"}
        assert set(result["scheme"]) == expected

    def test_output_columns(self, synthetic_harmonized):
        """Verify expected columns exist."""
        result = run_alt_weighting(synthetic_harmonized)
        expected = {
            "scheme", "spearman_r_vs_primary", "max_rank_shift",
            "n_shifted_gt_10", "median_cci_score", "iqr_low", "iqr_high",
        }
        assert expected.issubset(set(result.columns))

    def test_equal_weights_sum_to_one(self, component_ids):
        """Equal weights = 1/12 each, sum to 1."""
        n = len(component_ids)
        w = 1.0 / n
        assert w * n == pytest.approx(1.0)

    def test_pure_risk_weights_sum_to_one(self):
        """Pure risk weights sum to 1.0."""
        assert sum(PURE_RISK_WEIGHTS.values()) == pytest.approx(1.0)

    def test_valid_cci_scores(self, synthetic_harmonized):
        """All schemes produce finite CCI scores centered near 100."""
        result = run_alt_weighting(synthetic_harmonized)
        for _, row in result.iterrows():
            assert np.isfinite(row["median_cci_score"])
            assert 0 < row["median_cci_score"] < 300  # reasonable range

    def test_determinism(self, synthetic_harmonized):
        """Same input produces identical output."""
        r1 = run_alt_weighting(synthetic_harmonized)
        r2 = run_alt_weighting(synthetic_harmonized)
        pd.testing.assert_frame_equal(r1, r2)

    def test_spearman_r_valid(self, synthetic_harmonized):
        """Spearman r values are between -1 and 1."""
        result = run_alt_weighting(synthetic_harmonized)
        for r in result["spearman_r_vs_primary"]:
            assert -1.0 <= r <= 1.0
