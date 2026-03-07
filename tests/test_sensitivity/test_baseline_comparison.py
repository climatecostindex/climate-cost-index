"""Tests for sensitivity/baseline_comparison.py."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


class TestBaselineComparison:
    """Tests for run_baseline_comparison."""

    def test_missing_alt_data_returns_graceful_result(self, synthetic_harmonized):
        """When no alternative baseline columns exist, return data_unavailable."""
        from sensitivity.baseline_comparison import run_baseline_comparison

        result = run_baseline_comparison(synthetic_harmonized)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        # Primary row
        primary = result[result["baseline"] == "1991-2020"].iloc[0]
        assert primary["spearman_r_vs_primary"] == 1.0
        assert primary["status"] == "primary"
        # Alt row
        alt = result[result["baseline"] == "1981-2010"].iloc[0]
        assert alt["status"] == "data_unavailable"
        assert pd.isna(alt["spearman_r_vs_primary"])

    def test_output_columns_present(self, synthetic_harmonized):
        """Verify all required columns are present."""
        from sensitivity.baseline_comparison import run_baseline_comparison

        result = run_baseline_comparison(synthetic_harmonized)

        expected_cols = {"baseline", "spearman_r_vs_primary", "max_rank_shift", "n_shifted_gt_10", "status"}
        assert expected_cols.issubset(set(result.columns))

    def test_primary_baseline_row_always_present(self, synthetic_harmonized):
        """The 1991-2020 primary row should always exist."""
        from sensitivity.baseline_comparison import run_baseline_comparison

        result = run_baseline_comparison(synthetic_harmonized)

        primary_rows = result[result["baseline"] == "1991-2020"]
        assert len(primary_rows) == 1
        assert primary_rows.iloc[0]["spearman_r_vs_primary"] == 1.0

    def test_alt_data_detected_and_used(self, synthetic_harmonized):
        """When alternative columns are present, they should be detected and used."""
        from sensitivity.baseline_comparison import run_baseline_comparison

        # Add alternative baseline columns with slightly shifted values
        df = synthetic_harmonized.copy()
        df["hdd_anomaly__1981_2010"] = df["hdd_anomaly"] + 5.0
        df["cdd_anomaly__1981_2010"] = df["cdd_anomaly"] + 3.0

        # Mock compute_cci to avoid full pipeline
        mock_output = MagicMock()
        mock_scores = pd.DataFrame(
            {"cci_score": np.arange(20, dtype=float)},
            index=[f"{i:05d}" for i in range(1, 21)],
        )
        mock_output.scores = mock_scores

        # Second call returns slightly different scores
        mock_output_alt = MagicMock()
        alt_scores = pd.DataFrame(
            {"cci_score": np.arange(20, dtype=float) + np.random.default_rng(42).normal(0, 0.5, 20)},
            index=[f"{i:05d}" for i in range(1, 21)],
        )
        mock_output_alt.scores = alt_scores

        with patch("sensitivity.baseline_comparison.compute_cci", side_effect=[mock_output, mock_output_alt]):
            result = run_baseline_comparison(df)

        alt_row = result[result["baseline"] == "1981-2010"].iloc[0]
        assert alt_row["status"] == "computed"
        assert not pd.isna(alt_row["spearman_r_vs_primary"])
        # With small perturbation, should be highly correlated
        assert alt_row["spearman_r_vs_primary"] > 0.8

    def test_does_not_crash_on_empty_df(self):
        """Empty DataFrame should not crash."""
        from sensitivity.baseline_comparison import run_baseline_comparison

        empty_df = pd.DataFrame()
        result = run_baseline_comparison(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        alt = result[result["baseline"] == "1981-2010"].iloc[0]
        assert alt["status"] == "data_unavailable"
