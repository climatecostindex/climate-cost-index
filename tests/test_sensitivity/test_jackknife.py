"""Tests for sensitivity/jackknife.py."""

import pandas as pd
import pytest

from sensitivity.jackknife import run_jackknife


class TestJackknife:
    """Tests for leave-one-component-out jackknife."""

    def test_one_row_per_component(self, synthetic_harmonized):
        """Output has 12 rows, one per component."""
        result = run_jackknife(synthetic_harmonized)
        assert len(result) == 12

    def test_output_columns(self, synthetic_harmonized):
        """Verify expected columns exist."""
        result = run_jackknife(synthetic_harmonized)
        expected = {"excluded_component", "spearman_r", "max_rank_shift",
                    "n_shifted_gt_10", "most_affected_counties"}
        assert expected.issubset(set(result.columns))

    def test_spearman_r_less_than_one(self, synthetic_harmonized):
        """Removing a nonzero-weight component should change rankings."""
        result = run_jackknife(synthetic_harmonized)
        # At least some components should produce r < 1.0
        assert (result["spearman_r"] < 1.0).any()

    def test_all_components_listed(self, synthetic_harmonized, component_ids):
        """All 12 component IDs appear in the output."""
        result = run_jackknife(synthetic_harmonized)
        assert set(result["excluded_component"]) == set(component_ids)

    def test_determinism(self, synthetic_harmonized):
        """Same input produces identical output."""
        r1 = run_jackknife(synthetic_harmonized)
        r2 = run_jackknife(synthetic_harmonized)
        pd.testing.assert_frame_equal(
            r1.drop(columns=["most_affected_counties"]),
            r2.drop(columns=["most_affected_counties"]),
        )
