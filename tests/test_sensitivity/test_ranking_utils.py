"""Tests for sensitivity/ranking_utils.py."""

import numpy as np
import pandas as pd
import pytest

from sensitivity.ranking_utils import compare_rankings


def test_identical_scores():
    """Identical scores → Spearman r = 1.0, max shift = 0."""
    scores = pd.Series([100, 110, 90, 105, 95], index=["01001", "01002", "01003", "01004", "01005"])
    result = compare_rankings(scores, scores)
    assert result["spearman_r"] == pytest.approx(1.0)
    assert result["max_rank_shift"] == 0
    assert result["n_shifted_gt_10"] == 0
    assert result["n_shifted_gt_15"] == 0


def test_reversed_scores():
    """Reversed scores → Spearman r = -1.0."""
    primary = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    alt = pd.Series([5, 4, 3, 2, 1], index=["a", "b", "c", "d", "e"])
    result = compare_rankings(primary, alt)
    assert result["spearman_r"] == pytest.approx(-1.0)


def test_known_shift():
    """Known small shift: swap last two positions."""
    primary = pd.Series([10, 20, 30, 40, 50], index=["a", "b", "c", "d", "e"])
    alt = pd.Series([10, 20, 30, 50, 40], index=["a", "b", "c", "d", "e"])
    result = compare_rankings(primary, alt)
    assert result["max_rank_shift"] == 1
    assert result["n_shifted_gt_10"] == 0


def test_mismatched_fips_alignment():
    """Only shared FIPS are compared."""
    primary = pd.Series([100, 110, 90], index=["A", "B", "C"])
    alt = pd.Series([105, 95, 80], index=["B", "C", "D"])
    result = compare_rankings(primary, alt)
    # Should only compare on shared FIPS {B, C}
    assert len(result["top_shifted_counties"]) <= 2


def test_single_county():
    """Single shared county: degenerate case."""
    primary = pd.Series([100], index=["01001"])
    alt = pd.Series([110], index=["01001"])
    result = compare_rankings(primary, alt)
    # With 1 county, Spearman is NaN or degenerate
    assert result["max_rank_shift"] == 0


def test_output_keys():
    """Verify all expected keys are present."""
    scores = pd.Series([100, 110, 90], index=["a", "b", "c"])
    result = compare_rankings(scores, scores)
    expected_keys = {
        "spearman_r", "spearman_pvalue", "kendall_tau",
        "max_rank_shift", "n_shifted_gt_10", "n_shifted_gt_15",
        "top_shifted_counties",
    }
    assert expected_keys == set(result.keys())


def test_top_shifted_counties_format():
    """Top shifted counties have correct dict format."""
    primary = pd.Series(range(20), index=[f"f{i:02d}" for i in range(20)])
    alt = pd.Series(list(range(10, 20)) + list(range(10)), index=[f"f{i:02d}" for i in range(20)])
    result = compare_rankings(primary, alt)
    for entry in result["top_shifted_counties"]:
        assert "fips" in entry
        assert "primary_rank" in entry
        assert "alt_rank" in entry
        assert "shift" in entry
    # Should return at most 10
    assert len(result["top_shifted_counties"]) <= 10
