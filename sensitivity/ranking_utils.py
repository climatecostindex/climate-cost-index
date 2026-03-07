"""Shared rank comparison utilities for sensitivity modules."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)


def compare_rankings(
    primary_scores: pd.Series,
    alternative_scores: pd.Series,
) -> dict:
    """Compare two sets of CCI scores by ranking.

    Args:
        primary_scores: CCI scores from the primary run, indexed by fips.
        alternative_scores: CCI scores from the alternative run, indexed by fips.

    Returns:
        dict with keys: spearman_r, spearman_pvalue, kendall_tau,
        max_rank_shift, n_shifted_gt_10, n_shifted_gt_15,
        top_shifted_counties (list of dicts with fips, primary_rank, alt_rank, shift)
    """
    # Align on shared FIPS
    shared = primary_scores.index.intersection(alternative_scores.index)
    p = primary_scores.reindex(shared)
    a = alternative_scores.reindex(shared)

    # Compute ranks (higher score = higher rank number)
    p_ranks = p.rank(method="min")
    a_ranks = a.rank(method="min")

    # Spearman and Kendall
    if len(shared) < 2:
        sp_r, sp_p = np.nan, np.nan
        kt = np.nan
    else:
        sp_r, sp_p = spearmanr(p_ranks.values, a_ranks.values)
        kt, _ = kendalltau(p_ranks.values, a_ranks.values)

    # Rank shifts
    shifts = (a_ranks - p_ranks).abs()
    max_shift = int(shifts.max()) if len(shifts) > 0 else 0
    n_gt_10 = int((shifts > 10).sum())
    n_gt_15 = int((shifts > 15).sum())

    # Top 10 shifted counties
    top_idx = shifts.nlargest(10).index
    top_shifted = [
        {
            "fips": fips,
            "primary_rank": int(p_ranks[fips]),
            "alt_rank": int(a_ranks[fips]),
            "shift": int(shifts[fips]),
        }
        for fips in top_idx
    ]

    return {
        "spearman_r": float(sp_r) if not np.isnan(sp_r) else np.nan,
        "spearman_pvalue": float(sp_p) if not np.isnan(sp_p) else np.nan,
        "kendall_tau": float(kt) if not np.isnan(kt) else np.nan,
        "max_rank_shift": max_shift,
        "n_shifted_gt_10": n_gt_10,
        "n_shifted_gt_15": n_gt_15,
        "top_shifted_counties": top_shifted,
    }
