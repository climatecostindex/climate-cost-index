"""Sensitivity to overlap correlation threshold: r = 0.5, 0.6, 0.7."""

from __future__ import annotations

import logging

import pandas as pd

from config.components import get_weights
from config.settings import get_settings
from score.pipeline import compute_cci

from .ranking_utils import compare_rankings

logger = logging.getLogger(__name__)

THRESHOLDS = [0.5, 0.6, 0.7]


def run_correlation_threshold_sweep(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute CCI at each overlap correlation threshold, compare rankings.

    Tests thresholds of 0.5, 0.6, and 0.7 for the overlap penalty step.
    Lower thresholds flag more component pairs as overlapping.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame.

    Returns:
        DataFrame with one row per threshold:
            threshold, n_pairs_flagged, spearman_r_vs_primary, max_rank_shift,
            n_shifted_gt_10, penalties
    """
    settings = get_settings()
    weights = get_weights()

    # Primary run (threshold = 0.6)
    primary = compute_cci(harmonized_df, weights, settings)
    primary_scores = primary.scores["cci_score"]

    results = []
    for threshold in THRESHOLDS:
        modified = settings.model_copy(
            update={"overlap_correlation_threshold": threshold}
        )
        alt = compute_cci(harmonized_df, weights, modified)
        alt_scores = alt.scores["cci_score"]

        comparison = compare_rankings(primary_scores, alt_scores)

        results.append({
            "threshold": threshold,
            "n_pairs_flagged": len([
                1 for v in alt.penalties.values() if v < 1.0
            ]),
            "spearman_r_vs_primary": comparison["spearman_r"],
            "max_rank_shift": comparison["max_rank_shift"],
            "n_shifted_gt_10": comparison["n_shifted_gt_10"],
            "penalties": alt.penalties,
        })

        logger.info(
            "Correlation threshold %.2f: %d penalized components, "
            "Spearman r=%.4f vs primary",
            threshold,
            results[-1]["n_pairs_flagged"],
            comparison["spearman_r"],
        )

    return pd.DataFrame(results)
