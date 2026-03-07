"""Leave-one-component-out jackknife sensitivity."""

from __future__ import annotations

import logging

import pandas as pd

from config.components import get_weights
from config.settings import get_settings
from score.pipeline import compute_cci

from .ranking_utils import compare_rankings

logger = logging.getLogger(__name__)


def run_jackknife(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """Remove each component in turn, recompute CCI, compare rankings.

    For each of the 12 components, sets its weight to 0 and re-normalizes
    the remaining weights. Recomputes CCI and compares rankings to the
    primary (all-components) run.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame.

    Returns:
        DataFrame with one row per excluded component:
            excluded_component, spearman_r, max_rank_shift, n_shifted_gt_10,
            most_affected_counties
    """
    settings = get_settings()
    base_weights = get_weights()

    # Primary run with all components
    primary = compute_cci(harmonized_df, base_weights, settings)
    primary_scores = primary.scores["cci_score"]

    results = []
    for comp_id in base_weights:
        # Build reduced weight vector: set excluded to 0, re-normalize
        reduced = {k: v for k, v in base_weights.items() if k != comp_id}
        total = sum(reduced.values())
        if total == 0:
            logger.warning("All weights zero after excluding %s; skipping", comp_id)
            continue
        reduced = {k: v / total for k, v in reduced.items()}
        # Add excluded component with 0 weight so pipeline doesn't error
        reduced[comp_id] = 0.0

        # Recompute CCI
        alt = compute_cci(harmonized_df, reduced, settings)
        alt_scores = alt.scores["cci_score"]

        # Compare rankings
        comparison = compare_rankings(primary_scores, alt_scores)

        results.append({
            "excluded_component": comp_id,
            "spearman_r": comparison["spearman_r"],
            "max_rank_shift": comparison["max_rank_shift"],
            "n_shifted_gt_10": comparison["n_shifted_gt_10"],
            "most_affected_counties": comparison["top_shifted_counties"],
        })

        logger.info(
            "Jackknife: excluded %s → Spearman r=%.4f, max shift=%d",
            comp_id, comparison["spearman_r"], comparison["max_rank_shift"],
        )

    return pd.DataFrame(results)
