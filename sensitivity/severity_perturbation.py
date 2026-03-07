"""Sensitivity to storm severity tier weights: +/-25% perturbation.

Uses linear scaling of the storm_severity column as a first-order proxy
for perturbing the underlying tier weights. This slightly underestimates
sensitivity since actual tier weights affect a nonlinear mapping from
event counts/damage to severity scores. Sufficient for v1 robustness.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.components import get_weights
from config.settings import get_settings
from score.pipeline import compute_cci

from .ranking_utils import compare_rankings

logger = logging.getLogger(__name__)

PERTURBATION_FACTORS = [0.75, 0.875, 1.0, 1.125, 1.25]


def run_severity_perturbation(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """Perturb storm severity values, recompute CCI, compare rankings.

    Scales the storm_severity column by factors from 0.75 to 1.25
    (±25%) and measures ranking stability.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame.

    Returns:
        DataFrame with one row per perturbation factor:
            perturbation_factor, spearman_r_vs_primary, max_rank_shift,
            n_shifted_gt_10
    """
    settings = get_settings()
    weights = get_weights()

    # Primary run (factor = 1.0)
    primary = compute_cci(harmonized_df, weights, settings)
    primary_scores = primary.scores["cci_score"]

    results = []
    for factor in PERTURBATION_FACTORS:
        # Create a copy with scaled storm_severity
        modified_df = harmonized_df.copy()
        if "storm_severity" in modified_df.columns:
            modified_df["storm_severity"] = modified_df["storm_severity"] * factor

        alt = compute_cci(modified_df, weights, settings)
        alt_scores = alt.scores["cci_score"]

        comparison = compare_rankings(primary_scores, alt_scores)

        results.append({
            "perturbation_factor": factor,
            "spearman_r_vs_primary": comparison["spearman_r"],
            "max_rank_shift": comparison["max_rank_shift"],
            "n_shifted_gt_10": comparison["n_shifted_gt_10"],
        })

        logger.info(
            "Severity perturbation factor=%.3f: Spearman r=%.4f, max shift=%d",
            factor, comparison["spearman_r"], comparison["max_rank_shift"],
        )

    return pd.DataFrame(results)
