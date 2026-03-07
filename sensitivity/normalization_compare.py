"""Compare percentile vs z-score normalization rank-order correlation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.components import COMPONENTS, get_weights
from config.settings import get_settings
from score.composite import calibrate_k, compute_component_scores
from score.missingness import handle_missingness
from score.pipeline import CCIOutput, compute_cci
from score.transform_inputs import transform_inputs
from score.universe import define_universe
from score.winsorize import winsorize

from .ranking_utils import compare_rankings

logger = logging.getLogger(__name__)


def run_normalization_comparison(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute CCI using z-score instead of percentile, compare rankings.

    The primary pipeline uses percentile rank normalization (Step 3).
    This module substitutes z-score normalization and compares the
    resulting county rankings.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame.

    Returns:
        DataFrame with 2 rows ("percentile" and "z_score"):
            normalization, spearman_r_vs_primary, kendall_tau,
            max_rank_shift, n_shifted_gt_10
    """
    settings = get_settings()
    weights = get_weights()

    # Primary run (percentile normalization)
    primary = compute_cci(harmonized_df, weights, settings)
    primary_scores = primary.scores["cci_score"]

    # Z-score alternative path
    z_scores = _compute_zscore_cci(
        harmonized_df, weights, settings, primary,
    )

    # Compare
    comparison_pct = compare_rankings(primary_scores, primary_scores)
    comparison_z = compare_rankings(primary_scores, z_scores)

    results = [
        {
            "normalization": "percentile",
            "spearman_r_vs_primary": comparison_pct["spearman_r"],
            "kendall_tau": comparison_pct["kendall_tau"],
            "max_rank_shift": comparison_pct["max_rank_shift"],
            "n_shifted_gt_10": comparison_pct["n_shifted_gt_10"],
        },
        {
            "normalization": "z_score",
            "spearman_r_vs_primary": comparison_z["spearman_r"],
            "kendall_tau": comparison_z["kendall_tau"],
            "max_rank_shift": comparison_z["max_rank_shift"],
            "n_shifted_gt_10": comparison_z["n_shifted_gt_10"],
        },
    ]

    logger.info(
        "Normalization comparison: z-score Spearman r=%.4f, Kendall tau=%.4f vs percentile",
        comparison_z["spearman_r"], comparison_z["kendall_tau"],
    )

    return pd.DataFrame(results)


def _compute_zscore_cci(
    harmonized_df: pd.DataFrame,
    weights: dict[str, float],
    settings,
    primary: CCIOutput,
) -> pd.Series:
    """Compute CCI using z-score normalization instead of percentile.

    Replaces Step 3 (percentile rank) with z-score normalization.
    Reuses penalties and accelerations from the primary run.
    """
    component_ids = [c for c in COMPONENTS if c in harmonized_df.columns]
    scoring_year = settings.scoring_year

    scoring_year_df = harmonized_df[harmonized_df["year"] == scoring_year].copy()
    if "fips" in scoring_year_df.columns:
        scoring_year_df = scoring_year_df.set_index("fips")

    # Steps 1-2: same as primary
    transformed = transform_inputs(scoring_year_df)
    winsorized = winsorize(transformed, percentile=settings.winsorize_percentile)

    # Step 3 (alternative): z-score instead of percentile
    universe = define_universe(winsorized)
    scored = winsorized.loc[universe].copy()

    for comp_id in component_ids:
        if comp_id not in scored.columns:
            continue
        col = scored[comp_id]
        valid = col.dropna()
        if len(valid) < 2:
            continue
        mean = valid.mean()
        std = valid.std()
        if std < 1e-10:
            # Zero-variance: set to 0 (neutral)
            scored.loc[col.notna(), comp_id] = 0.0
            logger.warning("Component %s has zero variance; z-score set to 0", comp_id)
        else:
            scored[comp_id] = (col - mean) / std

    # Step 4: centering (z-scores are already ~zero-mean, subtract 0)
    # No additional centering needed for z-scores

    # Step 7: missingness
    scored = handle_missingness(scored)

    # Steps 8-9: component scores using primary penalties/accelerations
    comp_scores = compute_component_scores(
        scored, weights, primary.penalties, primary.accelerations,
    )
    raw_composite = comp_scores.sum(axis=1)

    # Step 10: scale
    k = calibrate_k(raw_composite, target_iqr=settings.target_iqr)
    cci = 100 + k * raw_composite
    cci.name = "cci_score"

    return cci
