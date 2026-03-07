"""Sensitivity to acceleration trailing window: 3/5/7/10-year windows."""

from __future__ import annotations

import logging
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.components import COMPONENTS, ComponentDef, get_weights
from config.settings import get_settings
from score.acceleration import (
    compute_acceleration_multipliers,
    compute_theil_sen_slopes,
)
from score.composite import calibrate_k, compute_component_scores
from score.pipeline import CCIOutput, compute_cci

from .ranking_utils import compare_rankings

logger = logging.getLogger(__name__)

WINDOWS = [3, 5, 7, 10]

# Static components: acceleration is always 1.0 regardless of window
STATIC_COMPONENTS = {"flood_exposure", "wildfire_score"}


def run_window_sensitivity(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute acceleration at each window length, compare rankings.

    Tests windows of 3, 5, 7, and 10 years for all non-static components.
    Static components (flood, wildfire) always get neutral acceleration.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame.

    Returns:
        DataFrame with one row per window:
            window_years, spearman_r_vs_primary, max_rank_shift,
            median_acceleration, pct_at_lower_bound, pct_at_upper_bound
    """
    settings = get_settings()
    weights = get_weights()

    # Primary run
    primary = compute_cci(harmonized_df, weights, settings)
    primary_scores = primary.scores["cci_score"]

    # Extract centered data for lightweight recomputation
    from sensitivity.weight_perturbation import _extract_centered
    centered, comp_ids = _extract_centered(harmonized_df, settings)

    results = []
    for window in WINDOWS:
        # Compute slopes with overridden window for all non-static components
        patched_components = _patch_windows(window)

        with patch("score.acceleration.COMPONENTS", patched_components):
            slopes = compute_theil_sen_slopes(
                harmonized_df,
                scoring_year=settings.scoring_year,
                component_ids=comp_ids,
                min_completeness=settings.acceleration_min_completeness,
            )

        accelerations = compute_acceleration_multipliers(
            slopes,
            bounds=settings.acceleration_bounds,
            epsilon_factor=settings.acceleration_denominator_epsilon_factor,
        )

        # Recompute Steps 8-10 with new accelerations
        comp_scores = compute_component_scores(
            centered, weights, primary.penalties, accelerations,
        )
        raw_composite = comp_scores.sum(axis=1)
        k = calibrate_k(raw_composite, target_iqr=settings.target_iqr)
        cci = 100 + k * raw_composite
        cci.name = "cci_score"

        comparison = compare_rankings(primary_scores, cci)

        # Acceleration distribution stats
        accel_cols = [c for c in accelerations.columns if c.endswith("_acceleration")]
        all_accel = accelerations[accel_cols].values.flatten()
        valid_accel = all_accel[~np.isnan(all_accel)]

        lower, upper = settings.acceleration_bounds
        n_valid = len(valid_accel)
        pct_lower = float((valid_accel == lower).sum() / n_valid * 100) if n_valid > 0 else 0.0
        pct_upper = float((valid_accel == upper).sum() / n_valid * 100) if n_valid > 0 else 0.0

        results.append({
            "window_years": window,
            "spearman_r_vs_primary": comparison["spearman_r"],
            "max_rank_shift": comparison["max_rank_shift"],
            "median_acceleration": float(np.median(valid_accel)) if n_valid > 0 else 1.0,
            "pct_at_lower_bound": pct_lower,
            "pct_at_upper_bound": pct_upper,
        })

        logger.info(
            "Window %d yr: Spearman r=%.4f, median accel=%.3f, "
            "%.1f%% at lower, %.1f%% at upper bound",
            window, comparison["spearman_r"],
            results[-1]["median_acceleration"],
            pct_lower, pct_upper,
        )

    return pd.DataFrame(results)


def _patch_windows(window: int) -> dict[str, ComponentDef]:
    """Create a patched COMPONENTS dict with overridden acceleration windows.

    Static components keep their original window. All others get the test window.
    ComponentDef is frozen, so we must create new instances.
    """
    patched = {}
    for comp_id, comp_def in COMPONENTS.items():
        if comp_id in STATIC_COMPONENTS:
            patched[comp_id] = comp_def
        else:
            patched[comp_id] = ComponentDef(
                id=comp_def.id,
                name=comp_def.name,
                source_module=comp_def.source_module,
                attribution=comp_def.attribution,
                confidence=comp_def.confidence,
                precedence_tier=comp_def.precedence_tier,
                base_weight=comp_def.base_weight,
                transform=comp_def.transform,
                acceleration_window=window,
                unit=comp_def.unit,
            )
    return patched
