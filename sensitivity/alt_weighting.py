"""Alternative weighting schemes: equal, pure budget, pure risk, neutral acceleration."""

from __future__ import annotations

import logging

import pandas as pd

from config.components import COMPONENTS, get_weights
from config.settings import get_settings
from score.pipeline import compute_cci

from .ranking_utils import compare_rankings

logger = logging.getLogger(__name__)

# Pure risk weights: emphasize hazard severity
PURE_RISK_WEIGHTS: dict[str, float] = {
    "storm_severity": 0.15,
    "flood_exposure": 0.12,
    "wildfire_score": 0.10,
    "drought_score": 0.10,
    "extreme_heat_days": 0.10,
    "pm25_annual": 0.08,
    "aqi_unhealthy_days": 0.05,
    "hdd_anomaly": 0.05,
    "cdd_anomaly": 0.05,
    "energy_cost_attributed": 0.08,
    "health_burden": 0.07,
    "fema_ia_burden": 0.05,
}


def run_alt_weighting(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute CCI under each alternative weighting scheme, compare rankings.

    Schemes:
        1. equal: All 12 components get 1/12.
        2. pure_budget: Primary weights with acceleration forced to 1.0.
        3. pure_risk: Hazard-severity-proportional weights.
        4. neutral_acceleration: Primary weights with acceleration forced to 1.0.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame.

    Returns:
        DataFrame with one row per scheme:
            scheme, spearman_r_vs_primary, max_rank_shift, n_shifted_gt_10,
            median_cci_score, iqr_low, iqr_high
    """
    settings = get_settings()
    base_weights = get_weights()
    comp_ids = list(COMPONENTS.keys())

    # Primary run
    primary = compute_cci(harmonized_df, base_weights, settings)
    primary_scores = primary.scores["cci_score"]

    schemes = _build_schemes(comp_ids, base_weights)
    results = []

    for scheme_name, scheme_weights, neutral_accel in schemes:
        if neutral_accel:
            alt = _compute_with_neutral_acceleration(
                harmonized_df, scheme_weights, settings, primary,
            )
        else:
            alt = compute_cci(harmonized_df, scheme_weights, settings)

        alt_scores = alt.scores["cci_score"]
        comparison = compare_rankings(primary_scores, alt_scores)

        results.append({
            "scheme": scheme_name,
            "spearman_r_vs_primary": comparison["spearman_r"],
            "max_rank_shift": comparison["max_rank_shift"],
            "n_shifted_gt_10": comparison["n_shifted_gt_10"],
            "median_cci_score": float(alt_scores.median()),
            "iqr_low": float(alt_scores.quantile(0.25)),
            "iqr_high": float(alt_scores.quantile(0.75)),
        })

        logger.info(
            "Alt weighting '%s': Spearman r=%.4f, median CCI=%.1f",
            scheme_name, comparison["spearman_r"], alt_scores.median(),
        )

    return pd.DataFrame(results)


def _build_schemes(
    comp_ids: list[str],
    base_weights: dict[str, float],
) -> list[tuple[str, dict[str, float], bool]]:
    """Build the 4 alternative weighting scheme definitions.

    Returns list of (name, weights_dict, neutral_acceleration_flag).
    """
    n = len(comp_ids)
    equal_weights = {c: 1.0 / n for c in comp_ids}

    return [
        ("equal", equal_weights, False),
        ("pure_budget", dict(base_weights), True),
        ("pure_risk", dict(PURE_RISK_WEIGHTS), False),
        ("neutral_acceleration", dict(base_weights), True),
    ]


def _compute_with_neutral_acceleration(
    harmonized_df: pd.DataFrame,
    weights: dict[str, float],
    settings,
    primary_output: "CCIOutput",
) -> "CCIOutput":
    """Recompute CCI with acceleration multipliers forced to 1.0.

    Runs the full pipeline but replaces acceleration values with 1.0.
    Uses the lightweight approach: extract centered from primary run,
    recompute Steps 8-10 with neutral acceleration.
    """
    from score.composite import calibrate_k, compute_component_scores

    # Extract centered data (post-Step 7) by re-running Steps 1-7
    from sensitivity.weight_perturbation import _extract_centered

    centered, comp_ids = _extract_centered(harmonized_df, settings)

    # Neutral acceleration: all 1.0
    neutral_accel = pd.DataFrame(
        1.0,
        index=centered.index,
        columns=[f"{c}_acceleration" for c in comp_ids],
    )

    # Steps 8-9
    comp_scores = compute_component_scores(
        centered, weights, primary_output.penalties, neutral_accel,
    )
    raw_composite = comp_scores.sum(axis=1)
    raw_composite.name = "raw_composite"

    # Step 10
    k = calibrate_k(raw_composite, target_iqr=settings.target_iqr)
    cci_scores = 100 + k * raw_composite
    cci_scores.name = "cci_score"

    scores_df = pd.DataFrame({
        "cci_score": cci_scores,
        "raw_composite": raw_composite,
    })
    scores_df.index.name = "fips"

    # Return a minimal CCIOutput for comparison purposes
    from score.pipeline import CCIOutput

    return CCIOutput(
        scores=scores_df,
        components=comp_scores,
        penalties=primary_output.penalties,
        accelerations=neutral_accel,
        corr_matrix=primary_output.corr_matrix,
        robustness_checks={},
        national=float(raw_composite.mean()),
        strain=pd.DataFrame({"cci_strain": pd.Series(dtype=float)}),
        dollar=pd.DataFrame({"cci_dollar": pd.Series(dtype=float)}),
        k=k,
        universe=centered.index,
    )
