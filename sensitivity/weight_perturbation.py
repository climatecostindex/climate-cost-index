"""Monte Carlo weight sensitivity: 10k iterations, weights +/-30%."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.components import get_weights
from config.settings import Settings, get_settings
from score.composite import calibrate_k, compute_component_scores
from score.pipeline import CCIOutput, compute_cci

from .ranking_utils import compare_rankings

logger = logging.getLogger(__name__)


def run_weight_perturbation(
    harmonized_df: pd.DataFrame,
    base_weights: dict[str, float],
    n_iterations: int = 10_000,
    perturbation_pct: float = 0.30,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate random weight vectors, recompute CCI, report stability.

    Uses a lightweight shortcut: runs the full pipeline once to get
    centered percentiles, penalties, and accelerations (Steps 1-7),
    then only recomputes Steps 8-10 for each Monte Carlo iteration.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame.
        base_weights: {component_id: normalized_weight} from get_weights().
        n_iterations: Number of Monte Carlo iterations.
        perturbation_pct: Max fraction to perturb each weight (0.30 = ±30%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (detail_df, summary_dict).
        detail_df columns: fips, primary_rank, mean_rank, rank_std,
            rank_p05, rank_p95, max_rank_shift, is_flagged.
        summary_dict: Spearman correlation distribution stats.
    """
    settings = get_settings()
    rng = np.random.default_rng(seed)

    # ── Run full pipeline once for baseline ──
    primary = compute_cci(harmonized_df, base_weights, settings)
    primary_scores = primary.scores["cci_score"]
    primary_ranks = primary_scores.rank(method="min")
    n_counties = len(primary_scores)

    # ── Extract intermediate results for lightweight shortcut ──
    # We need: centered (post-Step 7), penalties, accelerations, k
    # Re-run Steps 1-7 to get centered values (they're embedded in pipeline)
    # Actually, we can reconstruct centered from component_scores:
    #   component_scores[c] = centered[c] * weight[c] * penalty[c] * accel[c]
    # So: centered[c] = component_scores[c] / (weight[c] * penalty[c] * accel[c])
    # But it's simpler to re-extract from the pipeline internals.

    # Re-derive centered values from the pipeline
    centered, comp_ids = _extract_centered(harmonized_df, settings)
    penalties = primary.penalties
    accelerations = primary.accelerations

    # ── Monte Carlo iterations ──
    # Pre-allocate rank matrix: (n_iterations, n_counties)
    rank_matrix = np.empty((n_iterations, n_counties), dtype=np.float32)
    spearman_rs = np.empty(n_iterations, dtype=np.float64)

    fips_index = centered.index
    comp_ids_in_weights = [c for c in base_weights if c in centered.columns]

    for i in range(n_iterations):
        # Perturb weights
        perturbed = _perturb_weights(base_weights, perturbation_pct, rng)

        # Steps 8-9: recompute component scores and composite
        comp_scores = compute_component_scores(
            centered, perturbed, penalties, accelerations,
        )
        raw_composite = comp_scores.sum(axis=1)

        # Step 10: scale (use primary k for consistency across iterations)
        cci = 100 + primary.k * raw_composite
        ranks = cci.rank(method="min").values
        rank_matrix[i] = ranks

        # Spearman vs primary
        sp_r, _ = _fast_spearman(primary_ranks.values, ranks)
        spearman_rs[i] = sp_r

    # ── Aggregate results ──
    primary_rank_vals = primary_ranks.values

    mean_ranks = rank_matrix.mean(axis=0)
    std_ranks = rank_matrix.std(axis=0)
    p05_ranks = np.percentile(rank_matrix, 5, axis=0)
    p95_ranks = np.percentile(rank_matrix, 95, axis=0)
    min_ranks = rank_matrix.min(axis=0)
    max_ranks = rank_matrix.max(axis=0)
    max_shift = max_ranks - min_ranks

    # Flag: shift > 15 percentile points (as fraction of n_counties)
    flag_threshold = 15 * n_counties / 100
    is_flagged = max_shift > flag_threshold

    detail_df = pd.DataFrame({
        "fips": fips_index,
        "primary_rank": primary_rank_vals.astype(int),
        "mean_rank": mean_ranks,
        "rank_std": std_ranks,
        "rank_p05": p05_ranks,
        "rank_p95": p95_ranks,
        "max_rank_shift": max_shift.astype(int),
        "is_flagged": is_flagged,
    })

    summary = {
        "n_iterations": n_iterations,
        "perturbation_pct": perturbation_pct,
        "seed": seed,
        "spearman_distribution": {
            "p05": float(np.percentile(spearman_rs, 5)),
            "p25": float(np.percentile(spearman_rs, 25)),
            "p50": float(np.percentile(spearman_rs, 50)),
            "p75": float(np.percentile(spearman_rs, 75)),
            "p95": float(np.percentile(spearman_rs, 95)),
            "mean": float(np.mean(spearman_rs)),
        },
        "n_flagged": int(is_flagged.sum()),
    }

    logger.info(
        "Weight perturbation complete: %d iterations, median Spearman=%.4f, "
        "%d counties flagged",
        n_iterations, summary["spearman_distribution"]["p50"],
        summary["n_flagged"],
    )

    return detail_df, summary


def _perturb_weights(
    base_weights: dict[str, float],
    perturbation_pct: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Perturb each weight by uniform random factor, re-normalize to sum=1."""
    factors = {
        comp: rng.uniform(1 - perturbation_pct, 1 + perturbation_pct)
        for comp in base_weights
    }
    perturbed = {comp: base_weights[comp] * factors[comp] for comp in base_weights}
    total = sum(perturbed.values())
    return {comp: v / total for comp, v in perturbed.items()}


def _extract_centered(
    harmonized_df: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, list[str]]:
    """Re-run Steps 1-7 of the pipeline to extract centered percentiles.

    Returns centered DataFrame (post-missingness) and list of component IDs.
    """
    from config.components import COMPONENTS
    from score.center import center
    from score.missingness import handle_missingness
    from score.percentile import compute_percentiles
    from score.transform_inputs import transform_inputs
    from score.universe import define_universe
    from score.winsorize import winsorize

    component_ids = [c for c in COMPONENTS if c in harmonized_df.columns]
    scoring_year = settings.scoring_year

    scoring_year_df = harmonized_df[harmonized_df["year"] == scoring_year].copy()
    if "fips" in scoring_year_df.columns:
        scoring_year_df = scoring_year_df.set_index("fips")

    transformed = transform_inputs(scoring_year_df)
    winsorized = winsorize(transformed, percentile=settings.winsorize_percentile)
    universe = define_universe(winsorized)
    percentiled = compute_percentiles(winsorized, universe, component_ids=component_ids)
    centered = center(percentiled, component_ids=component_ids)
    centered = handle_missingness(centered)

    return centered, component_ids


def _fast_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fast Spearman correlation without scipy overhead per iteration."""
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)
