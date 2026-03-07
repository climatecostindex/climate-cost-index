"""Correlation matrix, precedence hierarchy, and overlap penalties (Step 5).

Penalties are computed BEFORE acceleration multipliers. Penalties reflect
structural redundancy (stable). Acceleration reflects trend dynamics (volatile).

Precedence hierarchy:
  Tier 1: Direct dollar-cost attributed (energy)
  Tier 2: Hazard burden proxies
  Tier 3: General exposure indicators
  Within tier: higher confidence > lower, then larger CE weight, then lower error

Penalty formula: d = (1 - r^2) for lower-precedence component
Cumulative floor: d_min = 0.2 (every component contributes >= 20%)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from config.components import COMPONENTS, ComponentDef

logger = logging.getLogger(__name__)

# Confidence rank for within-tier tiebreaking
_CONFIDENCE_RANK = {"A": 3, "B": 2, "C": 1}

# Census Bureau 4-region mapping by state FIPS prefix
CENSUS_REGIONS = {
    # Northeast
    "09": 1, "23": 1, "25": 1, "33": 1, "34": 1, "36": 1, "42": 1, "44": 1, "50": 1,
    # Midwest
    "17": 2, "18": 2, "19": 2, "20": 2, "26": 2, "27": 2, "29": 2, "31": 2,
    "38": 2, "39": 2, "46": 2, "55": 2,
    # South
    "01": 3, "05": 3, "10": 3, "11": 3, "12": 3, "13": 3, "21": 3, "22": 3,
    "24": 3, "28": 3, "37": 3, "40": 3, "45": 3, "47": 3, "48": 3, "51": 3, "54": 3,
    # West
    "02": 4, "04": 4, "06": 4, "08": 4, "15": 4, "16": 4, "30": 4, "32": 4,
    "35": 4, "41": 4, "49": 4, "53": 4, "56": 4,
}


def _component_sort_key(comp: ComponentDef) -> tuple:
    """Sort key for precedence: tier ASC, confidence DESC, weight DESC."""
    return (
        comp.precedence_tier.value,
        -_CONFIDENCE_RANK.get(comp.confidence, 0),
        -comp.base_weight,
    )


def _extract_component_data(
    df: pd.DataFrame,
    universe: pd.Index,
    component_ids: list[str],
) -> pd.DataFrame:
    """Extract component columns for scoring universe counties.

    Returns DataFrame indexed by fips with component columns only.
    """
    if "fips" in df.columns:
        data = df.set_index("fips").loc[universe, component_ids]
    elif df.index.isin(universe).any():
        data = df.loc[df.index.isin(universe), component_ids]
    else:
        data = df[component_ids]
    return data


def compute_correlation_matrix(
    centered: pd.DataFrame,
    universe: pd.Index,
    component_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Pearson correlation matrix across all components.

    Args:
        centered: Centered percentile DataFrame (scoring universe).
        universe: FIPS in scoring universe.
        component_ids: Components to include. Defaults to all in registry.

    Returns:
        Correlation matrix DataFrame (components x components).
    """
    comp_ids = component_ids or [c for c in COMPONENTS if c in centered.columns]

    if len(universe) < 2:
        logger.warning("Fewer than 2 counties in universe; correlation matrix is degenerate")
        return pd.DataFrame(np.nan, index=comp_ids, columns=comp_ids)

    data = _extract_component_data(centered, universe, comp_ids)
    corr = data.corr(method="pearson")
    logger.info("Step 5a: Computed %dx%d Pearson correlation matrix", len(comp_ids), len(comp_ids))
    return corr


def compute_overlap_penalties(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.6,
    precedence: dict | None = None,
    floor: float = 0.2,
    robustness_results: dict | None = None,
) -> tuple[dict[str, float], dict]:
    """Compute penalty factors for each component based on overlap.

    When robustness_results are provided, applies conservative flagging:
    if ANY method (Pearson, Spearman, distance correlation, partial correlation)
    flags a pair, it is flagged. Per SSRN Section 7.1.

    Args:
        corr_matrix: Pearson correlation matrix.
        threshold: Correlation threshold for flagging pairs (default 0.6).
        precedence: Optional override {component_id: sort_key}. If None, uses registry.
        floor: Minimum penalty factor (default 0.2).
        robustness_results: Output from compute_correlation_robustness.
            If provided, enables conservative flagging.

    Returns:
        Tuple of:
        - {component_id: penalty_factor} where penalty_factor in [floor, 1.0].
        - Documentation dict with flagged_pairs, precedence_decisions, raw_penalties.
    """
    comp_ids = list(corr_matrix.columns)
    penalties = {c: 1.0 for c in comp_ids}

    # Handle degenerate cases
    if len(comp_ids) < 2 or corr_matrix.isna().all().all():
        docs = {
            "flagged_pairs": [],
            "precedence_decisions": [],
            "penalties": dict(penalties),
        }
        logger.info("Step 5b: 0 flagged pairs (degenerate or empty input)")
        return penalties, docs

    flagged_pairs = []
    precedence_decisions = []

    # Sort components by precedence (higher precedence first)
    sorted_comps = sorted(
        [COMPONENTS[c] for c in comp_ids if c in COMPONENTS],
        key=_component_sort_key,
    )
    comp_order = {c.id: i for i, c in enumerate(sorted_comps)}

    # Build set of pairs flagged by alternative methods for conservative flagging
    alt_flagged: set[tuple[str, str]] = set()
    if robustness_results:
        for method_key in ("spearman_corr", "distance_corr", "partial_corr"):
            alt_corr = robustness_results.get(method_key)
            if alt_corr is None or not isinstance(alt_corr, pd.DataFrame):
                continue
            for i, c1 in enumerate(comp_ids):
                for c2 in comp_ids[i + 1:]:
                    if c1 in alt_corr.index and c2 in alt_corr.columns:
                        alt_r = alt_corr.loc[c1, c2]
                        if not np.isnan(alt_r) and abs(alt_r) > threshold:
                            pair = tuple(sorted([c1, c2]))
                            alt_flagged.add(pair)

    # Find flagged pairs (Pearson + conservative adoption from robustness)
    for i, c1 in enumerate(comp_ids):
        for c2 in comp_ids[i + 1:]:
            r = corr_matrix.loc[c1, c2]
            if np.isnan(r):
                continue
            pair = tuple(sorted([c1, c2]))
            pearson_flags = abs(r) > threshold
            alt_flags = pair in alt_flagged
            if pearson_flags or alt_flags:
                flagged_pairs.append((c1, c2, float(r)))

    # Apply penalties: lower-precedence component in each flagged pair
    for c1, c2, r in flagged_pairs:
        order1 = comp_order.get(c1, 999)
        order2 = comp_order.get(c2, 999)

        if order1 <= order2:
            winner, loser = c1, c2
        else:
            winner, loser = c2, c1

        raw_penalty = 1.0 - r ** 2
        precedence_decisions.append({
            "winner": winner,
            "loser": loser,
            "r": r,
            "raw_penalty": raw_penalty,
        })

        # Cumulative: multiply penalties for components penalized multiple times
        penalties[loser] *= raw_penalty

    # Apply floor
    for c in penalties:
        penalties[c] = max(penalties[c], floor)

    docs = {
        "flagged_pairs": flagged_pairs,
        "precedence_decisions": precedence_decisions,
        "penalties": dict(penalties),
    }

    n_flagged = len(flagged_pairs)
    n_penalized = sum(1 for v in penalties.values() if v < 1.0)
    logger.info(
        "Step 5b: %d flagged pairs (|r|>%.2f), %d components penalized",
        n_flagged, threshold, n_penalized,
    )
    return penalties, docs


def compute_correlation_robustness(
    centered: pd.DataFrame,
    universe: pd.Index,
    component_ids: list[str] | None = None,
    threshold: float = 0.6,
) -> dict:
    """Compute Spearman, distance, and partial correlation robustness checks.

    Also detects discrepancies between methods and documents them.

    Args:
        centered: Centered percentile DataFrame.
        universe: FIPS in scoring universe.
        component_ids: Components to check. Defaults to all in registry.
        threshold: Correlation threshold for flagging comparison (default 0.6).

    Returns:
        Dict with spearman_corr, distance_corr (if dcor available),
        partial_corr, discrepancies, and pearson_corr for comparison.
    """
    comp_ids = component_ids or [c for c in COMPONENTS if c in centered.columns]
    result: dict = {}

    if len(universe) < 2:
        logger.warning("Fewer than 2 counties; skipping robustness checks")
        result["spearman_corr"] = pd.DataFrame(np.nan, index=comp_ids, columns=comp_ids)
        result["distance_corr"] = None
        result["partial_corr"] = None
        result["discrepancies"] = []
        return result

    data = _extract_component_data(centered, universe, comp_ids)

    # Pearson (for comparison with alternatives)
    pearson = data.corr(method="pearson")
    result["pearson_corr"] = pearson

    # Spearman
    spearman = data.corr(method="spearman")
    result["spearman_corr"] = spearman

    # Distance correlation (optional)
    try:
        import warnings

        import dcor

        dcor_matrix = pd.DataFrame(
            np.zeros((len(comp_ids), len(comp_ids))),
            index=comp_ids, columns=comp_ids,
        )
        clean = data.dropna()
        if len(clean) >= 2:
            for i, c1 in enumerate(comp_ids):
                for c2 in comp_ids[i:]:
                    if c1 == c2:
                        dcor_matrix.loc[c1, c2] = 1.0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            d = dcor.distance_correlation(
                                clean[c1].values, clean[c2].values
                            )
                        dcor_matrix.loc[c1, c2] = d
                        dcor_matrix.loc[c2, c1] = d
        result["distance_corr"] = dcor_matrix
    except ImportError:
        logger.warning("dcor package not available; skipping distance correlation")
        result["distance_corr"] = None

    # Partial correlation controlling for Census region
    result["partial_corr"] = _compute_partial_correlation(data, comp_ids)

    # Discrepancy detection
    result["discrepancies"] = _compute_discrepancies(
        pearson, spearman, result["distance_corr"], result["partial_corr"],
        comp_ids, threshold,
    )

    n_disc = len(result["discrepancies"])
    if n_disc > 0:
        logger.info("Step 5c: %d flagging discrepancies between correlation methods", n_disc)
    logger.info("Step 5c: Robustness checks computed (Spearman + distance + partial correlation)")
    return result


def _compute_partial_correlation(
    data: pd.DataFrame,
    comp_ids: list[str],
) -> pd.DataFrame | None:
    """Compute partial correlation controlling for Census region.

    Regresses out Census region indicator variables from each component,
    then computes Pearson correlation on residuals.
    """
    # Derive Census region from index (fips codes)
    fips_index = data.index.astype(str).str.zfill(5)
    state_fips = fips_index.str[:2]
    regions = state_fips.map(CENSUS_REGIONS)

    # Drop rows where region is unknown
    valid_mask = regions.notna()
    if valid_mask.sum() < 3:
        logger.warning("Too few counties with known Census region for partial correlation")
        return None

    data_valid = data.loc[valid_mask].copy()
    regions_valid = regions[valid_mask].astype(int)

    # Create region dummies (drop one for full rank)
    region_dummies = pd.get_dummies(regions_valid, prefix="region", drop_first=True, dtype=float)
    region_dummies.index = data_valid.index

    # Regress out region from each component and collect residuals
    residuals = pd.DataFrame(index=data_valid.index, columns=comp_ids, dtype=float)
    for comp in comp_ids:
        y = data_valid[comp].values
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            residuals[comp] = np.nan
            continue
        X = region_dummies.values[valid]
        y_valid = y[valid]
        try:
            beta = np.linalg.lstsq(X, y_valid, rcond=None)[0]
            predicted = X @ beta
            resid = np.full(len(y), np.nan)
            resid[valid] = y_valid - predicted
            residuals[comp] = resid
        except np.linalg.LinAlgError:
            residuals[comp] = np.nan

    partial_corr = residuals.astype(float).corr(method="pearson")
    return partial_corr


def _compute_discrepancies(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    distance: pd.DataFrame | None,
    partial: pd.DataFrame | None,
    comp_ids: list[str],
    threshold: float,
) -> list[dict]:
    """Detect flagging discrepancies between correlation methods.

    A discrepancy exists when one method flags a pair (|r| > threshold)
    but another does not.

    Returns list of dicts documenting each discrepancy.
    """
    discrepancies = []

    for i, c1 in enumerate(comp_ids):
        for c2 in comp_ids[i + 1:]:
            methods = {}

            # Pearson
            r_p = pearson.loc[c1, c2] if c1 in pearson.index and c2 in pearson.columns else np.nan
            if not np.isnan(r_p):
                methods["pearson"] = (float(r_p), abs(r_p) > threshold)

            # Spearman
            r_s = spearman.loc[c1, c2] if c1 in spearman.index and c2 in spearman.columns else np.nan
            if not np.isnan(r_s):
                methods["spearman"] = (float(r_s), abs(r_s) > threshold)

            # Distance correlation
            if distance is not None and c1 in distance.index and c2 in distance.columns:
                r_d = distance.loc[c1, c2]
                if not np.isnan(r_d):
                    methods["distance"] = (float(r_d), abs(r_d) > threshold)

            # Partial correlation
            if partial is not None and c1 in partial.index and c2 in partial.columns:
                r_pc = partial.loc[c1, c2]
                if not np.isnan(r_pc):
                    methods["partial"] = (float(r_pc), abs(r_pc) > threshold)

            if len(methods) < 2:
                continue

            flags = {m: flagged for m, (_, flagged) in methods.items()}
            if len(set(flags.values())) > 1:
                # Discrepancy: methods disagree
                disc = {
                    "pair": (c1, c2),
                    "methods": {m: {"r": r, "flagged": f} for m, (r, f) in methods.items()},
                    "conservative_flag": any(f for f in flags.values()),
                }
                discrepancies.append(disc)
                logger.debug(
                    "Discrepancy for %s↔%s: %s",
                    c1, c2,
                    {m: f"r={r:.3f} {'FLAGGED' if fl else 'not flagged'}"
                     for m, (r, fl) in methods.items()},
                )

    return discrepancies
