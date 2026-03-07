"""Compute national percentile ranks (Step 3b of scoring pipeline).

Percentile ranks are computed over the scoring universe only.
All components use the SAME universe — never mix.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.components import COMPONENTS

logger = logging.getLogger(__name__)


def compute_percentiles(
    df: pd.DataFrame,
    universe: pd.Index,
    component_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute national percentile rank for each component across the scoring universe.

    Args:
        df: DataFrame with component columns, indexed by fips.
        universe: FIPS codes in the scoring universe.
        component_ids: Components to percentile-rank. Defaults to all in registry.

    Returns:
        DataFrame with same index as universe, component columns replaced by
        percentile ranks (0-100). NaN values remain NaN.
    """
    comp_ids = component_ids or [c for c in COMPONENTS if c in df.columns]

    # Restrict to scoring universe
    if "fips" in df.columns:
        scored = df.set_index("fips").loc[universe].copy()
    else:
        scored = df.loc[universe].copy()

    for comp_id in comp_ids:
        col = scored[comp_id]
        # rank(pct=True) gives fractional rank in [0, 1]; multiply by 100
        # NaN values are excluded from ranking and remain NaN
        scored[comp_id] = col.rank(pct=True, method="average") * 100

    logger.info(
        "Step 3b: Computed percentile ranks for %d components over %d counties",
        len(comp_ids), len(universe),
    )
    return scored
