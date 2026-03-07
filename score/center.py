"""Subtract 50 to center percentile ranks (Step 4 of scoring pipeline).

After centering, a county at the national median for a component
contributes 0 to the composite score for that component.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.components import COMPONENTS

logger = logging.getLogger(__name__)


def center(
    percentiled: pd.DataFrame,
    subtract: float = 50.0,
    component_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Center percentile ranks by subtracting the given value.

    Args:
        percentiled: DataFrame with percentile-ranked component columns.
        subtract: Value to subtract (default 50).
        component_ids: Components to center. Defaults to all in registry.

    Returns:
        Copy with centered component columns. Range: [-50, +50].
    """
    result = percentiled.copy()
    comp_ids = component_ids or [c for c in COMPONENTS if c in result.columns]

    for comp_id in comp_ids:
        result[comp_id] = result[comp_id] - subtract

    logger.info("Step 4: Centered %d components (subtracted %.0f)", len(comp_ids), subtract)
    return result
