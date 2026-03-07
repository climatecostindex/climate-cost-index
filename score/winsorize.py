"""99th percentile winsorization (Step 2 of scoring pipeline).

Compresses extreme values to reduce the influence of outliers
while preserving rank order.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.components import COMPONENTS

logger = logging.getLogger(__name__)


def winsorize(
    df: pd.DataFrame,
    percentile: float = 99.0,
    component_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Winsorize all component columns at the given upper percentile.

    Args:
        df: DataFrame with component columns.
        percentile: Upper percentile for clamping (default 99).
        component_ids: Components to winsorize. If None, uses all from registry.

    Returns:
        Copy of DataFrame with extreme upper-tail values clamped.
    """
    result = df.copy()
    comp_ids = component_ids or [c for c in COMPONENTS if c in df.columns]

    for comp_id in comp_ids:
        values = result[comp_id].dropna().values
        if len(values) == 0:
            continue
        p = np.percentile(values, percentile)
        n_clamped = (result[comp_id] > p).sum()
        result[comp_id] = result[comp_id].clip(upper=p)
        if n_clamped > 0:
            logger.debug(
                "Winsorized %s: %d values clamped to %.4f (p%g)",
                comp_id, n_clamped, p, percentile,
            )

    logger.info("Step 2: Winsorized %d components at p%.0f", len(comp_ids), percentile)
    return result
