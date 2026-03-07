"""Weighted sum -> S(c) -> CCI(c) (Steps 8-10 of scoring pipeline).

Step 8: component_score(i,c) = centered(i,c) * weight(i) * penalty(i) * acceleration(i,c)
Step 9: S(c) = sum over components of component_score(i,c)
Step 10: CCI(c) = 100 + k * S(c)
  where k is calibrated ONCE at v1 launch to target IQR of (80, 120)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_component_scores(
    centered: pd.DataFrame,
    weights: dict[str, float],
    penalties: dict[str, float],
    accelerations: pd.DataFrame,
) -> pd.DataFrame:
    """Compute weighted, penalized, accelerated component scores.

    Args:
        centered: Centered percentile DataFrame (post-missingness), indexed by fips.
        weights: {component_id: normalized_weight}.
        penalties: {component_id: penalty_factor} from overlap step.
        accelerations: DataFrame with '{comp}_acceleration' columns, indexed by fips.

    Returns:
        DataFrame with fips index and per-component score columns.
    """
    comp_ids = [c for c in weights if c in centered.columns]
    result = pd.DataFrame(index=centered.index)

    for comp_id in comp_ids:
        w = weights[comp_id]
        p = penalties.get(comp_id, 1.0)
        accel_col = f"{comp_id}_acceleration"

        if accel_col in accelerations.columns:
            a = accelerations[accel_col].reindex(centered.index, fill_value=1.0)
        else:
            a = 1.0

        result[comp_id] = centered[comp_id] * w * p * a

    logger.info("Step 8: Computed %d component scores", len(comp_ids))
    return result


def calibrate_k(
    raw_composite: pd.Series,
    target_iqr: tuple[float, float] = (80.0, 120.0),
) -> float:
    """Calibrate scaling constant k so CCI-Score IQR spans target range.

    Set ONCE at v1 launch. Fixed permanently until methodology version change.

    Args:
        raw_composite: Raw composite scores S(c).
        target_iqr: Target (lower, upper) of CCI-Score IQR.

    Returns:
        Scaling constant k.
    """
    target_span = target_iqr[1] - target_iqr[0]  # 40
    q1 = raw_composite.quantile(0.25)
    q3 = raw_composite.quantile(0.75)
    iqr_s = q3 - q1

    if iqr_s < 1e-10:
        logger.warning("Near-zero IQR (%.6f); defaulting k=1.0 to avoid explosion", iqr_s)
        return 1.0

    k = target_span / iqr_s
    logger.info(
        "Step 10: Calibrated k=%.4f (IQR(S)=%.4f, target span=%.0f)",
        k, iqr_s, target_span,
    )
    return k
