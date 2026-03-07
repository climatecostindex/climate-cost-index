"""Log-transform heavy-tailed variables (Step 1 of scoring pipeline).

Applies per-component transforms as defined in config/components.py:
- "log": np.log1p(x) for heavy-tailed distributions
- "sqrt": np.sqrt(x) for moderate skew
- "identity": no transform
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.components import COMPONENTS

logger = logging.getLogger(__name__)


def transform_inputs(
    harmonized_df: pd.DataFrame,
    transform_rules: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply variable-specific transforms to reduce skew.

    Args:
        harmonized_df: Scoring-year DataFrame with component columns.
        transform_rules: Optional override {component_id: "log"|"sqrt"|"identity"}.
            If None, uses ComponentDef.transform from config/components.py.

    Returns:
        DataFrame with component columns transformed in-place (copy returned).
    """
    df = harmonized_df.copy()
    rules = transform_rules or {c.id: c.transform for c in COMPONENTS.values()}
    component_ids = [c for c in rules if c in df.columns]

    for comp_id in component_ids:
        rule = rules[comp_id]
        col = df[comp_id]

        if rule == "identity":
            continue
        elif rule == "log":
            # log1p requires x >= 0; negative values pass through as-is
            mask_nonneg = col >= 0
            df.loc[mask_nonneg, comp_id] = np.log1p(col[mask_nonneg])
            logger.debug("log1p transform applied to %s", comp_id)
        elif rule == "sqrt":
            # signed sqrt: sign(x) * sqrt(|x|)
            df[comp_id] = np.sign(col) * np.sqrt(np.abs(col))
            logger.debug("sqrt transform applied to %s", comp_id)
        else:
            logger.warning("Unknown transform %r for %s, skipping", rule, comp_id)

    logger.info(
        "Step 1: Transformed %d components (log=%d, sqrt=%d, identity=%d)",
        len(component_ids),
        sum(1 for c in component_ids if rules[c] == "log"),
        sum(1 for c in component_ids if rules[c] == "sqrt"),
        sum(1 for c in component_ids if rules[c] == "identity"),
    )
    return df
