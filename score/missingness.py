"""Handle missing component values (Step 7 of scoring pipeline).

Imputation rules:
- Missing Preferred Core components: impute to 0 (national median after centering)
  and downgrade confidence grade for that component.
- Missing Required Core components: county excluded from universe (handled in universe.py).
"""

from __future__ import annotations

import logging

import pandas as pd

from config.components import COMPONENTS
from score.universe import PREFERRED_CORE

logger = logging.getLogger(__name__)


def handle_missingness(
    centered: pd.DataFrame,
    imputation_rules: dict | None = None,
) -> pd.DataFrame:
    """Apply imputation rules for missing component values.

    Args:
        centered: Centered percentile DataFrame (scoring universe only).
        imputation_rules: Optional override. If None, uses default Preferred Core rules.

    Returns:
        Updated DataFrame with imputed values and confidence metadata updates.
    """
    result = centered.copy()
    preferred = imputation_rules or {c: 0.0 for c in PREFERRED_CORE}
    n_imputed = 0

    for comp_id, fill_value in preferred.items():
        if comp_id not in result.columns:
            continue
        mask = result[comp_id].isna()
        count = mask.sum()
        if count > 0:
            result.loc[mask, comp_id] = fill_value
            n_imputed += count

            # Downgrade confidence for imputed counties
            conf_col = f"{comp_id}__confidence"
            if conf_col in result.columns:
                result.loc[mask, conf_col] = _downgrade_confidence(
                    result.loc[mask, conf_col]
                )

    # Special case: stale flood maps
    flood_flag_col = "flood_exposure__map_currency_flag"
    flood_conf_col = "flood_exposure__confidence"
    if flood_flag_col in result.columns and flood_conf_col in result.columns:
        stale_mask = result[flood_flag_col] == 1
        n_stale = stale_mask.sum()
        if n_stale > 0:
            result.loc[stale_mask, flood_conf_col] = "B"
            logger.info("Downgraded flood_exposure confidence for %d stale-map counties", n_stale)

    # Special case: storm severity reliability flag — log only, no downgrade
    storm_flag_col = "storm_severity__severity_reliability_flag"
    if storm_flag_col in result.columns:
        n_flagged = (result[storm_flag_col] == 1).sum()
        if n_flagged > 0:
            logger.info(
                "storm_severity reliability flag: %d counties flagged (informational only)",
                n_flagged,
            )

    logger.info("Step 7: Imputed %d missing Preferred Core values to 0", n_imputed)
    return result


def _downgrade_confidence(series: pd.Series) -> pd.Series:
    """Downgrade confidence grades: A→B, B→C, C stays C."""
    mapping = {"A": "B", "B": "C", "C": "C"}
    return series.map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
