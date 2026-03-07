"""Population-weighted national aggregate CCI-National.

CCI-National(t) = sum[ h(c) * S(c,t) ] / sum[ h(c) ]
where h(c) = housing units in county c, S(c) = raw composite score (not percentiled).

This is a non-percentile aggregate that tracks absolute national trend.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_national_aggregate(
    raw_composite: pd.Series,
    housing_units: pd.Series,
) -> float:
    """Compute housing-unit-weighted national CCI aggregate.

    Args:
        raw_composite: Raw composite S(c) per county, indexed by fips.
        housing_units: Housing units per county, indexed by fips.

    Returns:
        Single float: housing-unit-weighted average of S(c).
    """
    # Align on common fips
    common = raw_composite.index.intersection(housing_units.index)
    s = raw_composite.loc[common]
    h = housing_units.loc[common]

    # Drop NaN in either
    valid = s.notna() & h.notna() & (h > 0)
    s = s[valid]
    h = h[valid]

    if h.sum() == 0:
        logger.warning("No valid housing units; CCI-National = 0")
        return 0.0

    national = (h * s).sum() / h.sum()
    logger.info("CCI-National = %.4f (weighted over %d counties)", national, len(s))
    return float(national)
