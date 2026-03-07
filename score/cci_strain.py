"""Income-adjusted CCI variant: CCI-Strain.

CCI-Strain(c) = CCI-Score(c) / median_household_income(c), re-indexed.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_strain(
    cci_scores: pd.Series,
    median_income: pd.Series,
) -> pd.DataFrame:
    """Compute income-adjusted CCI-Strain for each county.

    Args:
        cci_scores: CCI-Score per county, indexed by fips.
        median_income: Median household income per county, indexed by fips.

    Returns:
        DataFrame with fips index and 'cci_strain' column, re-indexed so
        national median CCI-Strain = 100.
    """
    common = cci_scores.index.intersection(median_income.index)
    scores = cci_scores.loc[common]
    income = median_income.loc[common]

    # Avoid division by zero/NaN
    valid = income.notna() & (income > 0)
    raw_strain = scores[valid] / income[valid]

    # Re-index so median = 100
    strain_median = raw_strain.median()
    if strain_median > 0:
        cci_strain = (raw_strain / strain_median) * 100
    else:
        cci_strain = raw_strain * 0 + 100  # degenerate case

    result = pd.DataFrame({"cci_strain": cci_strain})
    result.index.name = "fips"

    logger.info(
        "CCI-Strain: %d counties, median=%.1f",
        len(result), result["cci_strain"].median(),
    )
    return result
