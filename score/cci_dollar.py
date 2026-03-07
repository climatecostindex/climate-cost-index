"""Energy-only dollar adder: CCI-Dollar.

CCI-Dollar is the climate-attributed energy cost in actual dollars
per household per year. This is the only fully attributed component in v1.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_dollar(
    energy_attributed_costs: pd.Series,
) -> pd.DataFrame:
    """Extract dollar-denominated climate-attributed energy cost per county.

    Args:
        energy_attributed_costs: Raw energy_cost_attributed values ($/household/year),
            indexed by fips. NOT transformed or percentiled.

    Returns:
        DataFrame with fips index and 'cci_dollar' column.
    """
    result = pd.DataFrame({"cci_dollar": energy_attributed_costs})
    result.index.name = "fips"

    valid = result["cci_dollar"].notna()
    logger.info(
        "CCI-Dollar: %d counties, mean=$%.2f, median=$%.2f",
        valid.sum(),
        result.loc[valid, "cci_dollar"].mean(),
        result.loc[valid, "cci_dollar"].median(),
    )
    return result
