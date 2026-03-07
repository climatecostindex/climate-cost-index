"""Define the scoring universe based on completeness checks.

A county is included in the scoring universe only if it has data for all
Required Core components. If excluded, it is excluded from ALL component
percentile computations. Never mix universes.

Output: pandas Index of FIPS codes in the scoring universe.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Required Core: county MUST have these to be scored.
# A county needs at least ONE of hdd_anomaly/cdd_anomaly + drought_score + storm_severity.
REQUIRED_CORE_DEGREE_DAY = {"hdd_anomaly", "cdd_anomaly"}
REQUIRED_CORE_OTHER = {"drought_score", "storm_severity"}
REQUIRED_CORE = REQUIRED_CORE_DEGREE_DAY | REQUIRED_CORE_OTHER

# All components not in Required Core are Preferred Core (imputed to 0 if missing).
PREFERRED_CORE = {
    "extreme_heat_days", "pm25_annual", "aqi_unhealthy_days",
    "flood_exposure", "wildfire_score", "energy_cost_attributed",
    "health_burden", "fema_ia_burden",
}


def define_universe(
    harmonized_df: pd.DataFrame,
    completeness_rules: dict | None = None,
) -> pd.Index:
    """Return FIPS codes of counties meeting completeness requirements.

    Args:
        harmonized_df: Scoring-year DataFrame indexed or containing 'fips' column.
        completeness_rules: Optional override. If None, uses default Required Core rules.

    Returns:
        pd.Index of FIPS codes in the scoring universe.
    """
    df = harmonized_df
    if "fips" in df.columns:
        df = df.set_index("fips")

    # At least one degree-day signal
    has_degree_day = df[list(REQUIRED_CORE_DEGREE_DAY)].notna().any(axis=1)

    # All other required core components
    has_other = df[list(REQUIRED_CORE_OTHER)].notna().all(axis=1)

    in_universe = has_degree_day & has_other
    universe = df.index[in_universe]

    logger.info(
        "Step 3a: Scoring universe = %d counties (excluded %d for missing Required Core)",
        len(universe),
        len(df) - len(universe),
    )
    return universe
