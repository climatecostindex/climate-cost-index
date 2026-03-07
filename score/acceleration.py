"""Theil-Sen acceleration multipliers (Step 6 of scoring pipeline).

For each component i, county c:
  - Compute Theil-Sen slope over trailing window (5yr continuous, 10yr events)
  - median_slope(i) = national median across counties
  - sigma(i) = national std dev

Denominator protection:
  epsilon = 0.1 * sigma(i)
  if |median_slope(i)| >= epsilon:
      a(i,c) = slope(i,c) / median_slope(i)
  else:
      a(i,c) = 1 + (slope(i,c) - median_slope(i)) / sigma(i)

Bounds: a(i,c) in [0.5, 3.0]
Min completeness: 80% of years in window, else a = 1.0

Design note: Slopes are computed on untransformed harmonized values (pre-Step 1).
The SSRN paper (Section 6.3) references "transformed raw values," but the pipeline
passes the full multi-year harmonized_df before log/sqrt transforms. Steps 1-5
operate on scoring-year-only data and are not available for historical years.
This is an intentional design decision: acceleration captures the raw trend in the
underlying hazard metric, not the trend in the transformed/percentiled score.

Completeness threshold: The spec says "48/60 months for 5-year" but harmonized data
is at annual grain (county-year). The annual analog is ceil(window * 0.8) years,
i.e. 4 of 5 years for a 5-year window.

Uses: scipy.stats.theilslopes
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from config.components import COMPONENTS

logger = logging.getLogger(__name__)


def compute_theil_sen_slopes(
    harmonized_df: pd.DataFrame,
    scoring_year: int,
    component_ids: list[str] | None = None,
    min_completeness: float = 0.8,
) -> pd.DataFrame:
    """Compute per-component per-county Theil-Sen slopes.

    Args:
        harmonized_df: Full multi-year harmonized DataFrame with 'fips' and 'year'.
        scoring_year: The target scoring year.
        component_ids: Components to compute slopes for. Defaults to all in registry.
        min_completeness: Minimum fraction of years with data required (default 0.8).

    Returns:
        DataFrame indexed by fips with columns '{component_id}_slope' for each component.
    """
    comp_ids = component_ids or [c for c in COMPONENTS if c in harmonized_df.columns]
    df = harmonized_df.copy()

    if "fips" not in df.columns:
        df = df.reset_index()

    all_fips = df["fips"].unique()
    slopes_data = {f"{c}_slope": pd.Series(np.nan, index=all_fips) for c in comp_ids}

    for comp_id in comp_ids:
        comp_def = COMPONENTS.get(comp_id)
        if comp_def is None:
            continue

        window = comp_def.acceleration_window
        start_year = scoring_year - window + 1

        # Filter to window
        window_df = df[(df["year"] >= start_year) & (df["year"] <= scoring_year)]

        # Check if component has enough years of data globally
        years_with_data = window_df.loc[window_df[comp_id].notna(), "year"].nunique()
        if years_with_data < 2:
            # Static component (e.g. flood, wildfire with only 1 year)
            slopes_data[f"{comp_id}_slope"][:] = 0.0
            logger.info(
                "Component %s is static (%d year(s) of data) — "
                "assigning neutral acceleration 1.0 for all counties",
                comp_id, years_with_data,
            )
            continue

        min_years = max(2, int(np.ceil(window * min_completeness)))

        # Group by fips and compute slope
        for fips, group in window_df.groupby("fips"):
            series = group[["year", comp_id]].dropna(subset=[comp_id])
            if len(series) < min_years:
                slopes_data[f"{comp_id}_slope"][fips] = np.nan  # insufficient data
                continue

            y = series[comp_id].values.astype(float)
            x = series["year"].values.astype(float)

            try:
                slope, _, _, _ = theilslopes(y, x)
                slopes_data[f"{comp_id}_slope"][fips] = slope
            except Exception:
                slopes_data[f"{comp_id}_slope"][fips] = np.nan

    result = pd.DataFrame(slopes_data, index=all_fips)
    result.index.name = "fips"

    # Per-component slope summary
    for comp_id in comp_ids:
        slope_col = f"{comp_id}_slope"
        if slope_col in result.columns:
            valid = result[slope_col].dropna()
            n_nan = result[slope_col].isna().sum()
            logger.info(
                "Step 6a [%s]: %d valid slopes, %d NaN (insufficient data), "
                "median=%.4f, std=%.4f",
                comp_id, len(valid), n_nan,
                valid.median() if len(valid) > 0 else 0.0,
                valid.std() if len(valid) > 0 else 0.0,
            )

    logger.info("Step 6a: Computed Theil-Sen slopes for %d components", len(comp_ids))
    return result


def compute_acceleration_multipliers(
    slopes: pd.DataFrame,
    bounds: tuple[float, float] = (0.5, 3.0),
    epsilon_factor: float = 0.1,
) -> pd.DataFrame:
    """Convert slopes to bounded acceleration multipliers with denominator protection.

    Args:
        slopes: DataFrame with '{component_id}_slope' columns, indexed by fips.
        bounds: (lower, upper) bounds for acceleration multipliers.
        epsilon_factor: Factor for denominator protection (default 0.1).

    Returns:
        DataFrame with '{component_id}_acceleration' columns, indexed by fips.
    """
    lower, upper = bounds
    accel_data = {}

    slope_cols = [c for c in slopes.columns if c.endswith("_slope")]

    for slope_col in slope_cols:
        comp_id = slope_col.replace("_slope", "")
        col = slopes[slope_col]
        valid = col.dropna()

        if len(valid) == 0:
            accel_data[f"{comp_id}_acceleration"] = pd.Series(1.0, index=slopes.index)
            continue

        median_slope = valid.median()
        sigma = valid.std()

        if sigma == 0 or np.isnan(sigma):
            # No variation — all neutral
            accel_data[f"{comp_id}_acceleration"] = pd.Series(1.0, index=slopes.index)
            continue

        epsilon = epsilon_factor * sigma

        accel = pd.Series(1.0, index=slopes.index)

        if abs(median_slope) >= epsilon:
            # Normal case: ratio to median
            accel = col / median_slope
        else:
            # Small-median case: offset formulation
            accel = 1.0 + (col - median_slope) / sigma

        # NaN slopes → neutral
        accel = accel.fillna(1.0)
        # Bound
        accel = accel.clip(lower=lower, upper=upper)

        form_used = "ratio" if abs(median_slope) >= epsilon else "difference"
        at_lower = (accel == lower).sum()
        at_upper = (accel == upper).sum()
        logger.info(
            "Step 6b [%s]: form=%s, median_slope=%.4f, sigma=%.4f, "
            "at_bounds=%d (%.1f=%.0f, %.1f=%.0f)",
            comp_id, form_used, median_slope, sigma,
            at_lower + at_upper, lower, at_lower, upper, at_upper,
        )

        accel_data[f"{comp_id}_acceleration"] = accel

    result = pd.DataFrame(accel_data, index=slopes.index)
    result.index.name = "fips"

    logger.info(
        "Step 6b: Computed acceleration multipliers (bounds [%.1f, %.1f])",
        lower, upper,
    )
    return result
