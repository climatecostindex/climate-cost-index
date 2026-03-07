"""Deterministic CCI computation pipeline.

THE EXACT SEQUENCE BELOW IS INVIOLABLE. No step may be reordered.
Each step depends on the output of the previous step.

See CCI_V1_Implementation_Plan.md Phase 3 for full specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from config.components import COMPONENTS, get_weights
from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class CCIOutput:
    """Complete output bundle from a CCI scoring run."""

    scores: pd.DataFrame          # CCI-Score per county
    components: pd.DataFrame      # Per-component contribution per county
    penalties: dict[str, float]   # Overlap penalty factors
    accelerations: pd.DataFrame   # Acceleration multipliers per component × county
    corr_matrix: pd.DataFrame     # Component correlation matrix
    robustness_checks: dict       # Spearman, distance correlation checks
    national: float               # CCI-National (housing-unit-weighted aggregate)
    strain: pd.DataFrame          # CCI-Strain (income-adjusted) per county
    dollar: pd.DataFrame          # CCI-Dollar (energy-only adder) per county
    k: float                      # Scaling constant (fixed at v1 launch)
    universe: pd.Index            # FIPS codes in the scoring universe

    def save(self, output_dir) -> None:
        """Persist all outputs to parquet files."""
        from pathlib import Path

        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.scores.to_parquet(d / "cci_scores.parquet")
        self.components.to_parquet(d / "cci_components.parquet")
        self.accelerations.to_parquet(d / "cci_accelerations.parquet")
        self.corr_matrix.to_parquet(d / "cci_correlation_matrix.parquet")
        self.strain.to_parquet(d / "cci_strain.parquet")
        self.dollar.to_parquet(d / "cci_dollar.parquet")

        # Save metadata sidecar
        import json
        meta = {
            "k": self.k,
            "national": self.national,
            "universe_size": len(self.universe),
            "penalties": self.penalties,
        }
        (d / "cci_scores_metadata.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved CCI outputs to %s", d)


def compute_cci(
    harmonized_df: pd.DataFrame,
    weights: dict[str, float],
    settings: Settings,
) -> CCIOutput:
    """Run the full deterministic scoring pipeline.

    Steps (order is fixed):
        1. Transform inputs (log/sqrt for heavy-tailed variables)
        2. Winsorize at 99th percentile
        3. Define scoring universe + compute national percentile ranks
        4. Center (subtract 50)
        5. Compute overlap penalties (correlation + precedence)
        6. Compute acceleration multipliers (Theil-Sen slopes)
        7. Handle missingness (imputation for Preferred Core)
        8. Compute weighted component scores
        9. Sum to composite S(c)
       10. Scale to CCI(c) = 100 + k * S(c)

    Args:
        harmonized_df: Full multi-year harmonized DataFrame with 'fips', 'year',
            and component columns.
        weights: {component_id: normalized_weight} from get_weights().
        settings: Scoring parameters.

    Returns:
        CCIOutput bundle with all scoring results.
    """
    from score.acceleration import (
        compute_acceleration_multipliers,
        compute_theil_sen_slopes,
    )
    from score.cci_dollar import compute_dollar
    from score.cci_national import compute_national_aggregate
    from score.cci_strain import compute_strain
    from score.center import center
    from score.composite import calibrate_k, compute_component_scores
    from score.missingness import handle_missingness
    from score.overlap import (
        compute_correlation_matrix,
        compute_correlation_robustness,
        compute_overlap_penalties,
    )
    from score.percentile import compute_percentiles
    from score.transform_inputs import transform_inputs
    from score.universe import define_universe
    from score.winsorize import winsorize

    component_ids = [c for c in COMPONENTS if c in harmonized_df.columns]
    scoring_year = settings.scoring_year
    logger.info("Starting CCI pipeline for year %d with %d components", scoring_year, len(component_ids))

    # Separate scoring year from historical data
    scoring_year_df = harmonized_df[harmonized_df["year"] == scoring_year].copy()
    if "fips" in scoring_year_df.columns:
        scoring_year_df = scoring_year_df.set_index("fips")
    logger.info("Scoring year %d: %d counties", scoring_year, len(scoring_year_df))

    # Save original energy costs for CCI-Dollar (before any transforms)
    energy_raw = scoring_year_df["energy_cost_attributed"].copy() if "energy_cost_attributed" in scoring_year_df.columns else pd.Series(dtype=float)

    # ── Step 1: Transform Inputs ──
    transformed = transform_inputs(scoring_year_df)

    # ── Step 2: Winsorize ──
    winsorized = winsorize(transformed, percentile=settings.winsorize_percentile)

    # ── Step 3: Define Universe + Percentile Rank ──
    universe = define_universe(winsorized)
    percentiled = compute_percentiles(winsorized, universe, component_ids=component_ids)

    # ── Step 4: Center ──
    centered = center(percentiled, component_ids=component_ids)

    # ── Step 5: Overlap Penalties ──
    corr_matrix = compute_correlation_matrix(centered, universe, component_ids=component_ids)
    penalties, penalty_docs = compute_overlap_penalties(
        corr_matrix,
        threshold=settings.overlap_correlation_threshold,
        floor=settings.overlap_penalty_floor,
    )
    robustness_checks = compute_correlation_robustness(centered, universe, component_ids=component_ids)

    # ── Step 6: Acceleration Multipliers ──
    # Uses FULL multi-year data (not just scoring year)
    slopes = compute_theil_sen_slopes(
        harmonized_df,
        scoring_year=scoring_year,
        component_ids=component_ids,
        min_completeness=settings.acceleration_min_completeness,
    )
    accelerations = compute_acceleration_multipliers(
        slopes,
        bounds=settings.acceleration_bounds,
        epsilon_factor=settings.acceleration_denominator_epsilon_factor,
    )

    # ── Step 7: Missingness Handling ──
    centered = handle_missingness(centered)

    # ── Step 8: Component Scores ──
    component_scores = compute_component_scores(
        centered, weights, penalties, accelerations,
    )

    # ── Step 9: Sum to Composite ──
    raw_composite = component_scores.sum(axis=1)
    raw_composite.name = "raw_composite"
    logger.info(
        "Step 9: Raw composite — mean=%.4f, median=%.4f, IQR=[%.4f, %.4f]",
        raw_composite.mean(), raw_composite.median(),
        raw_composite.quantile(0.25), raw_composite.quantile(0.75),
    )

    # ── Step 10: Scale to CCI-Score ──
    k = calibrate_k(raw_composite, target_iqr=settings.target_iqr)
    cci_scores = 100 + k * raw_composite
    cci_scores.name = "cci_score"

    scores_df = pd.DataFrame({
        "cci_score": cci_scores,
        "raw_composite": raw_composite,
    })
    scores_df.index.name = "fips"

    logger.info(
        "Step 10: CCI-Score — mean=%.1f, median=%.1f, IQR=[%.1f, %.1f], k=%.4f",
        cci_scores.mean(), cci_scores.median(),
        cci_scores.quantile(0.25), cci_scores.quantile(0.75), k,
    )

    # ── Data Quality Tier ──
    scores_df["data_quality_tier"] = _compute_quality_tier(scoring_year_df, universe)

    # ── Variant Outputs ──

    # Load Census ACS reference data for variants
    census_df = _load_census_acs(scoring_year, settings)

    # CCI-National: housing-unit-weighted aggregate
    housing_col = _find_housing_units_column(scoring_year_df)
    if housing_col is not None:
        housing_units = scoring_year_df[housing_col].reindex(universe)
    elif census_df is not None and "total_housing_units" in census_df.columns:
        housing_units = census_df.set_index("fips")["total_housing_units"].reindex(universe)
    else:
        housing_units = None

    if housing_units is not None:
        national = compute_national_aggregate(raw_composite, housing_units)
    else:
        logger.warning("No housing_units data found; CCI-National set to raw mean")
        national = float(raw_composite.mean())

    # CCI-Strain: income-adjusted
    income_col = _find_income_column(scoring_year_df)
    if income_col is not None:
        median_income = scoring_year_df[income_col].reindex(universe)
    elif census_df is not None and "median_household_income" in census_df.columns:
        median_income = census_df.set_index("fips")["median_household_income"].reindex(universe)
    else:
        median_income = None

    if median_income is not None:
        strain_df = compute_strain(cci_scores, median_income)
    else:
        logger.warning("No median_income data found; CCI-Strain unavailable")
        strain_df = pd.DataFrame({"cci_strain": pd.Series(dtype=float)})

    # CCI-Dollar: raw energy cost pass-through
    dollar_df = compute_dollar(energy_raw.reindex(universe))

    return CCIOutput(
        scores=scores_df,
        components=component_scores,
        penalties=penalties,
        accelerations=accelerations,
        corr_matrix=corr_matrix,
        robustness_checks=robustness_checks,
        national=national,
        strain=strain_df,
        dollar=dollar_df,
        k=k,
        universe=universe,
    )


def _load_census_acs(scoring_year: int, settings: Settings) -> pd.DataFrame | None:
    """Load Census ACS data for variant outputs (housing units, income)."""
    from pathlib import Path

    raw_dir = settings.raw_dir / "census_acs"
    # Try scoring year first, then fall back to most recent available
    for year in range(scoring_year, scoring_year - 5, -1):
        path = raw_dir / f"census_acs_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            logger.info("Loaded Census ACS data from %s (%d rows)", path.name, len(df))
            return df

    logger.warning("No Census ACS data found in %s", raw_dir)
    return None


def _find_housing_units_column(df: pd.DataFrame) -> str | None:
    """Find the housing units column in the DataFrame."""
    candidates = ["housing_units", "total_housing_units"]
    # Also check namespaced auxiliary columns
    for col in df.columns:
        if "housing_units" in col.lower():
            candidates.append(col)
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_income_column(df: pd.DataFrame) -> str | None:
    """Find the median household income column in the DataFrame."""
    candidates = ["median_household_income", "median_income"]
    for col in df.columns:
        if "median" in col.lower() and "income" in col.lower():
            candidates.append(col)
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _compute_quality_tier(
    scoring_year_df: pd.DataFrame,
    universe: pd.Index,
) -> pd.Series:
    """Assign a data quality tier to each county in the scoring universe.

    Tiers:
      - "gold": All 12 components from direct observation (no gap-fill).
      - "silver": Has gap-filled components, but all are within close range
        (temperature IDW or AQ <50km). No forward-filled or zero-filled.
      - "bronze": Has distant AQ interpolation (>50km), forward-filled
        health_burden, or substantial imputation.

    Args:
        scoring_year_df: Scoring-year DataFrame indexed by fips.
        universe: FIPS codes in the scoring universe.

    Returns:
        Series indexed by universe fips with tier values.
    """
    comp_ids = [c for c in COMPONENTS if c in scoring_year_df.columns]
    df = scoring_year_df.reindex(universe)

    tier = pd.Series("gold", index=universe)

    # Check for any gap-filled components
    gap_fill_cols = [f"{c}__gap_filled" for c in comp_ids if f"{c}__gap_filled" in df.columns]
    if gap_fill_cols:
        any_gap_filled = df[gap_fill_cols].any(axis=1)
        tier[any_gap_filled] = "silver"

    # Check for distant AQ interpolation (>50km → bronze)
    for aq_comp in ("pm25_annual", "aqi_unhealthy_days"):
        dist_col = f"{aq_comp}__nearest_monitor_km"
        flag_col = f"{aq_comp}__gap_filled"
        if dist_col in df.columns and flag_col in df.columns:
            far_aq = (df[flag_col] == True) & (df[dist_col] > 50)  # noqa: E712
            tier[far_aq] = "bronze"

    # Check for missing components (imputed to 0 in scoring)
    # Counties missing 3+ Preferred Core components → bronze
    from score.universe import PREFERRED_CORE
    preferred_in_df = [c for c in PREFERRED_CORE if c in df.columns]
    if preferred_in_df:
        n_missing = df[preferred_in_df].isna().sum(axis=1)
        tier[n_missing >= 3] = "bronze"

    counts = tier.value_counts()
    logger.info(
        "Data quality tiers: gold=%d, silver=%d, bronze=%d",
        counts.get("gold", 0), counts.get("silver", 0), counts.get("bronze", 0),
    )
    return tier
