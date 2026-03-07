"""Correlations with external indicators: insurance, FEMA IA, migration, etc.

Tests:
- Insurance premium growth vs state-avg CCI (expected r = 0.3-0.5)
- FEMA IA per household vs CCI (expected r = 0.3-0.6)
- Energy bill volatility vs CCI energy component (expected r = 0.3-0.5)
- Net domestic migration vs CCI, lagged (expected r = -0.1 to -0.3)
- Property value volatility vs CCI (expected r = 0.2-0.4)
- Utility rate case frequency vs state-avg CCI (expected r = 0.2-0.4)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _make_row(
    indicator: str,
    geographic_level: str = "county",
    pearson_r: float | None = None,
    spearman_r: float | None = None,
    p_value_pearson: float | None = None,
    p_value_spearman: float | None = None,
    n_observations: int | None = None,
    expected_r_low: float | None = None,
    expected_r_high: float | None = None,
    within_expected: bool | None = None,
    status: str = "success",
    note: str = "",
) -> dict:
    return {
        "indicator": indicator,
        "geographic_level": geographic_level,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "p_value_pearson": p_value_pearson,
        "p_value_spearman": p_value_spearman,
        "n_observations": n_observations,
        "expected_r_low": expected_r_low,
        "expected_r_high": expected_r_high,
        "within_expected": within_expected,
        "status": status,
        "note": note,
    }


def _correlate(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    """Compute Pearson and Spearman correlations for aligned, non-null pairs."""
    mask = x.notna() & y.notna()
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)
    n = len(x_clean)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n
    pr, pp = stats.pearsonr(x_clean, y_clean)
    sr, sp = stats.spearmanr(x_clean, y_clean)
    return float(pr), float(pp), float(sr), float(sp), n


def _validate_fema_ia(
    scored_df: pd.DataFrame,
    harmonized_df: pd.DataFrame,
    scoring_year: int,
) -> dict:
    """Indicator 1: FEMA IA per household."""
    if "fema_ia_burden" not in harmonized_df.columns:
        return _make_row(
            "fema_ia_per_household",
            status="data_unavailable",
            note="fema_ia_burden column not in harmonized data",
            expected_r_low=0.3,
            expected_r_high=0.6,
        )

    year_df = harmonized_df.loc[harmonized_df["year"] == scoring_year]
    if year_df.empty:
        return _make_row(
            "fema_ia_per_household",
            status="data_unavailable",
            note=f"No harmonized data for year {scoring_year}",
            expected_r_low=0.3,
            expected_r_high=0.6,
        )

    fema = year_df.groupby("fips")["fema_ia_burden"].mean()
    merged = scored_df[["cci_score"]].join(fema, how="inner")
    pr, pp, sr, sp, n = _correlate(merged["cci_score"], merged["fema_ia_burden"])

    if n < 3:
        return _make_row(
            "fema_ia_per_household",
            n_observations=n,
            status="insufficient_data",
            note=f"Only {n} matched counties",
            expected_r_low=0.3,
            expected_r_high=0.6,
        )

    within = 0.3 <= abs(pr) <= 0.6 if not np.isnan(pr) else None
    return _make_row(
        "fema_ia_per_household",
        pearson_r=pr,
        spearman_r=sr,
        p_value_pearson=pp,
        p_value_spearman=sp,
        n_observations=n,
        expected_r_low=0.3,
        expected_r_high=0.6,
        within_expected=within,
        status="success",
    )


def _validate_energy_volatility(
    scored_df: pd.DataFrame,
    harmonized_df: pd.DataFrame,
    scoring_year: int,
) -> dict:
    """Indicator 2: Energy bill volatility (trailing 5-year std of YoY changes)."""
    if "energy_cost_attributed" not in harmonized_df.columns:
        return _make_row(
            "energy_bill_volatility",
            status="data_unavailable",
            note="energy_cost_attributed column not in harmonized data",
            expected_r_low=0.3,
            expected_r_high=0.5,
        )

    window_start = scoring_year - 5
    window_df = harmonized_df.loc[
        (harmonized_df["year"] >= window_start) & (harmonized_df["year"] <= scoring_year),
        ["fips", "year", "energy_cost_attributed"],
    ].dropna(subset=["energy_cost_attributed"])

    if window_df.empty:
        return _make_row(
            "energy_bill_volatility",
            status="data_unavailable",
            note="No energy data in trailing 5-year window",
            expected_r_low=0.3,
            expected_r_high=0.5,
        )

    # Compute YoY changes, then std per county
    pivot = window_df.pivot_table(
        index="fips", columns="year", values="energy_cost_attributed",
    )
    yoy = pivot.diff(axis=1).iloc[:, 1:]  # drop first NaN column
    volatility = yoy.std(axis=1).rename("energy_volatility")
    volatility = volatility.dropna()

    merged = scored_df[["cci_score"]].join(volatility, how="inner")
    pr, pp, sr, sp, n = _correlate(merged["cci_score"], merged["energy_volatility"])

    if n < 3:
        return _make_row(
            "energy_bill_volatility",
            n_observations=n,
            status="insufficient_data",
            note=f"Only {n} matched counties",
            expected_r_low=0.3,
            expected_r_high=0.5,
        )

    within = 0.3 <= abs(pr) <= 0.5 if not np.isnan(pr) else None
    return _make_row(
        "energy_bill_volatility",
        pearson_r=pr,
        spearman_r=sr,
        p_value_pearson=pp,
        p_value_spearman=sp,
        n_observations=n,
        expected_r_low=0.3,
        expected_r_high=0.5,
        within_expected=within,
        status="success",
        note="Energy data is state-level; within-state counties share volatility",
    )


def _validate_migration(
    scored_df: pd.DataFrame,
    settings: Settings,
) -> dict:
    """Indicator 3: Net domestic migration (population change proxy)."""
    acs_dir = settings.raw_dir / "census_acs"
    if not acs_dir.exists():
        return _make_row(
            "net_domestic_migration",
            status="data_unavailable",
            note="Census ACS directory not found",
            expected_r_low=-0.3,
            expected_r_high=-0.1,
        )

    # Look for multi-year ACS parquet files
    acs_files = sorted(acs_dir.glob("census_acs_*.parquet"))
    if len(acs_files) < 2:
        return _make_row(
            "net_domestic_migration",
            status="data_unavailable",
            note=f"Need >=2 years of ACS data, found {len(acs_files)}",
            expected_r_low=-0.3,
            expected_r_high=-0.1,
        )

    try:
        frames = []
        for f in acs_files:
            df = pd.read_parquet(f)
            if "fips" in df.columns and "population" in df.columns and "year" in df.columns:
                frames.append(df[["fips", "year", "population"]])
        if len(frames) < 2:
            return _make_row(
                "net_domestic_migration",
                status="data_unavailable",
                note="ACS files lack required columns (fips, year, population)",
                expected_r_low=-0.3,
                expected_r_high=-0.1,
            )
        acs = pd.concat(frames, ignore_index=True)
        pivot = acs.pivot_table(index="fips", columns="year", values="population")
        if pivot.shape[1] < 2:
            return _make_row(
                "net_domestic_migration",
                status="data_unavailable",
                note="Fewer than 2 years of population data after pivoting",
                expected_r_low=-0.3,
                expected_r_high=-0.1,
            )

        # Use most recent 2 years for change rate
        cols = sorted(pivot.columns)
        pop_change = (pivot[cols[-1]] - pivot[cols[-2]]) / pivot[cols[-2]]
        pop_change = pop_change.replace([np.inf, -np.inf], np.nan).dropna()
        pop_change.name = "pop_change"

        merged = scored_df[["cci_score"]].join(pop_change, how="inner")
        pr, pp, sr, sp, n = _correlate(merged["cci_score"], merged["pop_change"])

        if n < 3:
            return _make_row(
                "net_domestic_migration",
                n_observations=n,
                status="insufficient_data",
                expected_r_low=-0.3,
                expected_r_high=-0.1,
            )

        within = -0.3 <= pr <= -0.1 if not np.isnan(pr) else None
        return _make_row(
            "net_domestic_migration",
            pearson_r=pr,
            spearman_r=sr,
            p_value_pearson=pp,
            p_value_spearman=sp,
            n_observations=n,
            expected_r_low=-0.3,
            expected_r_high=-0.1,
            within_expected=within,
            status="success",
            note="Population change used as migration proxy",
        )
    except Exception as e:
        logger.warning("Migration validation failed: %s", e)
        return _make_row(
            "net_domestic_migration",
            status="failed",
            note=str(e),
            expected_r_low=-0.3,
            expected_r_high=-0.1,
        )


def run_external_validation(
    scored_df: pd.DataFrame,
    harmonized_df: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Compute correlations between CCI and external indicators.

    Args:
        scored_df: DataFrame with fips index and 'cci_score' column.
        harmonized_df: Multi-year harmonized data (for FEMA IA and energy volatility).
        settings: Pipeline settings (for file paths).

    Returns:
        DataFrame with one row per indicator containing correlation results.
    """
    if settings is None:
        settings = get_settings()

    rows: list[dict] = []

    # Ensure scored_df has fips as index
    if "fips" in scored_df.columns:
        scored_df = scored_df.set_index("fips")

    if scored_df.empty or "cci_score" not in scored_df.columns:
        logger.warning("scored_df is empty or missing cci_score — skipping all indicators")
        for ind in [
            "fema_ia_per_household", "energy_bill_volatility",
            "net_domestic_migration", "insurance_premium_growth",
            "property_value_volatility", "utility_rate_case_frequency",
        ]:
            rows.append(_make_row(ind, status="no_scored_data"))
        return pd.DataFrame(rows)

    # Indicator 1: FEMA IA per household
    if harmonized_df is not None:
        rows.append(_validate_fema_ia(scored_df, harmonized_df, settings.scoring_year))
    else:
        rows.append(_make_row(
            "fema_ia_per_household",
            status="data_unavailable",
            note="harmonized_df not provided",
            expected_r_low=0.3,
            expected_r_high=0.6,
        ))

    # Indicator 2: Energy bill volatility
    if harmonized_df is not None:
        rows.append(_validate_energy_volatility(scored_df, harmonized_df, settings.scoring_year))
    else:
        rows.append(_make_row(
            "energy_bill_volatility",
            status="data_unavailable",
            note="harmonized_df not provided",
            expected_r_low=0.3,
            expected_r_high=0.5,
        ))

    # Indicator 3: Net domestic migration
    rows.append(_validate_migration(scored_df, settings))

    # Indicator 4: Insurance premium growth (NOT AVAILABLE in v1)
    rows.append(_make_row(
        "insurance_premium_growth",
        geographic_level="state",
        status="data_unavailable",
        note="Requires NAIC data — targeted for v2",
        expected_r_low=0.3,
        expected_r_high=0.5,
    ))

    # Indicator 5: Property value volatility (NOT AVAILABLE in v1)
    rows.append(_make_row(
        "property_value_volatility",
        status="data_unavailable",
        note="Requires Zillow/FHFA data — targeted for v2",
        expected_r_low=0.2,
        expected_r_high=0.4,
    ))

    # Indicator 6: Utility rate case frequency (NOT AVAILABLE in v1)
    rows.append(_make_row(
        "utility_rate_case_frequency",
        geographic_level="state",
        status="data_unavailable",
        note="Requires PUC filing data — targeted for v2",
        expected_r_low=0.2,
        expected_r_high=0.4,
    ))

    logger.info(
        "External validation: %d indicators computed, %d unavailable",
        sum(1 for r in rows if r["status"] == "success"),
        sum(1 for r in rows if r["status"] == "data_unavailable"),
    )

    return pd.DataFrame(rows)
