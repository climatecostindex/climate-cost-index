"""Isolate climate-attributed portion of residential energy costs.

THIS IS THE MOST METHODOLOGICALLY IMPORTANT TRANSFORM.

Input: State-level electricity prices and consumption from ingest/eia_energy.py
       RECS household consumption data from ingest/eia_energy.py
       HDD/CDD anomalies from transform/degree_days.py

Panel regression:
  consumption_it = a + b1(HDD_anomaly) + b2(CDD_anomaly)
                   + state_FE + year_FE + break_dummies + e

Climate-attributed cost = [b1 * HDD_anomaly + b2 * CDD_anomaly] * price

This is the ONLY component with full causal attribution in v1.

Output columns: fips, year, climate_attributed_energy_cost,
               total_energy_cost, attribution_fraction,
               regression_r_squared, structural_breaks_detected
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config.settings import HARMONIZED_DIR, RAW_DIR
from ingest.utils import normalize_fips

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RECS baseline dwelling for normalization (SSRN spec)
BASELINE_SQFT = 1800
BASELINE_DWELLING_TYPE = "single-family detached"
BASELINE_OCCUPANTS = 3
BASELINE_SQFT_RANGE = (1500, 2100)
MIN_BASELINE_MATCHES = 30

# Structural break detection threshold — 10% captures discrete rate-case
# events while ignoring normal market fluctuations (fuel cost passthrough,
# seasonal adjustments).  The SSRN paper says "rate-case step changes" without
# specifying a percentage; 10% aligns with energy economics literature for
# identifying regulatory pricing regime changes (typical rate cases produce
# 10-20% jumps).  The original 5% threshold was too sensitive, generating ~190
# break dummies for ~761 observations and inflating R² through overfitting.
RATE_CASE_PRICE_JUMP_PCT = 0.10

# File paths
ENERGY_COMBINED_PATH = RAW_DIR / "eia_energy" / "eia_state_aggregate.parquet"
ENERGY_COMBINED_PATH_ALT = RAW_DIR / "eia_energy" / "eia_energy_all.parquet"
ENERGY_DIR = RAW_DIR / "eia_energy"
ENERGY_PER_YEAR_GLOB = "eia_energy_*.parquet"
RECS_PATH = RAW_DIR / "eia_energy" / "eia_recs_microdata.parquet"
RECS_PATH_ALT = RAW_DIR / "eia_energy" / "recs_microdata.parquet"
CENSUS_COMBINED_PATH = RAW_DIR / "census_acs" / "census_acs_all.parquet"
CENSUS_DIR = RAW_DIR / "census_acs"
CENSUS_PER_YEAR_GLOB = "census_acs_*.parquet"
DEGREE_DAYS_DIR = HARMONIZED_DIR
DEGREE_DAYS_GLOB = "degree_days_*.parquet"

OUTPUT_COLUMNS = [
    "fips",
    "year",
    "climate_attributed_energy_cost",
    "total_energy_cost",
    "attribution_fraction",
    "regression_r_squared",
    "structural_breaks_detected",
]

METADATA_SOURCE = "EIA_ENERGY"
METADATA_CONFIDENCE = "A"
METADATA_ATTRIBUTION = "attributed"

# State FIPS → Census Division mapping (Bureau of Census divisions 1-9)
STATE_TO_CENSUS_DIVISION: dict[str, int] = {
    "09": 1, "23": 1, "25": 1, "33": 1, "44": 1, "50": 1,  # New England
    "34": 2, "36": 2, "42": 2,  # Middle Atlantic
    "17": 3, "18": 3, "26": 3, "39": 3, "55": 3,  # East North Central
    "19": 4, "20": 4, "27": 4, "29": 4, "31": 4, "38": 4, "46": 4,  # West North Central
    "10": 5, "11": 5, "12": 5, "13": 5, "24": 5, "37": 5,  # South Atlantic
    "45": 5, "51": 5, "54": 5,
    "01": 6, "21": 6, "28": 6, "47": 6,  # East South Central
    "05": 7, "22": 7, "40": 7, "48": 7,  # West South Central
    "04": 8, "08": 8, "16": 8, "30": 8, "32": 8, "35": 8,  # Mountain
    "49": 8, "56": 8,
    "02": 9, "06": 9, "15": 9, "41": 9, "53": 9,  # Pacific
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_energy_attribution(
    energy_data: pd.DataFrame,
    recs_data: pd.DataFrame,
    degree_days: pd.DataFrame,
    census_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run degree-day regression and isolate climate-attributed energy cost.

    Args:
        energy_data: State-level energy data with columns ``state_fips``,
            ``year``, ``electricity_price_cents_kwh``,
            ``electricity_consumption_mwh`` (million kWh).
        recs_data: RECS microdata with columns ``census_division``,
            ``dwelling_type``, ``square_footage``, ``num_occupants``,
            ``annual_electricity_kwh``.
        degree_days: County-level degree-day anomalies with columns
            ``fips``, ``year``, ``hdd_anomaly``, ``cdd_anomaly``.
        census_data: Census ACS county data with ``fips``, ``year``,
            ``total_housing_units``. Used to convert state-total
            consumption to per-household. If None, loaded from disk.

    Returns:
        DataFrame with columns per OUTPUT_COLUMNS at county-year grain.
    """
    # --- Validate inputs ---
    _validate_energy_columns(energy_data)
    _validate_recs_columns(recs_data)
    _validate_degree_days_columns(degree_days)

    if energy_data.empty:
        logger.warning("Empty energy data — returning empty result.")
        return _empty_output()
    if degree_days.empty:
        logger.warning("Empty degree-day data — returning empty result.")
        return _empty_output()

    # --- Step 1: Clean energy data ---
    energy = _clean_energy_data(energy_data)
    if energy.empty:
        logger.warning("No valid energy data after cleaning — returning empty result.")
        return _empty_output()

    # --- Step 2-3: Compute RECS normalization factors ---
    norm_factors = _compute_recs_normalization(recs_data)

    # --- Step 3b: Compute state-level housing units from Census ACS ---
    state_housing = _compute_state_housing_units(census_data, degree_days)

    # --- Step 4: Aggregate county degree-day anomalies to state level ---
    state_anomalies = _aggregate_anomalies_to_state(degree_days)
    if state_anomalies.empty:
        logger.warning("No state-level anomalies computed — returning empty result.")
        return _empty_output()

    # --- Step 5: Construct panel dataset ---
    panel = _construct_panel(energy, state_anomalies, norm_factors, state_housing)
    if panel.empty:
        logger.warning("Empty panel after merge — returning empty result.")
        return _empty_output()

    # --- Step 6: Detect structural breaks ---
    break_counts, break_dummies = _detect_all_structural_breaks(panel)

    # --- Step 7: Run panel regression ---
    beta_hdd, beta_cdd, r_squared = _run_panel_regression(panel, break_dummies)

    # --- Step 8: Compute climate-attributed energy cost ---
    state_results = _compute_climate_costs(panel, beta_hdd, beta_cdd, r_squared, break_counts)

    # --- Step 9: Map state-level results to counties ---
    county_results = _map_to_counties(state_results, degree_days)

    return county_results


def detect_structural_breaks(
    price_series: pd.Series,
    threshold: float = RATE_CASE_PRICE_JUMP_PCT,
) -> list[int]:
    """Detect discrete price jumps > threshold year-over-year.

    Args:
        price_series: Series indexed by year with electricity prices.
        threshold: Fractional threshold for detecting breaks (default 0.05 = 5%).

    Returns:
        List of years where a structural break was detected.
    """
    if len(price_series) < 2:
        return []

    sorted_series = price_series.sort_index()
    pct_change = sorted_series.pct_change()
    break_mask = pct_change.abs() > threshold
    break_years = sorted_series.index[break_mask].tolist()
    return [int(y) for y in break_years]


def run() -> pd.DataFrame:
    """Run the full transform: load from disk, compute, save parquet + metadata.

    Convenience entry point for ``pipeline/run_transform.py``.
    """
    energy_data = _load_energy_data()
    recs_data = _load_recs_data()
    degree_days = _load_degree_days()
    census_data = _load_census_data()

    result = compute_energy_attribution(energy_data, recs_data, degree_days, census_data)

    if result.empty:
        logger.warning("No energy attribution results to write.")
        return result

    # Save per-year outputs
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    years = sorted(result["year"].unique())
    for yr in years:
        year_df = result[result["year"] == yr].copy()
        parquet_path = HARMONIZED_DIR / f"energy_attribution_{yr}.parquet"
        metadata_path = HARMONIZED_DIR / f"energy_attribution_{yr}_metadata.json"

        year_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d counties)", parquet_path, len(year_df))

        _write_metadata(metadata_path, yr)
        logger.info("Wrote %s", metadata_path)

    logger.info(
        "Energy attribution transform complete: %d years, %d total county-year rows",
        len(years),
        len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_energy_columns(df: pd.DataFrame) -> None:
    required = {"state_fips", "year", "electricity_price_cents_kwh", "electricity_consumption_mwh"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Energy data missing columns: {sorted(missing)}")


def _validate_recs_columns(df: pd.DataFrame) -> None:
    required = {
        "census_division", "dwelling_type", "square_footage",
        "num_occupants", "annual_electricity_kwh",
    }
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"RECS data missing columns: {sorted(missing)}")


def _validate_degree_days_columns(df: pd.DataFrame) -> None:
    required = {"fips", "year", "hdd_anomaly", "cdd_anomaly"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Degree-day data missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Data loading helpers (for run() only)
# ---------------------------------------------------------------------------
def _load_energy_data() -> pd.DataFrame:
    """Load state-level energy data from cached parquet files."""
    for path in [ENERGY_COMBINED_PATH, ENERGY_COMBINED_PATH_ALT]:
        if path.exists():
            logger.info("Loading combined energy data from %s", path)
            return pd.read_parquet(path)

    # Fall back to per-year files
    per_year = sorted(ENERGY_DIR.glob(ENERGY_PER_YEAR_GLOB))
    per_year = [
        p for p in per_year
        if "all" not in p.name and "recs" not in p.name
        and "aggregate" not in p.name and "microdata" not in p.name
    ]
    if not per_year:
        raise FileNotFoundError(
            f"No energy data files found at {ENERGY_COMBINED_PATH} "
            f"or matching {ENERGY_DIR / ENERGY_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year energy files (fallback)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


def _load_recs_data() -> pd.DataFrame:
    """Load RECS microdata from cached parquet."""
    for path in [RECS_PATH, RECS_PATH_ALT]:
        if path.exists():
            logger.info("Loading RECS microdata from %s", path)
            return pd.read_parquet(path)
    raise FileNotFoundError(
        f"RECS microdata not found at {RECS_PATH} or {RECS_PATH_ALT}. "
        "Run eia_energy.py ingester first. RECS normalization is required."
    )


def _load_degree_days() -> pd.DataFrame:
    """Load degree-day anomalies from harmonized parquet files."""
    dd_files = sorted(DEGREE_DAYS_DIR.glob(DEGREE_DAYS_GLOB))
    if not dd_files:
        raise FileNotFoundError(
            f"No degree-day files found matching {DEGREE_DAYS_DIR / DEGREE_DAYS_GLOB}. "
            "Run degree_days.py (Module 2.3) first."
        )
    logger.info("Loading %d degree-day files", len(dd_files))
    dfs = [pd.read_parquet(p) for p in dd_files]
    return pd.concat(dfs, ignore_index=True)


def _load_census_data() -> pd.DataFrame:
    """Load Census ACS county data from cached parquet files."""
    if CENSUS_COMBINED_PATH.exists():
        logger.info("Loading combined Census ACS data from %s", CENSUS_COMBINED_PATH)
        return pd.read_parquet(CENSUS_COMBINED_PATH)

    per_year = sorted(CENSUS_DIR.glob(CENSUS_PER_YEAR_GLOB))
    per_year = [p for p in per_year if "all" not in p.name]
    if not per_year:
        raise FileNotFoundError(
            f"No Census ACS data found at {CENSUS_COMBINED_PATH} "
            f"or matching {CENSUS_DIR / CENSUS_PER_YEAR_GLOB}"
        )

    logger.info("Loading %d per-year Census ACS files (fallback)", len(per_year))
    dfs = [pd.read_parquet(p) for p in per_year]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def _clean_energy_data(energy_data: pd.DataFrame) -> pd.DataFrame:
    """Clean energy data: normalize FIPS, filter invalid values."""
    df = energy_data.copy()

    # Normalize state_fips to 2-digit zero-padded strings
    df["state_fips"] = df["state_fips"].apply(lambda x: f"{int(x):02d}")

    # Flag negative prices as corruption
    neg_price = df["electricity_price_cents_kwh"] < 0
    if neg_price.any():
        logger.warning(
            "Found %d rows with negative electricity price — setting to NaN",
            neg_price.sum(),
        )
        df.loc[neg_price, "electricity_price_cents_kwh"] = np.nan

    # Flag negative consumption as corruption
    neg_cons = df["electricity_consumption_mwh"] < 0
    if neg_cons.any():
        logger.warning(
            "Found %d rows with negative electricity consumption — setting to NaN",
            neg_cons.sum(),
        )
        df.loc[neg_cons, "electricity_consumption_mwh"] = np.nan

    # Drop rows with NaN price or consumption
    before = len(df)
    df = df.dropna(subset=["electricity_price_cents_kwh", "electricity_consumption_mwh"])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN price or consumption", dropped)

    return df


def _compute_recs_normalization(recs_data: pd.DataFrame) -> dict[int, float]:
    """Compute RECS baseline normalization factors by census division.

    Returns:
        Dict mapping census_division → normalization_factor.
        normalization_factor = baseline_consumption / avg_consumption.
    """
    if recs_data.empty:
        logger.warning("Empty RECS data — using normalization factor of 1.0 for all divisions.")
        return {}

    df = recs_data.copy()
    # Ensure census_division is int
    df["census_division"] = df["census_division"].astype(int)

    # Compute national-level baseline as fallback
    national_baseline = _compute_baseline_for_group(df)

    norm_factors: dict[int, float] = {}
    divisions = sorted(df["census_division"].unique())

    for div in divisions:
        div_data = df[df["census_division"] == div]
        baseline_matches = _filter_baseline_households(div_data)

        if len(baseline_matches) < MIN_BASELINE_MATCHES:
            logger.warning(
                "Census division %d has only %d baseline matches (< %d) — "
                "falling back to national-level normalization",
                div, len(baseline_matches), MIN_BASELINE_MATCHES,
            )
            norm_factors[div] = national_baseline
        else:
            baseline_kwh = baseline_matches["annual_electricity_kwh"].mean()
            avg_kwh = div_data["annual_electricity_kwh"].mean()
            if avg_kwh > 0:
                norm_factors[div] = baseline_kwh / avg_kwh
            else:
                norm_factors[div] = 1.0

    logger.info(
        "RECS normalization factors computed for %d census divisions", len(norm_factors)
    )
    return norm_factors


def _filter_baseline_households(df: pd.DataFrame) -> pd.DataFrame:
    """Filter RECS data to baseline dwelling profile."""
    mask = (
        (df["dwelling_type"].str.lower().str.contains("single-family", na=False))
        & (df["square_footage"] >= BASELINE_SQFT_RANGE[0])
        & (df["square_footage"] <= BASELINE_SQFT_RANGE[1])
        & (df["num_occupants"] == BASELINE_OCCUPANTS)
    )
    return df[mask]


def _compute_baseline_for_group(df: pd.DataFrame) -> float:
    """Compute national-level normalization factor as fallback."""
    baseline_matches = _filter_baseline_households(df)
    if baseline_matches.empty:
        logger.warning("No baseline matches even at national level — using factor 1.0")
        return 1.0
    avg_all = df["annual_electricity_kwh"].mean()
    if avg_all <= 0:
        return 1.0
    return baseline_matches["annual_electricity_kwh"].mean() / avg_all


def _compute_state_housing_units(
    census_data: pd.DataFrame | None,
    degree_days: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate county housing units to state-year totals.

    Args:
        census_data: Census ACS with fips, year, total_housing_units.
            If None, attempts to load from disk.
        degree_days: Used to determine required years.

    Returns:
        DataFrame with state_fips, year, total_housing_units (state total).
    """
    if census_data is None:
        try:
            census_data = _load_census_data()
        except FileNotFoundError:
            logger.warning(
                "Census ACS data not available — cannot compute per-household consumption. "
                "Falling back to state-total (NOT per-household)."
            )
            return pd.DataFrame(columns=["state_fips", "year", "total_housing_units"])

    df = census_data.copy()
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]

    # Aggregate county housing units to state level
    state_hu = (
        df.groupby(["state_fips", "year"])["total_housing_units"]
        .sum()
        .reset_index()
    )

    logger.info(
        "State housing units computed: %d state-year rows, %d states",
        len(state_hu),
        state_hu["state_fips"].nunique(),
    )
    return state_hu


def _aggregate_anomalies_to_state(degree_days: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county-level degree-day anomalies to state-level means."""
    df = degree_days.copy()

    # Extract state FIPS from county FIPS (first 2 digits)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]

    # Drop rows with NaN anomalies
    df = df.dropna(subset=["hdd_anomaly", "cdd_anomaly"])

    if df.empty:
        return pd.DataFrame(columns=["state_fips", "year", "hdd_anomaly", "cdd_anomaly"])

    state_anom = df.groupby(["state_fips", "year"]).agg(
        hdd_anomaly=("hdd_anomaly", "mean"),
        cdd_anomaly=("cdd_anomaly", "mean"),
    ).reset_index()

    logger.info(
        "Aggregated degree-day anomalies to state level: %d state-year rows across %d states",
        len(state_anom), state_anom["state_fips"].nunique(),
    )

    return state_anom


def _construct_panel(
    energy: pd.DataFrame,
    state_anomalies: pd.DataFrame,
    norm_factors: dict[int, float],
    state_housing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct the state x year panel dataset for regression.

    Consumption is converted to per-household kWh:
        (electricity_consumption_mwh * 1,000,000) / total_housing_units

    The EIA column ``electricity_consumption_mwh`` is actually **million kWh**
    (from the EIA ``electricity/retail-sales`` endpoint). So:
        per_household_kwh = million_kwh * 1_000_000 / housing_units

    RECS normalization is then applied to adjust for dwelling mix.
    """
    # Merge energy with anomalies
    panel = energy.merge(state_anomalies, on=["state_fips", "year"], how="inner")

    # Log states excluded due to missing anomaly data
    energy_states = set(energy["state_fips"].unique())
    anomaly_states = set(state_anomalies["state_fips"].unique())
    excluded = energy_states - anomaly_states
    if excluded:
        logger.warning(
            "Excluded %d states from panel — no degree-day anomaly data: %s",
            len(excluded), sorted(excluded),
        )

    if panel.empty:
        return panel

    # Apply RECS normalization: map state to census division, then apply factor
    panel["census_division"] = panel["state_fips"].map(STATE_TO_CENSUS_DIVISION)
    panel["norm_factor"] = panel["census_division"].map(norm_factors).fillna(1.0)

    # Convert state-total consumption to per-household kWh
    # electricity_consumption_mwh is actually million kWh (EIA API naming)
    has_housing = (
        state_housing is not None
        and not state_housing.empty
    )
    if has_housing:
        panel = panel.merge(
            state_housing[["state_fips", "year", "total_housing_units"]],
            on=["state_fips", "year"],
            how="left",
        )
        missing_hu = panel["total_housing_units"].isna()
        if missing_hu.any():
            # For missing years, forward/backward fill within each state
            panel["total_housing_units"] = (
                panel.groupby("state_fips")["total_housing_units"]
                .transform(lambda s: s.ffill().bfill())
            )
            still_missing = panel["total_housing_units"].isna()
            if still_missing.any():
                logger.warning(
                    "%d state-year rows missing housing units after fill — "
                    "these rows will use state-total consumption (not per-household)",
                    still_missing.sum(),
                )

        # per-household kWh = (million_kwh * 1,000,000) / housing_units
        valid_hu = panel["total_housing_units"].notna() & (panel["total_housing_units"] > 0)
        panel["per_household_kwh"] = np.where(
            valid_hu,
            panel["electricity_consumption_mwh"] * 1_000_000 / panel["total_housing_units"],
            panel["electricity_consumption_mwh"] * 1_000_000,  # fallback: treat as-is
        )
        panel["normalized_consumption"] = panel["per_household_kwh"] * panel["norm_factor"]

        median_kwh = panel.loc[valid_hu, "per_household_kwh"].median()
        logger.info(
            "Per-household consumption: median=%.0f kWh/year across %d state-year rows",
            median_kwh if not np.isnan(median_kwh) else 0,
            valid_hu.sum(),
        )
    else:
        logger.warning(
            "No housing unit data — falling back to state-total consumption. "
            "Results will NOT be per-household. Run census_acs ingester to fix."
        )
        # Fallback: million kWh → kWh (still state-total, not per-household)
        panel["normalized_consumption"] = (
            panel["electricity_consumption_mwh"] * 1_000_000 * panel["norm_factor"]
        )

    panel["electricity_price"] = panel["electricity_price_cents_kwh"]

    logger.info(
        "Panel constructed: %d state-year rows, %d states, years %d-%d",
        len(panel),
        panel["state_fips"].nunique(),
        panel["year"].min(),
        panel["year"].max(),
    )

    return panel


def _detect_all_structural_breaks(
    panel: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Detect structural breaks for all states and create dummy variables.

    Returns:
        Tuple of (break_counts dict, panel with break dummy columns added).
    """
    break_counts: dict[str, int] = {}
    # Collect all (state, break_year) pairs first, then build dummies in one shot
    all_breaks: list[tuple[str, int]] = []

    for state in sorted(panel["state_fips"].unique()):
        state_data = panel[panel["state_fips"] == state].sort_values("year")
        price_series = state_data.set_index("year")["electricity_price"]
        breaks = detect_structural_breaks(price_series)
        break_counts[state] = len(breaks)

        if breaks:
            logger.info("State %s: %d structural breaks at years %s", state, len(breaks), breaks)

        for break_year in breaks:
            all_breaks.append((state, break_year))

    # Build all break dummy columns at once to avoid DataFrame fragmentation
    if all_breaks:
        dummy_dict: dict[str, np.ndarray] = {}
        for state, break_year in all_breaks:
            col_name = f"break_{state}_{break_year}"
            dummy_dict[col_name] = (
                (panel["state_fips"].values == state)
                & (panel["year"].values == break_year)
            ).astype(int)
        panel = pd.concat([panel, pd.DataFrame(dummy_dict, index=panel.index)], axis=1)

    total_breaks = sum(break_counts.values())
    logger.info(
        "Structural break detection: %d total breaks across %d states",
        total_breaks, len(break_counts),
    )

    return break_counts, panel


def _run_panel_regression(
    panel: pd.DataFrame,
    break_dummies: pd.DataFrame,
) -> tuple[float, float, float]:
    """Run panel OLS regression with state and year fixed effects.

    Reports within-R² (variation explained after absorbing fixed effects)
    rather than overall R², which is inflated by state/year FE absorbing
    cross-sectional heterogeneity and is misleading for FE panel models.

    Returns:
        Tuple of (beta_hdd, beta_cdd, within_r_squared).
    """
    df = break_dummies.copy()

    # Build design matrix
    # Independent variables: hdd_anomaly, cdd_anomaly
    X_vars = ["hdd_anomaly", "cdd_anomaly"]

    # State fixed effects (dummies)
    states = sorted(df["state_fips"].unique())
    if len(states) > 1:
        # Drop first state for identification
        for state in states[1:]:
            col = f"fe_state_{state}"
            df[col] = (df["state_fips"] == state).astype(int)
            X_vars.append(col)

    # Year fixed effects (dummies)
    years = sorted(df["year"].unique())
    if len(years) > 1:
        # Drop first year for identification
        for yr in years[1:]:
            col = f"fe_year_{yr}"
            df[col] = (df["year"] == yr).astype(int)
            X_vars.append(col)

    # Add structural break dummies
    break_cols = [c for c in df.columns if c.startswith("break_")]
    X_vars.extend(break_cols)

    # Build X and y
    X = df[X_vars].astype(float)
    X = sm.add_constant(X)
    y = df["normalized_consumption"].astype(float)

    # Fit OLS
    model = sm.OLS(y, X).fit()

    beta_hdd = model.params.get("hdd_anomaly", 0.0)
    beta_cdd = model.params.get("cdd_anomaly", 0.0)

    # Compute within-R²: demean y by state, regress on climate vars only.
    # This measures how well HDD/CDD anomalies explain within-state
    # year-to-year consumption variation — the quantity that matters for
    # climate attribution.  Overall R² (here: %.4f) is dominated by state
    # FE absorbing cross-sectional differences in baseline consumption.
    y_demeaned = y - df.groupby("state_fips")["normalized_consumption"].transform("mean")
    X_climate = df[["hdd_anomaly", "cdd_anomaly"]].astype(float)
    X_climate_demeaned = X_climate - df.groupby("state_fips")[["hdd_anomaly", "cdd_anomaly"]].transform("mean")
    X_climate_demeaned = sm.add_constant(X_climate_demeaned)
    within_model = sm.OLS(y_demeaned, X_climate_demeaned).fit()
    within_r_squared = within_model.rsquared

    logger.info(
        "Panel regression results: beta_hdd=%.4f, beta_cdd=%.4f, "
        "overall_R²=%.4f, within_R²=%.4f (reported)",
        beta_hdd, beta_cdd, model.rsquared, within_r_squared,
    )

    # Validate coefficient signs
    if beta_hdd < 0:
        logger.warning(
            "beta_hdd is negative (%.4f) — more HDD anomaly should increase consumption. "
            "This may indicate data issues.",
            beta_hdd,
        )
    if beta_cdd < 0:
        logger.warning(
            "beta_cdd is negative (%.4f) — more CDD anomaly should increase consumption. "
            "This may indicate data issues.",
            beta_cdd,
        )

    return beta_hdd, beta_cdd, within_r_squared


def _compute_climate_costs(
    panel: pd.DataFrame,
    beta_hdd: float,
    beta_cdd: float,
    r_squared: float,
    break_counts: dict[str, int],
) -> pd.DataFrame:
    """Compute climate-attributed energy cost for each state-year."""
    df = panel[["state_fips", "year", "hdd_anomaly", "cdd_anomaly",
                "electricity_price", "normalized_consumption"]].copy()

    # Climate-attributed consumption (kWh)
    df["climate_attributed_consumption"] = (
        beta_hdd * df["hdd_anomaly"] + beta_cdd * df["cdd_anomaly"]
    )

    # Convert to dollars: consumption (kWh) * price (cents/kWh) / 100
    df["climate_attributed_energy_cost"] = (
        df["climate_attributed_consumption"] * df["electricity_price"] / 100.0
    )

    # Total energy cost in dollars
    df["total_energy_cost"] = (
        df["normalized_consumption"] * df["electricity_price"] / 100.0
    )

    # Attribution fraction
    df["attribution_fraction"] = np.where(
        df["total_energy_cost"] != 0,
        df["climate_attributed_energy_cost"] / df["total_energy_cost"],
        0.0,
    )

    df["regression_r_squared"] = r_squared
    df["structural_breaks_detected"] = df["state_fips"].map(break_counts).fillna(0).astype(int)

    return df[["state_fips", "year", "climate_attributed_energy_cost", "total_energy_cost",
               "attribution_fraction", "regression_r_squared", "structural_breaks_detected"]]


def _map_to_counties(
    state_results: pd.DataFrame,
    degree_days: pd.DataFrame,
) -> pd.DataFrame:
    """Map state-level attributed costs to county level.

    Uses county FIPS from degree_days data to identify counties per state.
    """
    # Get unique county-year combinations from degree_days
    dd = degree_days[["fips", "year"]].copy()
    dd["fips"] = dd["fips"].astype(str).str.zfill(5)
    dd["state_fips"] = dd["fips"].str[:2]
    county_state = dd[["fips", "year", "state_fips"]].drop_duplicates()

    # Merge state results to counties
    merged = county_state.merge(state_results, on=["state_fips", "year"], how="inner")

    # Log unmapped counties
    all_counties = set(county_state[["fips", "year"]].itertuples(index=False, name=None))
    mapped_counties = set(merged[["fips", "year"]].itertuples(index=False, name=None))
    unmapped = all_counties - mapped_counties
    if unmapped:
        unmapped_fips = {f for f, _ in unmapped}
        logger.warning(
            "%d county-year rows could not be mapped to state results (%d unique counties)",
            len(unmapped), len(unmapped_fips),
        )

    if merged.empty:
        logger.warning("No counties mapped to state results — returning empty output.")
        return _empty_output()

    # Enforce output schema
    result = merged[OUTPUT_COLUMNS].copy()

    # Enforce types
    result["fips"] = result["fips"].astype(str)
    result["year"] = result["year"].astype(int)
    result["structural_breaks_detected"] = result["structural_breaks_detected"].astype(int)

    logger.info(
        "Mapped state results to %d county-year rows across %d counties",
        len(result), result["fips"].nunique(),
    )

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS).astype({
        "fips": str,
        "year": int,
        "climate_attributed_energy_cost": float,
        "total_energy_cost": float,
        "attribution_fraction": float,
        "regression_r_squared": float,
        "structural_breaks_detected": int,
    })


def _write_metadata(path: Path, year: int) -> None:
    """Write metadata JSON sidecar alongside the parquet output."""
    meta = {
        "source": METADATA_SOURCE,
        "confidence": METADATA_CONFIDENCE,
        "attribution": METADATA_ATTRIBUTION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "data_vintage": str(year),
        "description": (
            f"County-level climate-attributed energy costs for {year}. "
            f"Panel regression isolating climate-driven consumption anomalies "
            f"from rate-case structural breaks. "
            f"Baseline dwelling: {BASELINE_SQFT} sqft, {BASELINE_DWELLING_TYPE}, "
            f"{BASELINE_OCCUPANTS} occupants."
        ),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
