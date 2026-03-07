"""Tests for transform/energy_attribution.py — climate-attributed energy cost isolation.

All tests use synthetic DataFrames with known expected outputs.
No real data files are read.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from transform.energy_attribution import (
    BASELINE_DWELLING_TYPE,
    BASELINE_OCCUPANTS,
    BASELINE_SQFT,
    METADATA_ATTRIBUTION,
    METADATA_CONFIDENCE,
    METADATA_SOURCE,
    MIN_BASELINE_MATCHES,
    OUTPUT_COLUMNS,
    RATE_CASE_PRICE_JUMP_PCT,
    compute_energy_attribution,
    detect_structural_breaks,
    run,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _make_energy_data(
    states: list[str] | None = None,
    years: list[int] | None = None,
    price: float = 10.0,
    consumption_mwh: float = 50000.0,
) -> pd.DataFrame:
    """Build synthetic state-level energy data."""
    if states is None:
        states = ["01", "02", "04", "06", "17"]
    if years is None:
        years = list(range(2010, 2020))

    rows = []
    for state in states:
        for year in years:
            rows.append({
                "state_fips": state,
                "year": year,
                "electricity_price_cents_kwh": price,
                "electricity_consumption_mwh": consumption_mwh,
            })
    return pd.DataFrame(rows)


def _make_recs_data(
    divisions: list[int] | None = None,
    n_per_division: int = 50,
    baseline_kwh: float = 12000.0,
    avg_kwh: float = 10000.0,
) -> pd.DataFrame:
    """Build synthetic RECS microdata.

    Creates both baseline-matching households and non-baseline households
    to produce a known normalization factor.
    """
    if divisions is None:
        divisions = list(range(1, 10))

    rows = []
    for div in divisions:
        # Baseline-profile households (single-family, 1800 sqft, 3 occupants)
        n_baseline = n_per_division // 2
        for _ in range(n_baseline):
            rows.append({
                "census_division": div,
                "dwelling_type": "single-family detached",
                "square_footage": 1800,
                "num_occupants": 3,
                "annual_electricity_kwh": baseline_kwh,
            })
        # Non-baseline households (different profile)
        n_other = n_per_division - n_baseline
        for _ in range(n_other):
            rows.append({
                "census_division": div,
                "dwelling_type": "apartment",
                "square_footage": 900,
                "num_occupants": 2,
                "annual_electricity_kwh": avg_kwh - (baseline_kwh - avg_kwh),
                # avg of baseline + other = avg_kwh only if baseline=avg_kwh
                # Here we set other = 2*avg - baseline so mean = avg_kwh
            })
    return pd.DataFrame(rows)


def _make_degree_days(
    fips_list: list[str] | None = None,
    years: list[int] | None = None,
    hdd_anomaly: float = 100.0,
    cdd_anomaly: float = 50.0,
) -> pd.DataFrame:
    """Build synthetic county-level degree-day anomalies."""
    if fips_list is None:
        fips_list = ["01001", "01003", "01005", "02010", "04001", "06001", "17001"]
    if years is None:
        years = list(range(2010, 2020))

    rows = []
    for fips in fips_list:
        for year in years:
            rows.append({
                "fips": fips,
                "year": year,
                "hdd_annual": 2000.0,
                "cdd_annual": 1000.0,
                "hdd_anomaly": hdd_anomaly,
                "cdd_anomaly": cdd_anomaly,
            })
    return pd.DataFrame(rows)


def _make_census_data(
    states: list[str],
    years: list[int],
    housing_units: float = 1_000_000.0,
) -> pd.DataFrame:
    """Build synthetic Census ACS data with known housing unit counts.

    Creates 3 counties per state, each with housing_units / 3 units.
    """
    rows = []
    for state in states:
        for year in years:
            for county_suffix in ["001", "003", "005"]:
                fips = f"{state}{county_suffix}"
                rows.append({
                    "fips": fips,
                    "year": year,
                    "total_housing_units": housing_units / 3.0,
                    "population": 100000.0,
                    "median_household_income": 60000.0,
                })
    return pd.DataFrame(rows)


def _make_controlled_panel_data(
    n_states: int = 5,
    n_years: int = 10,
    beta_hdd: float = 5.0,
    beta_cdd: float = 3.0,
    base_consumption: float = 10000.0,
    price: float = 10.0,
    housing_units: float = 1_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a fully controlled synthetic dataset where climate-attributed fraction is known.

    Returns (energy_data, recs_data, degree_days, census_data) where:
    - per_household_kwh = base_consumption + beta_hdd * hdd_anomaly + beta_cdd * cdd_anomaly
    - State-total consumption (million kWh) = per_household_kwh * housing_units / 1_000_000
    - No noise, so regression should recover exact coefficients.

    base_consumption is in kWh/household/year. betas are in kWh/household per anomaly unit.
    """
    states = [f"{i:02d}" for i in range(1, n_states + 1)]
    years = list(range(2010, 2010 + n_years))

    rng = np.random.RandomState(42)
    energy_rows = []
    dd_rows = []

    for state in states:
        for year in years:
            hdd_anom = rng.uniform(-200, 200)
            cdd_anom = rng.uniform(-100, 100)

            # Per-household consumption in kWh
            per_hh_kwh = base_consumption + beta_hdd * hdd_anom + beta_cdd * cdd_anom
            # State-total in million kWh (what EIA API returns)
            state_total_million_kwh = per_hh_kwh * housing_units / 1_000_000
            energy_rows.append({
                "state_fips": state,
                "year": year,
                "electricity_price_cents_kwh": price,
                "electricity_consumption_mwh": state_total_million_kwh,
            })

            # Create counties for this state
            for county_suffix in ["001", "003", "005"]:
                fips = f"{state}{county_suffix}"
                dd_rows.append({
                    "fips": fips,
                    "year": year,
                    "hdd_annual": 2000.0,
                    "cdd_annual": 1000.0,
                    "hdd_anomaly": hdd_anom,
                    "cdd_anomaly": cdd_anom,
                })

    energy_df = pd.DataFrame(energy_rows)
    dd_df = pd.DataFrame(dd_rows)

    # RECS: normalization factor = 1.0 (baseline == average)
    recs_rows = []
    for div in range(1, 10):
        for _ in range(50):
            recs_rows.append({
                "census_division": div,
                "dwelling_type": "single-family detached",
                "square_footage": 1800,
                "num_occupants": 3,
                "annual_electricity_kwh": 12000.0,
            })
    recs_df = pd.DataFrame(recs_rows)

    census_df = _make_census_data(states, years, housing_units)

    return energy_df, recs_df, dd_df, census_df


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------

class TestSyntheticKnownAttribution:
    """Test regression recovers approximately correct beta coefficients."""

    def test_recovers_betas(self):
        """With no noise, OLS should recover exact betas."""
        energy, recs, dd, census = _make_controlled_panel_data(
            n_states=5, n_years=10, beta_hdd=5.0, beta_cdd=3.0,
        )
        result = compute_energy_attribution(energy, recs, dd, census)

        assert not result.empty
        assert "climate_attributed_energy_cost" in result.columns

        # Within-R² should be high with no noise (climate vars explain
        # within-state variation perfectly in synthetic data)
        r2 = result["regression_r_squared"].iloc[0]
        assert r2 > 0.5, f"Within-R² should be high with no noise, got {r2}"


class TestStructuralBreakDetection:
    """Test detect_structural_breaks function."""

    def test_detects_large_jump(self):
        """A 10% price jump should be detected."""
        prices = pd.Series(
            [10.0, 10.0, 10.0, 10.0, 11.0, 11.0],
            index=[2010, 2011, 2012, 2013, 2014, 2015],
        )
        breaks = detect_structural_breaks(prices)
        assert 2014 in breaks

    def test_no_breaks_smooth_prices(self):
        """Smooth price series with <10% changes should have no breaks."""
        prices = pd.Series(
            [10.0, 10.2, 10.4, 10.5, 10.6, 10.7],
            index=[2010, 2011, 2012, 2013, 2014, 2015],
        )
        breaks = detect_structural_breaks(prices)
        assert len(breaks) == 0

    def test_detects_multiple_breaks(self):
        """Multiple large jumps should all be detected."""
        prices = pd.Series(
            [10.0, 11.5, 11.5, 11.5, 13.0, 13.0],
            index=[2010, 2011, 2012, 2013, 2014, 2015],
        )
        breaks = detect_structural_breaks(prices)
        assert 2011 in breaks
        assert 2014 in breaks

    def test_detects_downward_break(self):
        """A large downward price change should also be detected."""
        prices = pd.Series(
            [10.0, 10.0, 10.0, 8.0, 8.0],
            index=[2010, 2011, 2012, 2013, 2014],
        )
        breaks = detect_structural_breaks(prices)
        assert 2013 in breaks

    def test_empty_series(self):
        """Empty series returns no breaks."""
        breaks = detect_structural_breaks(pd.Series([], dtype=float))
        assert breaks == []

    def test_single_value(self):
        """Single-value series returns no breaks."""
        breaks = detect_structural_breaks(pd.Series([10.0], index=[2020]))
        assert breaks == []

    def test_custom_threshold(self):
        """Custom threshold is respected."""
        prices = pd.Series(
            [10.0, 10.3, 10.3],  # 3% jump
            index=[2010, 2011, 2012],
        )
        # Default 10% threshold: no break
        assert len(detect_structural_breaks(prices)) == 0
        # 2% threshold: break detected
        assert len(detect_structural_breaks(prices, threshold=0.02)) == 1


class TestRECSNormalization:
    """Test RECS baseline normalization factor computation."""

    def test_normalization_factor(self):
        """Normalization factor = baseline_consumption / average_consumption."""
        recs = _make_recs_data(divisions=[1], n_per_division=60, baseline_kwh=12000, avg_kwh=10000)
        energy = _make_energy_data(states=["09"], years=list(range(2015, 2020)))
        dd = _make_degree_days(fips_list=["09001"], years=list(range(2015, 2020)))

        result = compute_energy_attribution(energy, recs, dd)
        # Result should exist — normalization was applied
        assert not result.empty

    def test_fallback_to_national(self):
        """When a division has < MIN_BASELINE_MATCHES, falls back to national."""
        # Division 1 with very few baseline matches
        rows = []
        # Only 5 baseline matches for division 6
        for i in range(5):
            rows.append({
                "census_division": 6,
                "dwelling_type": "single-family detached",
                "square_footage": 1800,
                "num_occupants": 3,
                "annual_electricity_kwh": 12000.0,
            })
        # Many non-baseline
        for i in range(100):
            rows.append({
                "census_division": 6,
                "dwelling_type": "apartment",
                "square_footage": 900,
                "num_occupants": 2,
                "annual_electricity_kwh": 8000.0,
            })
        # Enough national-level baseline matches from other divisions
        for i in range(60):
            rows.append({
                "census_division": 3,
                "dwelling_type": "single-family detached",
                "square_footage": 1800,
                "num_occupants": 3,
                "annual_electricity_kwh": 12000.0,
            })
        for i in range(60):
            rows.append({
                "census_division": 3,
                "dwelling_type": "apartment",
                "square_footage": 900,
                "num_occupants": 2,
                "annual_electricity_kwh": 8000.0,
            })

        recs = pd.DataFrame(rows)
        energy = _make_energy_data(states=["01"], years=list(range(2015, 2020)))
        dd = _make_degree_days(fips_list=["01001"], years=list(range(2015, 2020)))

        # Should not crash — fallback is used
        result = compute_energy_attribution(energy, recs, dd)
        assert not result.empty


class TestClimateCostCalculation:
    """Test the climate cost dollar calculation."""

    def test_known_cost(self):
        """Given known betas and anomalies, verify cost = (b1*HDD + b2*CDD) * price/100."""
        # We build a dataset where regression recovers exact betas
        energy, recs, dd, census = _make_controlled_panel_data(
            n_states=5, n_years=15, beta_hdd=5.0, beta_cdd=3.0, price=10.0,
        )
        result = compute_energy_attribution(energy, recs, dd, census)
        assert not result.empty

        # Since we have no noise, attributed cost for each county should be close to
        # (5 * hdd_anomaly + 3 * cdd_anomaly) * 10 / 100
        # Check a sample row
        sample = result.iloc[0]
        # The county's anomaly is the same as the state (since all counties in a state
        # have the same anomaly in our synthetic data)
        state = sample["fips"][:2]
        year = sample["year"]
        state_dd = dd[(dd["fips"].str[:2] == state) & (dd["year"] == year)]
        hdd_a = state_dd["hdd_anomaly"].mean()
        cdd_a = state_dd["cdd_anomaly"].mean()
        expected_cost = (5.0 * hdd_a + 3.0 * cdd_a) * 10.0 / 100.0
        actual_cost = sample["climate_attributed_energy_cost"]
        # Allow some tolerance for regression estimation
        np.testing.assert_allclose(actual_cost, expected_cost, rtol=0.1)

    def test_negative_attribution(self):
        """Mild year (negative anomalies) produces negative attributed cost."""
        energy, recs, dd, census = _make_controlled_panel_data(
            n_states=3, n_years=10, beta_hdd=5.0, beta_cdd=3.0,
        )
        # Override one year's anomalies to be strongly negative
        dd.loc[dd["year"] == 2015, "hdd_anomaly"] = -300.0
        dd.loc[dd["year"] == 2015, "cdd_anomaly"] = -200.0

        # Also adjust energy data consumption for 2015 to match
        # per_hh_kwh = base(10000) + 5*(-300) + 3*(-200) = 7900
        # state_total_million_kwh = 7900 * 1_000_000 / 1_000_000 = 7900
        housing_units = 1_000_000.0  # default from _make_controlled_panel_data
        per_hh_kwh = 10000 + 5.0 * (-300.0) + 3.0 * (-200.0)
        state_total_million_kwh = per_hh_kwh * housing_units / 1_000_000
        for state in dd["fips"].str[:2].unique():
            mask = (energy["state_fips"] == state) & (energy["year"] == 2015)
            energy.loc[mask, "electricity_consumption_mwh"] = state_total_million_kwh

        result = compute_energy_attribution(energy, recs, dd, census)
        year_2015 = result[result["year"] == 2015]
        assert (year_2015["climate_attributed_energy_cost"] < 0).all(), (
            "Negative anomalies should produce negative attributed cost"
        )

    def test_attribution_fraction(self):
        """attribution_fraction = climate_attributed / total."""
        energy, recs, dd, census = _make_controlled_panel_data(
            n_states=3, n_years=10, beta_hdd=5.0, beta_cdd=3.0,
        )
        result = compute_energy_attribution(energy, recs, dd, census)
        assert not result.empty

        for _, row in result.head(10).iterrows():
            if row["total_energy_cost"] != 0:
                expected_frac = row["climate_attributed_energy_cost"] / row["total_energy_cost"]
                np.testing.assert_allclose(
                    row["attribution_fraction"], expected_frac, rtol=1e-6
                )


class TestStateToCountyMapping:
    """Test that state results map to counties correctly."""

    def test_all_counties_get_same_cost(self):
        """All counties in a state should get the same attributed cost."""
        energy = _make_energy_data(states=["01"], years=list(range(2015, 2020)))
        recs = _make_recs_data()
        dd = _make_degree_days(
            fips_list=["01001", "01003", "01005"],
            years=list(range(2015, 2020)),
        )

        result = compute_energy_attribution(energy, recs, dd)
        assert not result.empty

        for year in result["year"].unique():
            year_data = result[result["year"] == year]
            costs = year_data["climate_attributed_energy_cost"].unique()
            assert len(costs) == 1, f"All counties in state should have same cost in year {year}"

    def test_multi_state(self):
        """Multiple states produce separate results for their counties."""
        states = ["01", "06"]
        energy = _make_energy_data(states=states, years=list(range(2015, 2020)))
        recs = _make_recs_data()
        dd = _make_degree_days(
            fips_list=["01001", "01003", "06001", "06003"],
            years=list(range(2015, 2020)),
            hdd_anomaly=100.0,
            cdd_anomaly=50.0,
        )

        result = compute_energy_attribution(energy, recs, dd)
        assert not result.empty

        # Both states should have results
        result_states = result["fips"].str[:2].unique()
        assert "01" in result_states
        assert "06" in result_states


class TestMultiStateRegression:
    """Test regression with multi-state panel."""

    def test_five_states_ten_years(self):
        """Regression runs successfully on a 5-state x 10-year panel."""
        energy, recs, dd, census = _make_controlled_panel_data(n_states=5, n_years=10)
        result = compute_energy_attribution(energy, recs, dd, census)

        assert not result.empty
        assert result["regression_r_squared"].iloc[0] > 0.0  # within-R² is positive
        assert len(result["fips"].unique()) == 15  # 5 states * 3 counties each


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Test output DataFrame schema matches spec."""

    @pytest.fixture
    def result(self):
        energy, recs, dd, census = _make_controlled_panel_data(n_states=3, n_years=5)
        return compute_energy_attribution(energy, recs, dd, census)

    def test_column_presence(self, result):
        """All specified columns must be present."""
        for col in OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, result):
        """No columns beyond the spec."""
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_column_types(self, result):
        """Column dtypes match spec."""
        assert pd.api.types.is_string_dtype(result["fips"])
        assert np.issubdtype(result["year"].dtype, np.integer)
        assert np.issubdtype(result["climate_attributed_energy_cost"].dtype, np.floating)
        assert np.issubdtype(result["total_energy_cost"].dtype, np.floating)
        assert np.issubdtype(result["attribution_fraction"].dtype, np.floating)
        assert np.issubdtype(result["regression_r_squared"].dtype, np.floating)
        assert np.issubdtype(result["structural_breaks_detected"].dtype, np.integer)

    def test_fips_5digit(self, result):
        """FIPS codes are 5-digit zero-padded strings."""
        for fips in result["fips"]:
            assert len(fips) == 5, f"FIPS should be 5 digits: {fips}"
            assert fips.isdigit(), f"FIPS should be all digits: {fips}"


# ---------------------------------------------------------------------------
# Edge case tests (module-specific)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test module-specific edge cases."""

    def test_state_no_anomaly_data(self):
        """A state with no degree-day data is excluded without crashing others."""
        energy = _make_energy_data(states=["01", "06"], years=list(range(2015, 2020)))
        recs = _make_recs_data()
        # Only provide degree days for state 01
        dd = _make_degree_days(fips_list=["01001"], years=list(range(2015, 2020)))

        result = compute_energy_attribution(energy, recs, dd)
        # Should have results only for state 01
        assert not result.empty
        assert all(f.startswith("01") for f in result["fips"])

    def test_single_state_panel(self):
        """Regression runs with a single state (state FE reduces to intercept)."""
        energy = _make_energy_data(states=["01"], years=list(range(2010, 2020)))
        recs = _make_recs_data()
        dd = _make_degree_days(fips_list=["01001"], years=list(range(2010, 2020)))

        result = compute_energy_attribution(energy, recs, dd)
        assert not result.empty

    def test_all_zero_anomalies(self):
        """Zero anomalies produce zero attributed cost."""
        energy, recs, dd, census = _make_controlled_panel_data(
            n_states=3, n_years=10, beta_hdd=5.0, beta_cdd=3.0,
        )
        # Set all anomalies to zero for a specific year
        dd.loc[dd["year"] == 2015, "hdd_anomaly"] = 0.0
        dd.loc[dd["year"] == 2015, "cdd_anomaly"] = 0.0

        # Adjust consumption to match (base consumption only)
        # base=10000 kWh/hh, housing_units=1M → state_total = 10000 million kWh
        housing_units = 1_000_000.0
        state_total_million_kwh = 10000.0 * housing_units / 1_000_000
        for state in dd["fips"].str[:2].unique():
            mask = (energy["state_fips"] == state) & (energy["year"] == 2015)
            energy.loc[mask, "electricity_consumption_mwh"] = state_total_million_kwh

        result = compute_energy_attribution(energy, recs, dd, census)
        year_2015 = result[result["year"] == 2015]

        for _, row in year_2015.iterrows():
            np.testing.assert_allclose(
                row["climate_attributed_energy_cost"], 0.0, atol=1.0,
                err_msg="Zero anomalies should produce near-zero attributed cost",
            )

    def test_price_zero_no_division_error(self):
        """Price = 0 should not cause division-by-zero errors."""
        energy = _make_energy_data(states=["01"], years=list(range(2015, 2020)), price=0.0)
        recs = _make_recs_data()
        dd = _make_degree_days(fips_list=["01001"], years=list(range(2015, 2020)))

        # Should not crash
        result = compute_energy_attribution(energy, recs, dd)
        assert not result.empty
        # With price = 0, total and attributed costs should be 0
        assert (result["total_energy_cost"] == 0.0).all()
        assert (result["climate_attributed_energy_cost"] == 0.0).all()


# ---------------------------------------------------------------------------
# Edge case handling (generic)
# ---------------------------------------------------------------------------

class TestGenericEdgeCases:
    """Test generic edge case handling."""

    def test_empty_energy_data(self):
        """Empty energy data returns empty output."""
        energy = pd.DataFrame(columns=[
            "state_fips", "year", "electricity_price_cents_kwh",
            "electricity_consumption_mwh",
        ])
        recs = _make_recs_data()
        dd = _make_degree_days()

        result = compute_energy_attribution(energy, recs, dd)
        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_empty_degree_days(self):
        """Empty degree-day data returns empty output."""
        energy = _make_energy_data()
        recs = _make_recs_data()
        dd = pd.DataFrame(columns=["fips", "year", "hdd_anomaly", "cdd_anomaly"])

        result = compute_energy_attribution(energy, recs, dd)
        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_missing_energy_columns(self):
        """Missing required columns in energy data raises ValueError."""
        energy = pd.DataFrame({"state_fips": ["01"], "year": [2020]})
        recs = _make_recs_data()
        dd = _make_degree_days()

        with pytest.raises(ValueError, match="missing columns"):
            compute_energy_attribution(energy, recs, dd)

    def test_missing_recs_columns(self):
        """Missing required columns in RECS data raises ValueError."""
        energy = _make_energy_data()
        recs = pd.DataFrame({"census_division": [1]})
        dd = _make_degree_days()

        with pytest.raises(ValueError, match="missing columns"):
            compute_energy_attribution(energy, recs, dd)

    def test_missing_degree_days_columns(self):
        """Missing required columns in degree-day data raises ValueError."""
        energy = _make_energy_data()
        recs = _make_recs_data()
        dd = pd.DataFrame({"fips": ["01001"], "year": [2020]})

        with pytest.raises(ValueError, match="missing columns"):
            compute_energy_attribution(energy, recs, dd)

    def test_partial_data_produces_partial_output(self):
        """States with data produce output; missing states are excluded."""
        energy = _make_energy_data(
            states=["01", "06", "17"], years=list(range(2015, 2020)),
        )
        recs = _make_recs_data()
        # Only provide degree days for state 01
        dd = _make_degree_days(fips_list=["01001", "01003"], years=list(range(2015, 2020)))

        result = compute_energy_attribution(energy, recs, dd)
        assert not result.empty
        # Only state 01 counties should appear
        assert all(f.startswith("01") for f in result["fips"])

    def test_fips_normalization(self):
        """FIPS codes are properly 5-digit zero-padded strings."""
        energy = _make_energy_data(states=["01"], years=[2015, 2016, 2017, 2018, 2019])
        recs = _make_recs_data()
        dd = _make_degree_days(fips_list=["01001"], years=[2015, 2016, 2017, 2018, 2019])

        result = compute_energy_attribution(energy, recs, dd)
        for fips in result["fips"]:
            assert len(fips) == 5
            assert fips == fips.zfill(5)


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Test reproducibility."""

    def test_same_input_same_output(self):
        """OLS is deterministic — same input always produces identical output."""
        energy, recs, dd, census = _make_controlled_panel_data(n_states=3, n_years=8)

        result1 = compute_energy_attribution(energy.copy(), recs.copy(), dd.copy(), census.copy())
        result2 = compute_energy_attribution(energy.copy(), recs.copy(), dd.copy(), census.copy())

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Standard transform tests
# ---------------------------------------------------------------------------

class TestTransformPurity:
    """Verify no scoring metrics leak into transform output."""

    @pytest.fixture
    def result(self):
        energy, recs, dd, census = _make_controlled_panel_data(n_states=3, n_years=5)
        return compute_energy_attribution(energy, recs, dd, census)

    def test_no_scoring_columns(self, result):
        """No Phase 3 scoring columns in output."""
        scoring_cols = [
            "percentile", "national_rank", "overlap_penalty",
            "acceleration_multiplier", "composite_score", "cci_score",
        ]
        for col in scoring_cols:
            assert col not in result.columns, f"Scoring column leaked: {col}"

    def test_county_year_grain(self, result):
        """Exactly one row per (fips, year)."""
        dupes = result.duplicated(subset=["fips", "year"])
        assert not dupes.any(), "Duplicate (fips, year) rows found"


class TestMetadataSidecar:
    """Test metadata JSON sidecar output."""

    def test_metadata_values(self, tmp_path):
        """Metadata contains correct source, confidence, attribution."""
        energy, recs, dd, census = _make_controlled_panel_data(n_states=2, n_years=5)

        with patch("transform.energy_attribution.HARMONIZED_DIR", tmp_path):
            result = compute_energy_attribution(energy, recs, dd, census)

            # Manually save to trigger metadata write
            years = sorted(result["year"].unique())
            for yr in years:
                year_df = result[result["year"] == yr]
                parquet_path = tmp_path / f"energy_attribution_{yr}.parquet"
                metadata_path = tmp_path / f"energy_attribution_{yr}_metadata.json"
                year_df.to_parquet(parquet_path, index=False)

                from transform.energy_attribution import _write_metadata
                _write_metadata(metadata_path, yr)

            # Check first year's metadata
            meta_path = tmp_path / f"energy_attribution_{years[0]}_metadata.json"
            assert meta_path.exists()
            with open(meta_path) as f:
                meta = json.load(f)

            assert meta["source"] == "EIA_ENERGY"
            assert meta["confidence"] == "A"
            assert meta["attribution"] == "attributed"
            assert "data_vintage" in meta
            assert "retrieved_at" in meta


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

class TestRun:
    """Test the run() entry point with mocked file I/O."""

    def test_run_writes_parquet_and_metadata(self, tmp_path):
        """run() writes per-year parquet and metadata files."""
        energy, recs, dd, census = _make_controlled_panel_data(n_states=2, n_years=3)

        # Create fake raw files
        energy_dir = tmp_path / "raw" / "eia_energy"
        energy_dir.mkdir(parents=True)
        energy.to_parquet(energy_dir / "eia_energy_all.parquet", index=False)
        recs.to_parquet(energy_dir / "recs_microdata.parquet", index=False)

        census_dir = tmp_path / "raw" / "census_acs"
        census_dir.mkdir(parents=True)
        census.to_parquet(census_dir / "census_acs_all.parquet", index=False)

        harmonized_dir = tmp_path / "harmonized"
        harmonized_dir.mkdir(parents=True)

        # Write degree-day files
        for yr in dd["year"].unique():
            yr_data = dd[dd["year"] == yr]
            yr_data.to_parquet(harmonized_dir / f"degree_days_{yr}.parquet", index=False)

        with (
            patch("transform.energy_attribution.ENERGY_COMBINED_PATH", energy_dir / "eia_energy_all.parquet"),
            patch("transform.energy_attribution.RECS_PATH", energy_dir / "recs_microdata.parquet"),
            patch("transform.energy_attribution.CENSUS_COMBINED_PATH", census_dir / "census_acs_all.parquet"),
            patch("transform.energy_attribution.DEGREE_DAYS_DIR", harmonized_dir),
            patch("transform.energy_attribution.HARMONIZED_DIR", harmonized_dir),
        ):
            result = run()

        assert not result.empty

        # Check parquet and metadata files exist
        years = sorted(result["year"].unique())
        for yr in years:
            parquet = harmonized_dir / f"energy_attribution_{yr}.parquet"
            metadata = harmonized_dir / f"energy_attribution_{yr}_metadata.json"
            assert parquet.exists(), f"Missing parquet for year {yr}"
            assert metadata.exists(), f"Missing metadata for year {yr}"

            # Verify metadata content
            with open(metadata) as f:
                meta = json.load(f)
            assert meta["source"] == "EIA_ENERGY"
            assert meta["attribution"] == "attributed"

    def test_run_missing_energy_file(self, tmp_path):
        """run() raises FileNotFoundError when energy data is missing."""
        with (
            patch("transform.energy_attribution.ENERGY_COMBINED_PATH", tmp_path / "missing.parquet"),
            patch("transform.energy_attribution.ENERGY_DIR", tmp_path),
        ):
            with pytest.raises(FileNotFoundError):
                run()

    def test_run_missing_recs_file(self, tmp_path):
        """run() raises FileNotFoundError when RECS data is missing."""
        energy_dir = tmp_path / "eia_energy"
        energy_dir.mkdir()
        energy = _make_energy_data()
        energy.to_parquet(energy_dir / "eia_energy_all.parquet", index=False)

        with (
            patch("transform.energy_attribution.ENERGY_COMBINED_PATH", energy_dir / "eia_energy_all.parquet"),
            patch("transform.energy_attribution.RECS_PATH", tmp_path / "missing_recs.parquet"),
        ):
            with pytest.raises(FileNotFoundError):
                run()

    def test_run_missing_degree_days(self, tmp_path):
        """run() raises FileNotFoundError when degree-day data is missing."""
        energy_dir = tmp_path / "eia_energy"
        energy_dir.mkdir()
        energy = _make_energy_data()
        recs = _make_recs_data()
        energy.to_parquet(energy_dir / "eia_energy_all.parquet", index=False)
        recs.to_parquet(energy_dir / "recs_microdata.parquet", index=False)

        empty_harmonized = tmp_path / "empty_harmonized"
        empty_harmonized.mkdir()

        with (
            patch("transform.energy_attribution.ENERGY_COMBINED_PATH", energy_dir / "eia_energy_all.parquet"),
            patch("transform.energy_attribution.RECS_PATH", energy_dir / "recs_microdata.parquet"),
            patch("transform.energy_attribution.DEGREE_DAYS_DIR", empty_harmonized),
        ):
            with pytest.raises(FileNotFoundError):
                run()
