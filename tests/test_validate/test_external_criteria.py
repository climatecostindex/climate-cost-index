"""Tests for validate/external_criteria.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validate.external_criteria import run_external_validation


EXPECTED_COLUMNS = {
    "indicator", "geographic_level", "pearson_r", "spearman_r",
    "p_value_pearson", "p_value_spearman", "n_observations",
    "expected_r_low", "expected_r_high", "within_expected", "status", "note",
}


def _make_scored(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "cci_score": rng.normal(100, 15, n),
    }, index=pd.Index(fips, name="fips"))


def _make_harmonized(n: int = 100, years: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips_list = [f"{i:05d}" for i in range(1, n + 1)]
    rows = []
    for yr in range(2020, 2020 + years):
        for fips in fips_list:
            rows.append({
                "fips": fips,
                "year": yr,
                "fema_ia_burden": rng.exponential(500),
                "energy_cost_attributed": rng.normal(1200, 300),
            })
    return pd.DataFrame(rows)


class TestKnownCorrelation:
    def test_fema_ia_returns_valid_correlation(self):
        scored = _make_scored(200)
        harmonized = _make_harmonized(200, years=1)
        harmonized["year"] = 2024

        result = run_external_validation(scored, harmonized)
        fema_row = result[result["indicator"] == "fema_ia_per_household"].iloc[0]
        assert fema_row["status"] == "success"
        assert fema_row["n_observations"] > 0
        assert -1 <= fema_row["pearson_r"] <= 1
        assert 0 < fema_row["p_value_pearson"] <= 1

    def test_known_correlation_magnitude(self):
        """Synthetic data with known correlation ~0.5."""
        rng = np.random.RandomState(99)
        n = 500
        fips = [f"{i:05d}" for i in range(1, n + 1)]
        x = rng.normal(100, 15, n)
        noise = rng.normal(0, 10, n)
        # y = 0.6*x + noise → r ≈ 0.6*15/sqrt(0.36*225+100) ≈ 0.67
        y = 0.6 * x + noise

        scored = pd.DataFrame({"cci_score": x}, index=pd.Index(fips, name="fips"))
        harmonized = pd.DataFrame({
            "fips": fips, "year": 2024, "fema_ia_burden": y,
            "energy_cost_attributed": rng.normal(1000, 200, n),
        })

        result = run_external_validation(scored, harmonized)
        fema_row = result[result["indicator"] == "fema_ia_per_household"].iloc[0]
        assert fema_row["status"] == "success"
        assert fema_row["pearson_r"] > 0.4  # should be ~0.67


class TestEnergyVolatility:
    def test_computes_volatility(self):
        scored = _make_scored(100)
        harmonized = _make_harmonized(100, years=6)
        harmonized["year"] = harmonized.groupby("fips").cumcount() + 2019

        result = run_external_validation(scored, harmonized)
        energy_row = result[result["indicator"] == "energy_bill_volatility"].iloc[0]
        assert energy_row["status"] == "success"
        assert energy_row["n_observations"] > 0


class TestMissingData:
    def test_unavailable_indicators_no_crash(self):
        scored = _make_scored(50)
        result = run_external_validation(scored, harmonized_df=None)
        for ind in ["insurance_premium_growth", "property_value_volatility",
                     "utility_rate_case_frequency"]:
            row = result[result["indicator"] == ind].iloc[0]
            assert row["status"] == "data_unavailable"

    def test_no_harmonized_marks_fema_and_energy_unavailable(self):
        scored = _make_scored(50)
        result = run_external_validation(scored, harmonized_df=None)
        for ind in ["fema_ia_per_household", "energy_bill_volatility"]:
            row = result[result["indicator"] == ind].iloc[0]
            assert row["status"] == "data_unavailable"


class TestPValues:
    def test_p_values_in_range(self):
        scored = _make_scored(200)
        harmonized = _make_harmonized(200, years=6)
        harmonized["year"] = harmonized.groupby("fips").cumcount() + 2019

        result = run_external_validation(scored, harmonized)
        success_rows = result[result["status"] == "success"]
        for _, row in success_rows.iterrows():
            if row["p_value_pearson"] is not None:
                assert 0 < row["p_value_pearson"] <= 1
            if row["p_value_spearman"] is not None:
                assert 0 < row["p_value_spearman"] <= 1


class TestOutputSchema:
    def test_all_columns_present(self):
        scored = _make_scored(50)
        result = run_external_validation(scored, harmonized_df=None)
        assert EXPECTED_COLUMNS.issubset(set(result.columns))

    def test_six_indicators_returned(self):
        scored = _make_scored(50)
        result = run_external_validation(scored, harmonized_df=None)
        assert len(result) == 6


class TestEmptyInput:
    def test_empty_scored_df(self):
        scored = pd.DataFrame(columns=["cci_score"])
        scored.index.name = "fips"
        result = run_external_validation(scored)
        assert len(result) == 6
        assert all(result["status"].isin(["no_scored_data"]))

    def test_scored_with_fips_column(self):
        """scored_df with fips as column instead of index."""
        rng = np.random.RandomState(42)
        scored = pd.DataFrame({
            "fips": [f"{i:05d}" for i in range(1, 51)],
            "cci_score": rng.normal(100, 15, 50),
        })
        result = run_external_validation(scored, harmonized_df=None)
        assert len(result) == 6
