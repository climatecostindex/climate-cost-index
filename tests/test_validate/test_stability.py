"""Tests for validate/stability.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validate.stability import (
    run_stability_analysis,
    _rank_persistence,
    _decile_turnover,
    _score_band_stability,
    _component_contribution_stability,
)


def _make_multi_year(n: int = 100, years: list[int] | None = None,
                     seed: int = 42) -> pd.DataFrame:
    """Create multi-year scores with stable rankings."""
    rng = np.random.RandomState(seed)
    if years is None:
        years = [2023, 2024]
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    rows = []
    base_scores = rng.normal(100, 15, n)
    for yr in years:
        noise = rng.normal(0, 2, n)  # small noise → high stability
        for i, f in enumerate(fips):
            rows.append({"fips": f, "year": yr, "cci_score": base_scores[i] + noise[i]})
    return pd.DataFrame(rows)


class TestSingleYear:
    def test_insufficient_years(self):
        df = pd.DataFrame({
            "fips": ["01001", "01002"],
            "year": [2024, 2024],
            "cci_score": [100, 110],
        })
        result = run_stability_analysis(df)
        assert len(result) == 1
        assert result.iloc[0]["status"] == "insufficient_years"

    def test_no_crash(self):
        df = pd.DataFrame({
            "fips": ["01001"],
            "year": [2024],
            "cci_score": [100],
        })
        result = run_stability_analysis(df)
        assert isinstance(result, pd.DataFrame)


class TestIdenticalYears:
    def test_perfect_stability(self):
        n = 100
        fips = [f"{i:05d}" for i in range(1, n + 1)]
        scores = np.linspace(70, 130, n)
        df = pd.DataFrame([
            {"fips": f, "year": 2023, "cci_score": s}
            for f, s in zip(fips, scores)
        ] + [
            {"fips": f, "year": 2024, "cci_score": s}
            for f, s in zip(fips, scores)
        ])

        result = run_stability_analysis(df)
        rp = result[result["metric"] == "rank_persistence"]
        assert len(rp) == 1
        assert rp.iloc[0]["value"] == pytest.approx(1.0, abs=1e-6)
        assert rp.iloc[0]["passes"] is True

    def test_zero_turnover(self):
        n = 100
        fips = [f"{i:05d}" for i in range(1, n + 1)]
        scores = np.linspace(70, 130, n)
        df = pd.DataFrame([
            {"fips": f, "year": 2023, "cci_score": s}
            for f, s in zip(fips, scores)
        ] + [
            {"fips": f, "year": 2024, "cci_score": s}
            for f, s in zip(fips, scores)
        ])

        result = run_stability_analysis(df)
        turnover = result[result["metric"].str.startswith("decile_turnover")]
        for _, row in turnover.iterrows():
            assert row["value"] == 0.0
            assert row["passes"] is True


class TestKnownDecileTurnover:
    def test_30_percent_turnover(self):
        """Swap 3 of top 10 with 3 from middle → 30% turnover."""
        n = 100
        fips = [f"{i:05d}" for i in range(1, n + 1)]
        scores_y1 = np.linspace(70, 130, n)  # sorted ascending
        scores_y2 = scores_y1.copy()

        # Top decile is indices 90-99 (scores ~124-130)
        # Swap indices 97,98,99 with 50,51,52
        scores_y2[97], scores_y2[50] = scores_y2[50], scores_y2[97]
        scores_y2[98], scores_y2[51] = scores_y2[51], scores_y2[98]
        scores_y2[99], scores_y2[52] = scores_y2[52], scores_y2[99]

        df = pd.DataFrame([
            {"fips": f, "year": 2023, "cci_score": s}
            for f, s in zip(fips, scores_y1)
        ] + [
            {"fips": f, "year": 2024, "cci_score": s}
            for f, s in zip(fips, scores_y2)
        ])

        turnover_results = _decile_turnover(df)
        top_turnover = [r for r in turnover_results if r["decile"] == "top"][0]
        assert top_turnover["turnover_pct"] == pytest.approx(30.0, abs=5.0)
        assert top_turnover["passes"] is False  # > 20%


class TestRankPersistence:
    def test_high_correlation(self):
        df = _make_multi_year(200, seed=42)
        rp = _rank_persistence(df)
        assert len(rp) == 1
        assert rp[0]["spearman_r"] > 0.9  # small noise → high r

    def test_confidence_interval(self):
        df = _make_multi_year(200, seed=42)
        rp = _rank_persistence(df)
        r = rp[0]
        assert r["ci_low"] < r["spearman_r"] < r["ci_high"]


class TestScoreBandStability:
    def test_high_stability(self):
        """County stays in same 10-point band for 5 years."""
        fips = "01001"
        df = pd.DataFrame([
            {"fips": fips, "year": 2020 + i, "cci_score": 102 + i}
            for i in range(5)
        ])  # scores 102-106, all in band 100
        result = _score_band_stability(df)
        assert result.iloc[0]["stability_class"] == "high_stability"
        assert result.iloc[0]["years_in_modal_band"] >= 4

    def test_low_stability(self):
        """County oscillates across bands."""
        fips = "01001"
        df = pd.DataFrame([
            {"fips": fips, "year": 2020, "cci_score": 90},
            {"fips": fips, "year": 2021, "cci_score": 110},
            {"fips": fips, "year": 2022, "cci_score": 90},
            {"fips": fips, "year": 2023, "cci_score": 110},
            {"fips": fips, "year": 2024, "cci_score": 130},
        ])
        result = _score_band_stability(df)
        assert result.iloc[0]["stability_class"] == "low_stability"

    def test_insufficient_years(self):
        df = pd.DataFrame([
            {"fips": "01001", "year": 2023, "cci_score": 100},
            {"fips": "01001", "year": 2024, "cci_score": 105},
        ])
        result = _score_band_stability(df)
        assert result.iloc[0].get("status") == "insufficient_years"


class TestComponentContributionStability:
    def test_driver_changed(self):
        """Primary driver changes from energy to storm."""
        df = pd.DataFrame([
            {"fips": "01001", "year": 2023, "energy_cost_attributed": 10, "storm_severity": 2},
            {"fips": "01001", "year": 2024, "energy_cost_attributed": 2, "storm_severity": 10},
        ])
        result = _component_contribution_stability(df)
        assert len(result) == 1
        assert bool(result.iloc[0]["driver_changed"]) is True
        assert result.iloc[0]["primary_driver_year1"] == "energy_cost_attributed"
        assert result.iloc[0]["primary_driver_year2"] == "storm_severity"

    def test_driver_stable(self):
        df = pd.DataFrame([
            {"fips": "01001", "year": 2023, "energy_cost_attributed": 10, "storm_severity": 2},
            {"fips": "01001", "year": 2024, "energy_cost_attributed": 12, "storm_severity": 3},
        ])
        result = _component_contribution_stability(df)
        assert bool(result.iloc[0]["driver_changed"]) is False


class TestMissingColumns:
    def test_missing_required_columns(self):
        df = pd.DataFrame({"fips": ["01001"], "cci_score": [100]})
        result = run_stability_analysis(df)
        assert result.iloc[0]["status"] == "missing_columns"
