"""Tests for the CCI scoring pipeline (Module 3.1).

All tests use synthetic DataFrames — no real data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.components import COMPONENTS, get_weights
from config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPONENT_IDS = list(COMPONENTS.keys())


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance with test defaults."""
    defaults = dict(
        scoring_year=2024,
        winsorize_percentile=99.0,
        overlap_correlation_threshold=0.6,
        overlap_penalty_floor=0.2,
        acceleration_bounds=(0.5, 3.0),
        acceleration_denominator_epsilon_factor=0.1,
        acceleration_min_completeness=0.8,
        target_iqr=(80.0, 120.0),
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_scoring_year_df(n_counties: int = 10, year: int = 2024, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic scoring-year DataFrame."""
    rng = np.random.RandomState(seed)
    fips = [f"{i:05d}" for i in range(1, n_counties + 1)]
    data = {"fips": fips, "year": year}
    for comp_id in COMPONENT_IDS:
        data[comp_id] = rng.uniform(0, 100, n_counties).astype(float)
    # Make energy_cost_attributed have some negatives
    data["energy_cost_attributed"] = rng.uniform(-50, 200, n_counties)
    return pd.DataFrame(data)


def _make_multiyear_df(n_counties: int = 5, years: range = range(2019, 2025), seed: int = 42) -> pd.DataFrame:
    """Create a synthetic multi-year harmonized DataFrame."""
    rng = np.random.RandomState(seed)
    rows = []
    for year in years:
        for i in range(1, n_counties + 1):
            row = {"fips": f"{i:05d}", "year": year}
            for comp_id in COMPONENT_IDS:
                # Linear trend + noise
                row[comp_id] = 10.0 * i + 2.0 * (year - 2019) + rng.normal(0, 1)
            row["energy_cost_attributed"] = 50.0 + 5.0 * (year - 2019) + rng.normal(0, 5)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1: Transform Inputs
# ---------------------------------------------------------------------------

class TestTransformInputs:
    def test_log_transform(self):
        from score.transform_inputs import transform_inputs

        df = pd.DataFrame({
            "fips": ["00001"], "year": [2024],
            "storm_severity": [100.0],
        }).set_index("fips")

        result = transform_inputs(df, transform_rules={"storm_severity": "log"})
        expected = np.log1p(100.0)
        assert abs(result["storm_severity"].iloc[0] - expected) < 1e-10

    def test_sqrt_transform(self):
        from score.transform_inputs import transform_inputs

        df = pd.DataFrame({
            "fips": ["00001"], "year": [2024],
            "extreme_heat_days": [25.0],
        }).set_index("fips")

        result = transform_inputs(df, transform_rules={"extreme_heat_days": "sqrt"})
        assert abs(result["extreme_heat_days"].iloc[0] - 5.0) < 1e-10

    def test_identity_transform(self):
        from score.transform_inputs import transform_inputs

        df = pd.DataFrame({
            "fips": ["00001"], "year": [2024],
            "pm25_annual": [12.0],
        }).set_index("fips")

        result = transform_inputs(df, transform_rules={"pm25_annual": "identity"})
        assert result["pm25_annual"].iloc[0] == 12.0

    def test_negative_energy_passes_through(self):
        from score.transform_inputs import transform_inputs

        df = pd.DataFrame({
            "fips": ["00001"], "year": [2024],
            "energy_cost_attributed": [-50.0],
        }).set_index("fips")

        # energy_cost_attributed uses identity, but test with log to verify negative handling
        result = transform_inputs(df, transform_rules={"energy_cost_attributed": "log"})
        assert result["energy_cost_attributed"].iloc[0] == -50.0

    def test_sqrt_signed(self):
        """sqrt transform uses signed sqrt for negative values."""
        from score.transform_inputs import transform_inputs

        df = pd.DataFrame({
            "fips": ["00001"], "year": [2024],
            "extreme_heat_days": [-4.0],
        }).set_index("fips")

        result = transform_inputs(df, transform_rules={"extreme_heat_days": "sqrt"})
        assert abs(result["extreme_heat_days"].iloc[0] - (-2.0)) < 1e-10

    def test_uses_component_registry_defaults(self):
        from score.transform_inputs import transform_inputs

        df = pd.DataFrame({
            "fips": ["00001"], "year": [2024],
            "storm_severity": [100.0],
            "extreme_heat_days": [25.0],
            "pm25_annual": [12.0],
        }).set_index("fips")

        result = transform_inputs(df)
        assert abs(result["storm_severity"].iloc[0] - np.log1p(100.0)) < 1e-10
        assert abs(result["extreme_heat_days"].iloc[0] - 5.0) < 1e-10
        assert result["pm25_annual"].iloc[0] == 12.0


# ---------------------------------------------------------------------------
# Step 2: Winsorize
# ---------------------------------------------------------------------------

class TestWinsorize:
    def test_extreme_value_clamped(self):
        from score.winsorize import winsorize

        values = list(range(100)) + [10000]  # one extreme outlier
        df = pd.DataFrame({
            "fips": [f"{i:05d}" for i in range(101)],
            "drought_score": values,
        }).set_index("fips")

        result = winsorize(df, percentile=99.0, component_ids=["drought_score"])
        p99 = np.percentile(values, 99)
        assert result["drought_score"].max() <= p99

    def test_nan_unchanged(self):
        from score.winsorize import winsorize

        df = pd.DataFrame({
            "fips": ["00001", "00002", "00003"],
            "drought_score": [1.0, np.nan, 100.0],
        }).set_index("fips")

        result = winsorize(df, percentile=99.0, component_ids=["drought_score"])
        assert pd.isna(result["drought_score"].iloc[1])

    def test_upper_tail_only(self):
        """Lower tail should NOT be winsorized."""
        from score.winsorize import winsorize

        df = pd.DataFrame({
            "fips": [f"{i:05d}" for i in range(10)],
            "drought_score": [-100.0] + [50.0] * 9,
        }).set_index("fips")

        result = winsorize(df, percentile=99.0, component_ids=["drought_score"])
        # The low outlier should be preserved
        assert result["drought_score"].min() == -100.0


# ---------------------------------------------------------------------------
# Step 3: Universe + Percentile
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_universe_excludes_missing_required(self):
        from score.universe import define_universe

        df = pd.DataFrame({
            "fips": [f"{i:05d}" for i in range(1, 11)],
            "hdd_anomaly": [10.0] * 10,
            "cdd_anomaly": [5.0] * 10,
            "drought_score": [3.0] * 10,
            "storm_severity": [2.0] * 8 + [np.nan, np.nan],  # 2 missing
        })

        universe = define_universe(df)
        assert len(universe) == 8
        assert "00009" not in universe
        assert "00010" not in universe

    def test_universe_needs_at_least_one_degree_day(self):
        from score.universe import define_universe

        df = pd.DataFrame({
            "fips": ["00001", "00002"],
            "hdd_anomaly": [np.nan, 10.0],
            "cdd_anomaly": [np.nan, np.nan],
            "drought_score": [3.0, 3.0],
            "storm_severity": [2.0, 2.0],
        })

        universe = define_universe(df)
        assert len(universe) == 1
        assert "00002" in universe


class TestPercentile:
    def test_percentile_over_universe_only(self):
        from score.percentile import compute_percentiles
        from score.universe import define_universe

        # 8 counties in universe, 2 excluded
        df = pd.DataFrame({
            "fips": [f"{i:05d}" for i in range(1, 11)],
            "hdd_anomaly": list(range(1, 11)),
            "cdd_anomaly": [5.0] * 10,
            "drought_score": [3.0] * 10,
            "storm_severity": [2.0] * 8 + [np.nan, np.nan],
        })
        universe = define_universe(df)

        result = compute_percentiles(df.set_index("fips"), universe, component_ids=["hdd_anomaly"])
        # Percentiles should only span the 8 in-universe counties
        assert len(result) == 8


# ---------------------------------------------------------------------------
# Step 4: Center
# ---------------------------------------------------------------------------

class TestCenter:
    def test_center_subtraction(self):
        from score.center import center

        df = pd.DataFrame({
            "hdd_anomaly": [75.0, 30.0, 50.0],
        }, index=["00001", "00002", "00003"])

        result = center(df, component_ids=["hdd_anomaly"])
        assert result["hdd_anomaly"].iloc[0] == 25.0
        assert result["hdd_anomaly"].iloc[1] == -20.0
        assert result["hdd_anomaly"].iloc[2] == 0.0


# ---------------------------------------------------------------------------
# Step 5: Overlap Penalties
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_high_correlation_penalizes_lower_tier(self):
        from score.overlap import compute_correlation_matrix, compute_overlap_penalties

        # Create two components with high correlation (r ≈ 0.8)
        rng = np.random.RandomState(42)
        n = 100
        x = rng.normal(0, 1, n)
        y = 0.8 * x + 0.6 * rng.normal(0, 1, n)  # r ≈ 0.8

        df = pd.DataFrame({
            "energy_cost_attributed": x,  # Tier 1
            "health_burden": y,           # Tier 3
        }, index=[f"{i:05d}" for i in range(n)])

        universe = df.index
        corr = compute_correlation_matrix(df, universe, component_ids=["energy_cost_attributed", "health_burden"])

        r = corr.loc["energy_cost_attributed", "health_burden"]
        assert abs(r) > 0.6  # sanity check

        penalties, docs = compute_overlap_penalties(corr, threshold=0.6, floor=0.2)

        # Tier 1 should be unpenalized
        assert penalties["energy_cost_attributed"] == 1.0
        # Tier 3 should be penalized
        expected_penalty = 1.0 - r ** 2
        assert abs(penalties["health_burden"] - max(expected_penalty, 0.2)) < 0.05

    def test_low_correlation_no_penalty(self):
        from score.overlap import compute_overlap_penalties

        corr = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]],
            index=["hdd_anomaly", "drought_score"],
            columns=["hdd_anomaly", "drought_score"],
        )
        penalties, _ = compute_overlap_penalties(corr, threshold=0.6, floor=0.2)
        assert penalties["hdd_anomaly"] == 1.0
        assert penalties["drought_score"] == 1.0

    def test_penalty_floor(self):
        from score.overlap import compute_overlap_penalties

        # Very high correlation → penalty near 0, but floored at 0.2
        corr = pd.DataFrame(
            [[1.0, 0.99], [0.99, 1.0]],
            index=["energy_cost_attributed", "health_burden"],
            columns=["energy_cost_attributed", "health_burden"],
        )
        penalties, _ = compute_overlap_penalties(corr, threshold=0.6, floor=0.2)
        assert penalties["health_burden"] >= 0.2


# ---------------------------------------------------------------------------
# Step 6: Acceleration
# ---------------------------------------------------------------------------

class TestAcceleration:
    def test_known_linear_trend(self):
        from score.acceleration import compute_acceleration_multipliers, compute_theil_sen_slopes

        # 5 years of data, pure linear trend for one county
        rows = []
        for year in range(2020, 2025):
            rows.append({"fips": "00001", "year": year, "drought_score": 10.0 + 2.0 * (year - 2020)})
            rows.append({"fips": "00002", "year": year, "drought_score": 10.0 + 1.0 * (year - 2020)})
        df = pd.DataFrame(rows)

        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"])
        # County 1 slope = 2.0, County 2 slope = 1.0
        assert abs(slopes.loc["00001", "drought_score_slope"] - 2.0) < 0.01
        assert abs(slopes.loc["00002", "drought_score_slope"] - 1.0) < 0.01

    def test_static_component_neutral(self):
        from score.acceleration import compute_theil_sen_slopes

        # Only 1 year of data (like flood_exposure)
        df = pd.DataFrame({
            "fips": ["00001", "00002"],
            "year": [2024, 2024],
            "flood_exposure": [0.5, 0.8],
        })
        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["flood_exposure"])
        # Should be 0 slope (neutral → accel=1.0 after multiplier)
        assert slopes["flood_exposure_slope"].iloc[0] == 0.0

    def test_insufficient_completeness_neutral(self):
        from score.acceleration import compute_acceleration_multipliers, compute_theil_sen_slopes

        # 5-year window but only 2 data points for county (below 80% of 5 = 4)
        rows = [
            {"fips": "00001", "year": 2020, "drought_score": 10.0},
            {"fips": "00001", "year": 2024, "drought_score": 20.0},
        ]
        df = pd.DataFrame(rows)

        slopes = compute_theil_sen_slopes(df, scoring_year=2024, component_ids=["drought_score"], min_completeness=0.8)
        # With min_completeness=0.8, need 4 of 5 years → 2 is insufficient → NaN slope
        accel = compute_acceleration_multipliers(slopes)
        # NaN slope → neutral acceleration
        assert accel["drought_score_acceleration"].iloc[0] == 1.0

    def test_acceleration_bounds(self):
        from score.acceleration import compute_acceleration_multipliers

        slopes = pd.DataFrame({
            "drought_score_slope": [100.0, -100.0, 1.0],
        }, index=["00001", "00002", "00003"])

        accel = compute_acceleration_multipliers(slopes, bounds=(0.5, 3.0))
        assert accel["drought_score_acceleration"].max() <= 3.0
        assert accel["drought_score_acceleration"].min() >= 0.5


# ---------------------------------------------------------------------------
# Step 7: Missingness
# ---------------------------------------------------------------------------

class TestMissingness:
    def test_preferred_core_imputed_to_zero(self):
        from score.missingness import handle_missingness

        df = pd.DataFrame({
            "pm25_annual": [np.nan, 10.0],
            "hdd_anomaly": [5.0, 5.0],
        }, index=["00001", "00002"])

        result = handle_missingness(df)
        assert result["pm25_annual"].iloc[0] == 0.0
        assert result["pm25_annual"].iloc[1] == 10.0

    def test_stale_flood_map_downgrades_confidence(self):
        from score.missingness import handle_missingness

        df = pd.DataFrame({
            "flood_exposure": [0.5, 0.8],
            "flood_exposure__map_currency_flag": [1, 0],
            "flood_exposure__confidence": ["A", "A"],
        }, index=["00001", "00002"])

        result = handle_missingness(df)
        assert result["flood_exposure__confidence"].iloc[0] == "B"
        assert result["flood_exposure__confidence"].iloc[1] == "A"


# ---------------------------------------------------------------------------
# Steps 8-10: Composite + Scale
# ---------------------------------------------------------------------------

class TestComposite:
    def test_component_score_formula(self):
        from score.composite import compute_component_scores

        centered = pd.DataFrame({
            "hdd_anomaly": [25.0],
            "drought_score": [-10.0],
        }, index=["00001"])

        weights = {"hdd_anomaly": 0.5, "drought_score": 0.5}
        penalties = {"hdd_anomaly": 1.0, "drought_score": 0.8}
        accel = pd.DataFrame({
            "hdd_anomaly_acceleration": [1.5],
            "drought_score_acceleration": [1.0],
        }, index=["00001"])

        result = compute_component_scores(centered, weights, penalties, accel)
        # hdd: 25 * 0.5 * 1.0 * 1.5 = 18.75
        assert abs(result["hdd_anomaly"].iloc[0] - 18.75) < 1e-10
        # drought: -10 * 0.5 * 0.8 * 1.0 = -4.0
        assert abs(result["drought_score"].iloc[0] - (-4.0)) < 1e-10

    def test_calibrate_k(self):
        from score.composite import calibrate_k

        raw = pd.Series([10, 20, 30, 40, 50])
        k = calibrate_k(raw, target_iqr=(80.0, 120.0))
        q1 = raw.quantile(0.25)
        q3 = raw.quantile(0.75)
        expected_k = 40.0 / (q3 - q1)
        assert abs(k - expected_k) < 1e-10

    def test_k_guard_small_iqr(self):
        from score.composite import calibrate_k

        raw = pd.Series([10.0, 10.0, 10.0, 10.0])  # zero IQR
        k = calibrate_k(raw)
        assert k == 1.0  # guard against explosion

    def test_negative_composite_below_100(self):
        from score.composite import calibrate_k

        raw = pd.Series([-5, -3, -1, 1, 3])
        k = calibrate_k(raw)
        cci = 100 + k * raw
        assert cci.min() < 100


# ---------------------------------------------------------------------------
# Variant Outputs
# ---------------------------------------------------------------------------

class TestCCINational:
    def test_weighted_mean(self):
        from score.cci_national import compute_national_aggregate

        raw = pd.Series([10.0, 20.0], index=["00001", "00002"])
        housing = pd.Series([100, 300], index=["00001", "00002"])
        result = compute_national_aggregate(raw, housing)
        expected = (100 * 10 + 300 * 20) / 400
        assert abs(result - expected) < 1e-10


class TestCCIStrain:
    def test_reindexed_median_100(self):
        from score.cci_strain import compute_strain

        scores = pd.Series([90.0, 100.0, 110.0], index=["00001", "00002", "00003"])
        income = pd.Series([50000.0, 60000.0, 70000.0], index=["00001", "00002", "00003"])
        result = compute_strain(scores, income)
        assert abs(result["cci_strain"].median() - 100.0) < 1e-10


class TestCCIDollar:
    def test_passthrough(self):
        from score.cci_dollar import compute_dollar

        energy = pd.Series([150.0, -30.0, 200.0], index=["00001", "00002", "00003"])
        result = compute_dollar(energy)
        assert result["cci_dollar"].iloc[0] == 150.0
        assert result["cci_dollar"].iloc[1] == -30.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        from score.pipeline import compute_cci

        df = _make_multiyear_df(n_counties=20, years=range(2019, 2025))
        weights = get_weights()
        settings = _make_settings()

        result1 = compute_cci(df, weights, settings)
        result2 = compute_cci(df, weights, settings)

        pd.testing.assert_frame_equal(result1.scores, result2.scores)
        pd.testing.assert_frame_equal(result1.components, result2.components)
        assert result1.k == result2.k


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_components_missing_excluded(self):
        from score.universe import define_universe

        df = pd.DataFrame({
            "fips": ["00001"],
            "hdd_anomaly": [np.nan],
            "cdd_anomaly": [np.nan],
            "drought_score": [np.nan],
            "storm_severity": [np.nan],
        })
        universe = define_universe(df)
        assert len(universe) == 0

    def test_single_county_universe(self):
        from score.percentile import compute_percentiles

        df = pd.DataFrame({
            "hdd_anomaly": [50.0],
        }, index=["00001"])
        universe = pd.Index(["00001"])

        result = compute_percentiles(df, universe, component_ids=["hdd_anomaly"])
        # Single county → rank 1/1 → pct = 100 (rank of 1 out of 1)
        assert result["hdd_anomaly"].iloc[0] == 100.0


# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_cci_output_fields(self):
        from score.pipeline import CCIOutput, compute_cci

        df = _make_multiyear_df(n_counties=20, years=range(2019, 2025))
        weights = get_weights()
        settings = _make_settings()

        result = compute_cci(df, weights, settings)

        assert isinstance(result, CCIOutput)
        assert "cci_score" in result.scores.columns
        assert result.scores.index.name == "fips"
        assert isinstance(result.penalties, dict)
        assert isinstance(result.universe, pd.Index)
        assert isinstance(result.k, float)
        assert len(result.penalties) > 0


# ---------------------------------------------------------------------------
# Full End-to-End with 5 Counties
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_five_county_pipeline(self):
        """Trace through the full pipeline with 5 synthetic counties."""
        from score.pipeline import compute_cci

        rng = np.random.RandomState(123)
        n = 5
        years = list(range(2019, 2025))
        rows = []
        for year in years:
            for i in range(1, n + 1):
                row = {
                    "fips": f"{i:05d}",
                    "year": year,
                }
                for comp_id in COMPONENT_IDS:
                    row[comp_id] = max(0, 10.0 * i + rng.normal(0, 3))
                row["energy_cost_attributed"] = 50 + 10 * i + rng.normal(0, 5)
                rows.append(row)
        df = pd.DataFrame(rows)

        weights = get_weights()
        settings = _make_settings()

        result = compute_cci(df, weights, settings)

        # Basic structural checks
        assert len(result.universe) == n
        assert len(result.scores) == n
        assert result.scores["cci_score"].notna().all()
        assert result.k > 0

        # CCI-Score should be centered around 100
        median_cci = result.scores["cci_score"].median()
        assert 50 < median_cci < 150  # reasonable range for 5 counties

        # Components should all be present
        assert len(result.components.columns) > 0
