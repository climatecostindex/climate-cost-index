"""Tests for score/overlap.py — correlation, precedence, and penalty computation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from score.overlap import (
    CENSUS_REGIONS,
    _component_sort_key,
    _compute_discrepancies,
    _compute_partial_correlation,
    compute_correlation_matrix,
    compute_correlation_robustness,
    compute_overlap_penalties,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_centered(fips_list: list[str], data: dict[str, list[float]]) -> pd.DataFrame:
    """Build a synthetic centered DataFrame."""
    df = pd.DataFrame(data)
    df["fips"] = fips_list
    return df


def _make_universe(fips_list: list[str]) -> pd.Index:
    return pd.Index(fips_list, name="fips")


# Use real state FIPS from different regions for partial correlation tests
_NORTHEAST_FIPS = [f"09{str(i).zfill(3)}" for i in range(1, 11)]  # CT
_MIDWEST_FIPS = [f"17{str(i).zfill(3)}" for i in range(1, 11)]    # IL
_SOUTH_FIPS = [f"12{str(i).zfill(3)}" for i in range(1, 11)]      # FL
_WEST_FIPS = [f"06{str(i).zfill(3)}" for i in range(1, 11)]       # CA
_ALL_REGION_FIPS = _NORTHEAST_FIPS + _MIDWEST_FIPS + _SOUTH_FIPS + _WEST_FIPS


# ---------------------------------------------------------------------------
# Correlation matrix tests
# ---------------------------------------------------------------------------


class TestComputeCorrelationMatrix:
    def test_perfectly_correlated(self):
        """Two perfectly correlated components should have r = 1.0."""
        fips = [f"0100{i}" for i in range(10)]
        x = list(range(10))
        df = _make_centered(fips, {"comp_a": x, "comp_b": x})
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        assert corr.loc["comp_a", "comp_b"] == pytest.approx(1.0)
        assert corr.loc["comp_b", "comp_a"] == pytest.approx(1.0)

    def test_uncorrelated(self):
        """Independent components should have |r| well below threshold."""
        np.random.seed(42)
        n = 200
        fips = [f"{str(i).zfill(5)}" for i in range(n)]
        x = np.random.randn(n).tolist()
        y = np.random.randn(n).tolist()
        df = _make_centered(fips, {"comp_a": x, "comp_b": y})
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        assert abs(corr.loc["comp_a", "comp_b"]) < 0.3

    def test_known_correlation(self):
        """Component y = 2x + noise should give r close to expected value."""
        np.random.seed(123)
        n = 1000
        fips = [f"{str(i).zfill(5)}" for i in range(n)]
        x = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        y = 2 * x + noise

        df = _make_centered(fips, {"comp_a": x.tolist(), "comp_b": y.tolist()})
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        # r should be high (> 0.9)
        assert corr.loc["comp_a", "comp_b"] > 0.9

    def test_universe_subset(self):
        """Only counties in the universe should be included."""
        fips = [f"0100{i}" for i in range(10)]
        x = list(range(10))
        y = list(range(9, -1, -1))  # Perfect negative correlation
        df = _make_centered(fips, {"comp_a": x, "comp_b": y})

        # Use only first 5
        universe = _make_universe(fips[:5])

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        # Should still be -1.0 for the subset (both halves are monotonic)
        assert corr.loc["comp_a", "comp_b"] == pytest.approx(-1.0)

    def test_nan_values_pairwise(self):
        """NaN values should be handled via pairwise complete observations."""
        fips = [f"0100{i}" for i in range(6)]
        df = _make_centered(fips, {
            "comp_a": [1, 2, 3, 4, 5, float("nan")],
            "comp_b": [1, 2, 3, 4, 5, 6],
        })
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        # First 5 are perfectly correlated
        assert corr.loc["comp_a", "comp_b"] == pytest.approx(1.0)

    def test_single_county_degenerate(self):
        """Single county in universe should return NaN correlation matrix."""
        fips = ["01001"]
        df = _make_centered(fips, {"comp_a": [1.0], "comp_b": [2.0]})
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        assert np.isnan(corr.loc["comp_a", "comp_b"])

    def test_zero_variance_component(self):
        """Component with zero variance should produce NaN correlation."""
        fips = [f"0100{i}" for i in range(5)]
        df = _make_centered(fips, {
            "comp_a": [1.0, 1.0, 1.0, 1.0, 1.0],  # zero variance
            "comp_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        assert np.isnan(corr.loc["comp_a", "comp_b"])

    def test_index_based_universe(self):
        """Should work when fips is the index rather than a column."""
        fips = [f"0100{i}" for i in range(5)]
        df = pd.DataFrame({
            "comp_a": range(5),
            "comp_b": range(5),
        }, index=pd.Index(fips, name="fips"))
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            corr = compute_correlation_matrix(df, universe, component_ids=["comp_a", "comp_b"])

        assert corr.loc["comp_a", "comp_b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Precedence hierarchy tests
# ---------------------------------------------------------------------------


def _mock_components():
    """Create mock component definitions for testing precedence."""
    from config.components import Attribution, ComponentDef, OverlapPrecedenceTier

    return {
        "tier1_comp": ComponentDef(
            id="tier1_comp", name="Tier 1", source_module="test",
            attribution=Attribution.ATTRIBUTED, confidence="A",
            precedence_tier=OverlapPrecedenceTier.DIRECT_DOLLAR_ATTRIBUTED,
            base_weight=0.15, transform="identity",
            acceleration_window=5, unit="test",
        ),
        "tier2_confA_w10": ComponentDef(
            id="tier2_confA_w10", name="Tier 2 A w10", source_module="test",
            attribution=Attribution.PROXY, confidence="A",
            precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
            base_weight=0.10, transform="identity",
            acceleration_window=5, unit="test",
        ),
        "tier2_confB_w12": ComponentDef(
            id="tier2_confB_w12", name="Tier 2 B w12", source_module="test",
            attribution=Attribution.PROXY, confidence="B",
            precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
            base_weight=0.12, transform="identity",
            acceleration_window=5, unit="test",
        ),
        "tier2_confA_w08": ComponentDef(
            id="tier2_confA_w08", name="Tier 2 A w08", source_module="test",
            attribution=Attribution.PROXY, confidence="A",
            precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
            base_weight=0.08, transform="identity",
            acceleration_window=5, unit="test",
        ),
        "tier3_comp": ComponentDef(
            id="tier3_comp", name="Tier 3", source_module="test",
            attribution=Attribution.PROXY, confidence="B",
            precedence_tier=OverlapPrecedenceTier.GENERAL_EXPOSURE,
            base_weight=0.06, transform="log",
            acceleration_window=5, unit="test",
        ),
    }


class TestPrecedence:
    def test_tier1_beats_tier2(self):
        """Tier 1 component should never be penalized by Tier 2."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        assert penalties["tier2_confA_w10"] < 1.0

    def test_same_tier_confidence_wins(self):
        """Within same tier, higher confidence (A) beats lower (B)."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["tier2_confA_w10", "tier2_confB_w12"],
            columns=["tier2_confA_w10", "tier2_confB_w12"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        # A confidence wins even though B has higher weight
        assert penalties["tier2_confA_w10"] == pytest.approx(1.0)
        assert penalties["tier2_confB_w12"] < 1.0

    def test_same_tier_same_confidence_weight_wins(self):
        """Same tier, same confidence: larger weight wins."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["tier2_confA_w10", "tier2_confA_w08"],
            columns=["tier2_confA_w10", "tier2_confA_w08"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier2_confA_w10"] == pytest.approx(1.0)
        assert penalties["tier2_confA_w08"] < 1.0

    def test_three_tier2_components(self):
        """Three Tier 2 components with pairwise ordering maintained."""
        comps = _mock_components()
        # All three correlated: confA_w10 > confA_w08 > confB_w12
        corr = pd.DataFrame(
            [[1.0, 0.8, 0.7],
             [0.8, 1.0, 0.7],
             [0.7, 0.7, 1.0]],
            index=["tier2_confA_w10", "tier2_confA_w08", "tier2_confB_w12"],
            columns=["tier2_confA_w10", "tier2_confA_w08", "tier2_confB_w12"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        # Highest precedence not penalized
        assert penalties["tier2_confA_w10"] == pytest.approx(1.0)
        # Others penalized
        assert penalties["tier2_confA_w08"] < 1.0
        assert penalties["tier2_confB_w12"] < 1.0


# ---------------------------------------------------------------------------
# Penalty computation tests
# ---------------------------------------------------------------------------


class TestPenaltyComputation:
    def test_r_0_8_penalty(self):
        """r = 0.8 -> penalty = 1 - 0.64 = 0.36."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier2_confA_w10"] == pytest.approx(0.36)

    def test_negative_correlation(self):
        """r = -0.7 -> penalty = 1 - 0.49 = 0.51."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, -0.7], [-0.7, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier2_confA_w10"] == pytest.approx(0.51)

    def test_below_threshold_no_penalty(self):
        """r = 0.5 (below 0.6 threshold) -> no penalty."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        assert penalties["tier2_confA_w10"] == pytest.approx(1.0)

    def test_exactly_at_threshold_no_penalty(self):
        """r = 0.6 exactly -> no penalty (spec says |r| > 0.6, strict inequality)."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.6], [0.6, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier2_confA_w10"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cumulative penalties and floor
# ---------------------------------------------------------------------------


class TestCumulativePenaltiesAndFloor:
    def test_cumulative_multiplication(self):
        """B penalized by A (r=0.8) and C (r=0.7) -> cumulative with floor."""
        comps = _mock_components()
        # tier1 beats tier2_confA_w10 beats tier2_confA_w08
        # tier1 ↔ tier2_confA_w10: r=0.8, tier2_confA_w08 ↔ tier2_confA_w10: r=0.7
        # But we need tier2_confA_w10 to be penalized by two different comps.
        # tier1 ↔ tier2_confA_w10: r=0.8 (flagged, tier2_confA_w10 loses)
        # tier2_confA_w10 ↔ ... we need another higher-precedence comp.
        # Let's use a direct correlation matrix approach:
        corr = pd.DataFrame(
            [[1.0, 0.8, 0.0],
             [0.8, 1.0, 0.7],
             [0.0, 0.7, 1.0]],
            index=["tier1_comp", "tier2_confA_w10", "tier2_confA_w08"],
            columns=["tier1_comp", "tier2_confA_w10", "tier2_confA_w08"],
        )
        # tier2_confA_w10 penalized by tier1 (r=0.8, penalty=0.36)
        # tier2_confA_w08 penalized by tier2_confA_w10 (r=0.7, penalty=0.51)
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        assert penalties["tier2_confA_w10"] == pytest.approx(1.0 - 0.64)  # 0.36
        assert penalties["tier2_confA_w08"] == pytest.approx(1.0 - 0.49)  # 0.51

    def test_double_penalty_with_floor(self):
        """Component penalized twice below floor should be floored to 0.2."""
        comps = _mock_components()
        # tier2_confB_w12 penalized by tier1_comp (r=0.8) AND tier2_confA_w10 (r=0.7)
        # precedence: tier1 > tier2_confA_w10 > tier2_confB_w12
        corr = pd.DataFrame(
            [[1.0, 0.0, 0.8],
             [0.0, 1.0, 0.7],
             [0.8, 0.7, 1.0]],
            index=["tier1_comp", "tier2_confA_w10", "tier2_confB_w12"],
            columns=["tier1_comp", "tier2_confA_w10", "tier2_confB_w12"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        # tier2_confB_w12: 0.36 * 0.51 = 0.1836 -> floored to 0.2
        assert penalties["tier2_confB_w12"] == pytest.approx(0.2)

    def test_single_high_r_floors(self):
        """r=0.95 -> penalty = 1 - 0.9025 = 0.0975 -> floored to 0.2."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier2_confA_w10"] == pytest.approx(0.2)

    def test_floor_exactly_0_2(self):
        """Verify floor is exactly 0.2."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.99], [0.99, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier2_confA_w10"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Chain overlap handling
# ---------------------------------------------------------------------------


class TestChainOverlap:
    def test_three_tier_chain(self):
        """A (Tier 1) -> B (Tier 2) -> C (Tier 3), sequential penalties."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.7, 0.0],
             [0.7, 1.0, 0.8],
             [0.0, 0.8, 1.0]],
            index=["tier1_comp", "tier2_confA_w10", "tier3_comp"],
            columns=["tier1_comp", "tier2_confA_w10", "tier3_comp"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        assert penalties["tier2_confA_w10"] == pytest.approx(1.0 - 0.49)  # 0.51
        assert penalties["tier3_comp"] == pytest.approx(1.0 - 0.64)  # 0.36

    def test_full_chain_all_flagged(self):
        """A↔B, B↔C, A↔C all flagged. C gets cumulative penalty from A and B."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.7, 0.8],
             [0.7, 1.0, 0.7],
             [0.8, 0.7, 1.0]],
            index=["tier1_comp", "tier2_confA_w10", "tier3_comp"],
            columns=["tier1_comp", "tier2_confA_w10", "tier3_comp"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        # tier2 penalized by tier1 (r=0.7): 1-0.49 = 0.51
        assert penalties["tier2_confA_w10"] == pytest.approx(0.51)
        # tier3 penalized by tier1 (r=0.8, p=0.36) and tier2 (r=0.7, p=0.51)
        # cumulative = 0.36 * 0.51 = 0.1836 -> floored to 0.2
        assert penalties["tier3_comp"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------


class TestRobustnessChecks:
    def test_spearman_computed(self):
        """Verify Spearman correlation is returned."""
        np.random.seed(42)
        fips = _ALL_REGION_FIPS
        n = len(fips)
        df = _make_centered(fips, {
            "comp_a": np.random.randn(n).tolist(),
            "comp_b": np.random.randn(n).tolist(),
        })
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            result = compute_correlation_robustness(df, universe, component_ids=["comp_a", "comp_b"])

        assert "spearman_corr" in result
        assert isinstance(result["spearman_corr"], pd.DataFrame)
        assert result["spearman_corr"].shape == (2, 2)

    def test_distance_corr_none_when_not_installed(self):
        """Distance correlation should be None when dcor is not available."""
        fips = _ALL_REGION_FIPS[:5]
        df = _make_centered(fips, {"comp_a": [1, 2, 3, 4, 5], "comp_b": [5, 4, 3, 2, 1]})
        universe = _make_universe(fips)

        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "dcor":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            with patch("builtins.__import__", side_effect=mock_import):
                result = compute_correlation_robustness(
                    df, universe, component_ids=["comp_a", "comp_b"]
                )

        assert result["distance_corr"] is None

    def test_partial_correlation_computed(self):
        """Verify partial correlation is returned and is a DataFrame."""
        np.random.seed(42)
        fips = _ALL_REGION_FIPS
        n = len(fips)
        df = _make_centered(fips, {
            "comp_a": np.random.randn(n).tolist(),
            "comp_b": np.random.randn(n).tolist(),
        })
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            result = compute_correlation_robustness(df, universe, component_ids=["comp_a", "comp_b"])

        assert "partial_corr" in result
        assert isinstance(result["partial_corr"], pd.DataFrame)

    def test_partial_corr_controls_for_region(self):
        """Partial correlation should reduce r when correlation is region-driven."""
        np.random.seed(42)
        fips = _ALL_REGION_FIPS
        n = len(fips)

        # Create data where both components are driven by region
        # Region 1 (NE): high, Region 2 (MW): medium, Region 3 (S): low, Region 4 (W): very low
        region_effect = []
        for f in fips:
            region = CENSUS_REGIONS.get(f[:2], 1)
            region_effect.append(region * 10)

        noise = np.random.randn(n) * 2
        comp_a = [r + n for r, n in zip(region_effect, noise)]
        noise2 = np.random.randn(n) * 2
        comp_b = [r + n for r, n in zip(region_effect, noise2)]

        df = _make_centered(fips, {"comp_a": comp_a, "comp_b": comp_b})
        universe = _make_universe(fips)

        with patch("score.overlap.COMPONENTS", {"comp_a": None, "comp_b": None}):
            result = compute_correlation_robustness(df, universe, component_ids=["comp_a", "comp_b"])

        pearson_r = abs(result["pearson_corr"].loc["comp_a", "comp_b"])
        partial_r = abs(result["partial_corr"].loc["comp_a", "comp_b"])

        # Partial correlation should be meaningfully lower
        assert partial_r < pearson_r


# ---------------------------------------------------------------------------
# Discrepancy detection
# ---------------------------------------------------------------------------


class TestDiscrepancyDetection:
    def test_no_discrepancies_when_methods_agree(self):
        """All methods agree -> no discrepancies."""
        comp_ids = ["comp_a", "comp_b"]
        pearson = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]], index=comp_ids, columns=comp_ids
        )
        spearman = pd.DataFrame(
            [[1.0, 0.75], [0.75, 1.0]], index=comp_ids, columns=comp_ids
        )
        # Both flag (above 0.6)
        discrepancies = _compute_discrepancies(pearson, spearman, None, None, comp_ids, 0.6)
        assert len(discrepancies) == 0

    def test_discrepancy_pearson_flags_spearman_does_not(self):
        """Pearson flags but Spearman does not -> discrepancy recorded."""
        comp_ids = ["comp_a", "comp_b"]
        pearson = pd.DataFrame(
            [[1.0, 0.65], [0.65, 1.0]], index=comp_ids, columns=comp_ids
        )
        spearman = pd.DataFrame(
            [[1.0, 0.55], [0.55, 1.0]], index=comp_ids, columns=comp_ids
        )
        discrepancies = _compute_discrepancies(pearson, spearman, None, None, comp_ids, 0.6)
        assert len(discrepancies) == 1
        d = discrepancies[0]
        assert d["pair"] == ("comp_a", "comp_b")
        assert d["methods"]["pearson"]["flagged"] == True  # noqa: E712
        assert d["methods"]["spearman"]["flagged"] == False  # noqa: E712
        assert d["conservative_flag"] == True  # noqa: E712

    def test_conservative_flagging_in_penalties(self):
        """When robustness results flag a pair that Pearson doesn't, conservative = flag it."""
        comps = _mock_components()
        # Pearson: r=0.55 (below threshold)
        corr = pd.DataFrame(
            [[1.0, 0.55], [0.55, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        # Spearman: r=0.65 (above threshold)
        robustness = {
            "spearman_corr": pd.DataFrame(
                [[1.0, 0.65], [0.65, 1.0]],
                index=["tier1_comp", "tier2_confA_w10"],
                columns=["tier1_comp", "tier2_confA_w10"],
            ),
            "distance_corr": None,
            "partial_corr": None,
        }

        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(
                corr, robustness_results=robustness
            )

        # Should be flagged due to Spearman (conservative)
        # But penalty uses Pearson r value: 1 - 0.55^2 = 1 - 0.3025 = 0.6975
        assert penalties["tier2_confA_w10"] < 1.0
        assert len(docs["flagged_pairs"]) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_universe(self):
        """Empty universe -> all penalties 1.0."""
        comps = _mock_components()
        corr = pd.DataFrame(
            np.nan, index=["tier1_comp"], columns=["tier1_comp"]
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        assert len(docs["flagged_pairs"]) == 0

    def test_single_component(self):
        """Only 1 component -> no pairs, penalty = 1.0."""
        comps = _mock_components()
        corr = pd.DataFrame([[1.0]], index=["tier1_comp"], columns=["tier1_comp"])
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        assert len(docs["flagged_pairs"]) == 0

    def test_all_nan_correlation_matrix(self):
        """All NaN correlation -> all penalties 1.0."""
        comps = _mock_components()
        corr = pd.DataFrame(
            np.nan,
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert penalties["tier1_comp"] == pytest.approx(1.0)
        assert penalties["tier2_confA_w10"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Documentation output
# ---------------------------------------------------------------------------


class TestDocumentation:
    def test_docs_structure(self):
        """Verify docs dict has required keys."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            penalties, docs = compute_overlap_penalties(corr)

        assert "flagged_pairs" in docs
        assert "precedence_decisions" in docs
        assert "penalties" in docs

    def test_flagged_pairs_format(self):
        """Flagged pairs should be (comp1, comp2, r) tuples."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            _, docs = compute_overlap_penalties(corr)

        assert len(docs["flagged_pairs"]) == 1
        pair = docs["flagged_pairs"][0]
        assert len(pair) == 3
        assert isinstance(pair[2], float)

    def test_precedence_decisions_format(self):
        """Precedence decisions should include winner, loser, r, raw_penalty."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["tier1_comp", "tier2_confA_w10"],
            columns=["tier1_comp", "tier2_confA_w10"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            _, docs = compute_overlap_penalties(corr)

        assert len(docs["precedence_decisions"]) == 1
        dec = docs["precedence_decisions"][0]
        assert dec["winner"] == "tier1_comp"
        assert dec["loser"] == "tier2_confA_w10"
        assert dec["r"] == pytest.approx(0.8)
        assert dec["raw_penalty"] == pytest.approx(0.36)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        """Running twice with same input should produce identical penalties."""
        comps = _mock_components()
        corr = pd.DataFrame(
            [[1.0, 0.8, 0.3],
             [0.8, 1.0, 0.7],
             [0.3, 0.7, 1.0]],
            index=["tier1_comp", "tier2_confA_w10", "tier3_comp"],
            columns=["tier1_comp", "tier2_confA_w10", "tier3_comp"],
        )
        with patch("score.overlap.COMPONENTS", comps):
            p1, d1 = compute_overlap_penalties(corr)
            p2, d2 = compute_overlap_penalties(corr)

        assert p1 == p2
        assert d1["flagged_pairs"] == d2["flagged_pairs"]
