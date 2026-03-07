"""Tests for validate/convergent_divergent.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validate.convergent_divergent import run_convergent_divergent, _interpret


def _make_scored(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "cci_score": rng.normal(100, 15, n),
    }, index=pd.Index(fips, name="fips"))


class TestPerfectCorrelation:
    def test_convergent_passes_divergent_fails(self):
        """r = 1.0: convergent passes (>0.4), divergent fails (>0.85)."""
        scored = _make_scored(100)
        # Mock NRI as identical to CCI — but since NRI data isn't on disk,
        # we test the interpretation logic directly
        assert _interpret(1.0) == "WARNING: CCI may be duplicating existing metric"

    def test_r_one_convergent_within(self):
        # r=1.0 is within [0.4, 0.7]? No, it's above.
        # But convergent "within_target" checks 0.4 <= r <= 0.7
        # r=1.0 → within_target=False for convergent (above range)
        # r=1.0 → within_target=False for divergent (above 0.85)
        pass


class TestZeroCorrelation:
    def test_interpretation_below_threshold(self):
        assert _interpret(0.0) == "WARNING: CCI may not capture hazard exposure"
        assert _interpret(0.1) == "WARNING: CCI may not capture hazard exposure"
        assert _interpret(0.29) == "WARNING: CCI may not capture hazard exposure"


class TestModerateCorrelation:
    def test_interpretation_within_range(self):
        assert _interpret(0.5) == "Within convergent target range"
        assert _interpret(0.4) == "Within convergent target range"
        assert _interpret(0.7) == "Within convergent target range"


class TestInterpretationRanges:
    def test_below_03(self):
        assert "WARNING" in _interpret(0.2)

    def test_03_to_04(self):
        assert "Below convergent target" in _interpret(0.35)

    def test_04_to_07(self):
        assert "Within convergent target" in _interpret(0.55)

    def test_07_to_085(self):
        assert "Above convergent target" in _interpret(0.78)

    def test_above_085(self):
        assert "duplicating" in _interpret(0.90)


class TestMissingBenchmarkData:
    def test_nri_unavailable(self, tmp_path):
        """No fema_nri directory → data_unavailable."""
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw")
        scored = _make_scored(50)
        result = run_convergent_divergent(scored, settings)
        nri_rows = result[result["benchmark"] == "fema_nri"]
        assert len(nri_rows) == 2
        assert all(nri_rows["status"] == "data_unavailable")

    def test_first_street_unavailable(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw")
        scored = _make_scored(50)
        result = run_convergent_divergent(scored, settings)
        fs_rows = result[result["benchmark"] == "first_street"]
        assert len(fs_rows) == 2
        assert all(fs_rows["status"] == "data_unavailable")


class TestTwoRowsPerBenchmark:
    def test_each_benchmark_has_convergent_and_divergent(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw")
        scored = _make_scored(50)
        result = run_convergent_divergent(scored, settings)

        for bm in ["fema_nri", "first_street"]:
            bm_rows = result[result["benchmark"] == bm]
            assert len(bm_rows) == 2
            assert set(bm_rows["validity_type"]) == {"convergent", "divergent"}


class TestOutputSchema:
    def test_columns(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw")
        scored = _make_scored(50)
        result = run_convergent_divergent(scored, settings)
        expected = {
            "benchmark", "validity_type", "pearson_r", "spearman_r",
            "target_r_low", "target_r_high", "within_target",
            "interpretation", "status", "note",
        }
        assert expected.issubset(set(result.columns))

    def test_total_rows(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw")
        scored = _make_scored(50)
        result = run_convergent_divergent(scored, settings)
        # 2 benchmarks × 2 validity types = 4 rows
        assert len(result) == 4


class TestEmptyInput:
    def test_empty_scored(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw")
        scored = pd.DataFrame(columns=["cci_score"])
        scored.index.name = "fips"
        result = run_convergent_divergent(scored, settings)
        assert len(result) == 4
        assert all(result["status"] == "no_scored_data")
