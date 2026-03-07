"""Tests for validate/report.py."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from validate.report import (
    run_validation_suite,
    generate_validation_report,
)


def _make_scored(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "cci_score": rng.normal(100, 15, n),
    }, index=pd.Index(fips, name="fips"))


def _make_harmonized(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fips_list = [f"{i:05d}" for i in range(1, n + 1)]
    rows = []
    for yr in range(2020, 2025):
        for fips in fips_list:
            rows.append({
                "fips": fips,
                "year": yr,
                "fema_ia_burden": rng.exponential(500),
                "energy_cost_attributed": rng.normal(1200, 300),
            })
    return pd.DataFrame(rows)


class TestAllValidatorsSucceed:
    def test_complete_summary(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw", validation_dir=tmp_path / "val")

        scored = _make_scored()
        harmonized = _make_harmonized()

        results = run_validation_suite(
            scored_df=scored,
            harmonized_df=harmonized,
            settings=settings,
        )

        assert "external_criteria" in results
        assert "convergent_divergent" in results
        assert "stability" in results

        summary = generate_validation_report(results)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0


class TestMissingDataGraceful:
    def test_no_harmonized(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw", validation_dir=tmp_path / "val")

        scored = _make_scored()
        results = run_validation_suite(scored_df=scored, settings=settings)
        summary = generate_validation_report(results)

        # Should not crash
        assert isinstance(summary, pd.DataFrame)
        # External criteria indicators that need harmonized should be unavailable
        ext = results["external_criteria"]
        fema_row = ext[ext["indicator"] == "fema_ia_per_household"].iloc[0]
        assert fema_row["status"] == "data_unavailable"


class TestNoMultiYearScores:
    def test_stability_marked_unavailable(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw", validation_dir=tmp_path / "val")

        scored = _make_scored()
        results = run_validation_suite(scored_df=scored, settings=settings)
        summary = generate_validation_report(results)

        stability_rows = summary[summary["test_category"] == "stability"]
        assert len(stability_rows) >= 1
        assert stability_rows.iloc[0]["status"] == "no_multi_year_data"


class TestOutputStructure:
    def test_summary_columns(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw", validation_dir=tmp_path / "val")

        scored = _make_scored()
        results = run_validation_suite(scored_df=scored, settings=settings)
        summary = generate_validation_report(results)

        expected_cols = {"test_category", "test_name", "result", "threshold", "passes", "status"}
        assert expected_cols.issubset(set(summary.columns))


class TestOverallAssessment:
    def test_counts(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw", validation_dir=tmp_path / "val")

        scored = _make_scored()
        harmonized = _make_harmonized()
        results = run_validation_suite(
            scored_df=scored,
            harmonized_df=harmonized,
            settings=settings,
        )
        summary = generate_validation_report(results)

        n_passed = (summary["passes"] == True).sum()  # noqa: E712
        n_failed = ((summary["passes"] == False) & (summary["status"] == "success")).sum()  # noqa: E712
        n_skipped = summary["status"].isin([
            "data_unavailable", "no_multi_year_data",
            "no_scored_data", "insufficient_years",
        ]).sum()

        # At least some indicators should be skipped (insurance, property, etc.)
        assert n_skipped >= 3
        # Total should account for all rows
        assert n_passed + n_failed + n_skipped <= len(summary)


class TestValidatorFailure:
    def test_error_captured(self, tmp_path):
        from config.settings import Settings
        settings = Settings(raw_dir=tmp_path / "raw", validation_dir=tmp_path / "val")

        scored = _make_scored()

        with patch("validate.report.run_external_validation", side_effect=RuntimeError("boom")):
            results = run_validation_suite(scored_df=scored, settings=settings)

        assert isinstance(results["external_criteria"], dict)
        assert results["external_criteria"]["status"] == "failed"

        summary = generate_validation_report(results)
        ext_rows = summary[summary["test_category"] == "external_criteria"]
        assert ext_rows.iloc[0]["status"] == "failed"
