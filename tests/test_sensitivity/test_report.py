"""Tests for sensitivity/report.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


def _mock_module_result(spearman_r: float = 0.95, max_shift: int = 5) -> pd.DataFrame:
    """Create a simple DataFrame mimicking a sensitivity module result."""
    return pd.DataFrame(
        [
            {
                "scheme": "variant_a",
                "spearman_r_vs_primary": spearman_r,
                "max_rank_shift": max_shift,
                "n_shifted_gt_10": 1,
            }
        ]
    )


def _mock_wp_result() -> tuple[pd.DataFrame, dict]:
    """Mock weight_perturbation result (detail_df, summary_dict)."""
    detail = pd.DataFrame(
        {"fips": ["00001", "00002"], "rank_mean": [1.5, 2.5], "rank_std": [0.3, 0.4]}
    )
    summary = {
        "spearman_r_median": 0.97,
        "max_rank_shift_p95": 8,
        "n_unstable_counties": 2,
    }
    return (detail, summary)


def _mock_jackknife_result() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"excluded_component": "hdd_anomaly", "spearman_r": 0.96, "max_rank_shift": 4, "n_shifted_gt_10": 0},
            {"excluded_component": "drought_score", "spearman_r": 0.93, "max_rank_shift": 7, "n_shifted_gt_10": 1},
        ]
    )


def _mock_baseline_result() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"baseline": "1991-2020", "spearman_r_vs_primary": 1.0, "max_rank_shift": 0, "n_shifted_gt_10": 0, "status": "primary"},
            {"baseline": "1981-2010", "spearman_r_vs_primary": np.nan, "max_rank_shift": np.nan, "n_shifted_gt_10": np.nan, "status": "data_unavailable"},
        ]
    )


# Patches for all 8 sensitivity modules
_MODULE_PATCHES = {
    "sensitivity.report.run_weight_perturbation": _mock_wp_result,
    "sensitivity.report.run_jackknife": _mock_jackknife_result,
    "sensitivity.report.run_alt_weighting": lambda df: _mock_module_result(0.94),
    "sensitivity.report.run_correlation_threshold_sweep": lambda df: _mock_module_result(0.96),
    "sensitivity.report.run_normalization_comparison": lambda df: _mock_module_result(0.91),
    "sensitivity.report.run_severity_perturbation": lambda df: _mock_module_result(0.98),
    "sensitivity.report.run_window_sensitivity": lambda df: _mock_module_result(0.95),
    "sensitivity.report.run_baseline_comparison": lambda df: _mock_baseline_result(),
}


@pytest.fixture
def mock_all_modules():
    """Patch all sensitivity modules to return valid results."""
    # weight_perturbation needs special handling (takes extra args via lambda in report.py)
    def wp_wrapper(df, base_weights, n_iterations=10000, perturbation_pct=0.30):
        return _mock_wp_result()

    patches = {
        "sensitivity.report.run_weight_perturbation": wp_wrapper,
        "sensitivity.report.run_jackknife": lambda df: _mock_jackknife_result(),
        "sensitivity.report.run_alt_weighting": lambda df: _mock_module_result(0.94),
        "sensitivity.report.run_correlation_threshold_sweep": lambda df: _mock_module_result(0.96),
        "sensitivity.report.run_normalization_comparison": lambda df: _mock_module_result(0.91),
        "sensitivity.report.run_severity_perturbation": lambda df: _mock_module_result(0.98),
        "sensitivity.report.run_window_sensitivity": lambda df: _mock_module_result(0.95),
        "sensitivity.report.run_baseline_comparison": lambda df: _mock_baseline_result(),
    }
    with patch.multiple("sensitivity.report", **{k.split(".")[-1]: v for k, v in patches.items()}):
        yield


@pytest.fixture
def simple_harmonized():
    """Minimal harmonized DataFrame for module calls."""
    return pd.DataFrame({"fips": ["00001"], "year": [2024], "hdd_anomaly": [10.0]})


class TestRunSensitivitySuite:
    """Tests for run_sensitivity_suite."""

    def test_all_modules_succeed(self, simple_harmonized, mock_all_modules, tmp_path):
        """All 8 modules run and return success status."""
        from sensitivity.report import run_sensitivity_suite

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.validation_dir = tmp_path
            mock_settings.return_value = s

            with patch("sensitivity.report.get_weights", return_value={"hdd_anomaly": 1.0}):
                results = run_sensitivity_suite(simple_harmonized)

        module_names = [
            "weight_perturbation", "jackknife", "alt_weighting",
            "correlation_threshold", "normalization", "severity_perturbation",
            "window_sensitivity", "baseline_comparison",
        ]
        for name in module_names:
            assert name in results, f"Missing module: {name}"
            assert results[name]["status"] == "success"

        assert "_meta" in results
        assert results["_meta"]["n_success"] == 8

    def test_one_module_fails_others_continue(self, simple_harmonized, tmp_path):
        """A failing module should not stop other modules."""
        from sensitivity.report import run_sensitivity_suite

        def wp_wrapper(df, base_weights, n_iterations=10000, perturbation_pct=0.30):
            return _mock_wp_result()

        patches = {
            "run_weight_perturbation": wp_wrapper,
            "run_jackknife": MagicMock(side_effect=RuntimeError("jackknife exploded")),
            "run_alt_weighting": lambda df: _mock_module_result(0.94),
            "run_correlation_threshold_sweep": lambda df: _mock_module_result(0.96),
            "run_normalization_comparison": lambda df: _mock_module_result(0.91),
            "run_severity_perturbation": lambda df: _mock_module_result(0.98),
            "run_window_sensitivity": lambda df: _mock_module_result(0.95),
            "run_baseline_comparison": lambda df: _mock_baseline_result(),
        }

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.validation_dir = tmp_path
            mock_settings.return_value = s

            with patch("sensitivity.report.get_weights", return_value={"hdd_anomaly": 1.0}):
                with patch.multiple("sensitivity.report", **patches):
                    results = run_sensitivity_suite(simple_harmonized)

        assert results["jackknife"]["status"] == "failed"
        assert "exploded" in results["jackknife"]["error"]
        assert results["weight_perturbation"]["status"] == "success"
        assert results["alt_weighting"]["status"] == "success"
        assert results["_meta"]["n_success"] == 7


class TestGenerateSensitivityReport:
    """Tests for generate_sensitivity_report."""

    def _make_results(self, **overrides) -> dict:
        """Build a mock results dict."""
        base = {
            "weight_perturbation": {"result": _mock_wp_result(), "status": "success", "elapsed_s": 1.0},
            "jackknife": {"result": _mock_jackknife_result(), "status": "success", "elapsed_s": 0.5},
            "alt_weighting": {"result": _mock_module_result(0.94), "status": "success", "elapsed_s": 0.3},
            "correlation_threshold": {"result": _mock_module_result(0.96), "status": "success", "elapsed_s": 0.2},
            "normalization": {"result": _mock_module_result(0.91), "status": "success", "elapsed_s": 0.2},
            "severity_perturbation": {"result": _mock_module_result(0.98), "status": "success", "elapsed_s": 0.1},
            "window_sensitivity": {"result": _mock_module_result(0.95), "status": "success", "elapsed_s": 0.2},
            "baseline_comparison": {"result": _mock_baseline_result(), "status": "success", "elapsed_s": 0.1},
            "_meta": {"total_elapsed_s": 2.6, "n_modules": 8, "n_success": 8},
        }
        base.update(overrides)
        return base

    def test_complete_summary_columns(self, tmp_path):
        """Summary should have expected columns."""
        from sensitivity.report import generate_sensitivity_report

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            summary = generate_sensitivity_report(self._make_results())

        expected_cols = {"test_name", "variant", "spearman_r_vs_primary", "max_rank_shift", "n_flagged_counties", "status"}
        assert expected_cols.issubset(set(summary.columns))

    def test_all_modules_have_rows(self, tmp_path):
        """Every module should have at least one row in the summary."""
        from sensitivity.report import generate_sensitivity_report

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            summary = generate_sensitivity_report(self._make_results())

        module_names = {"weight_perturbation", "jackknife", "alt_weighting",
                        "correlation_threshold", "normalization", "severity_perturbation",
                        "window_sensitivity", "baseline_comparison"}
        assert module_names.issubset(set(summary["test_name"].unique()))

    def test_failed_module_recorded(self, tmp_path):
        """A failed module should appear with status='failed'."""
        from sensitivity.report import generate_sensitivity_report

        results = self._make_results(
            jackknife={"status": "failed", "error": "boom", "elapsed_s": 0.1}
        )

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            summary = generate_sensitivity_report(results)

        jk_rows = summary[summary["test_name"] == "jackknife"]
        assert len(jk_rows) == 1
        assert jk_rows.iloc[0]["status"] == "failed"

    def test_robustness_robust(self, tmp_path):
        """All Spearman r > 0.90 → 'robust'."""
        from sensitivity.report import generate_sensitivity_report

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            summary = generate_sensitivity_report(self._make_results())

        assert summary.attrs["robustness"] == "robust"

    def test_robustness_moderately_robust(self, tmp_path):
        """One Spearman r = 0.85 → 'moderately_robust'."""
        from sensitivity.report import generate_sensitivity_report

        results = self._make_results(
            normalization={"result": _mock_module_result(0.85), "status": "success", "elapsed_s": 0.1}
        )

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            summary = generate_sensitivity_report(results)

        assert summary.attrs["robustness"] == "moderately_robust"

    def test_robustness_fragile(self, tmp_path):
        """One Spearman r = 0.75 → 'fragile'."""
        from sensitivity.report import generate_sensitivity_report

        results = self._make_results(
            normalization={"result": _mock_module_result(0.75), "status": "success", "elapsed_s": 0.1}
        )

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            summary = generate_sensitivity_report(results)

        assert summary.attrs["robustness"] == "fragile — methodological review needed"

    def test_empty_results_graceful(self, tmp_path):
        """If all modules fail, summary should still be a valid DataFrame."""
        from sensitivity.report import generate_sensitivity_report

        results = {
            name: {"status": "failed", "error": "boom", "elapsed_s": 0.0}
            for name in [
                "weight_perturbation", "jackknife", "alt_weighting",
                "correlation_threshold", "normalization", "severity_perturbation",
                "window_sensitivity", "baseline_comparison",
            ]
        }
        results["_meta"] = {"total_elapsed_s": 0.0, "n_modules": 8, "n_success": 0}

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            summary = generate_sensitivity_report(results)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 8
        assert all(summary["status"] == "failed")

    def test_metadata_json_saved(self, tmp_path):
        """Verify sensitivity_metadata.json is written with expected keys."""
        from sensitivity.report import generate_sensitivity_report

        with patch("sensitivity.report.get_settings") as mock_settings:
            s = MagicMock()
            s.validation_dir = tmp_path
            s.monte_carlo_iterations = 100
            s.weight_perturbation_pct = 0.30
            s.overlap_correlation_threshold = 0.6
            mock_settings.return_value = s

            generate_sensitivity_report(self._make_results())

        meta_path = tmp_path / "sensitivity" / "sensitivity_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert "robustness" in meta
        assert "module_statuses" in meta
        assert meta["robustness"] == "robust"
        assert meta["n_modules"] == 8
