"""Aggregate all sensitivity analysis results into a summary report."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.components import get_weights
from config.settings import get_settings
from score.pipeline import compute_cci
from sensitivity.weight_perturbation import run_weight_perturbation
from sensitivity.jackknife import run_jackknife
from sensitivity.alt_weighting import run_alt_weighting
from sensitivity.correlation_threshold import run_correlation_threshold_sweep
from sensitivity.normalization_compare import run_normalization_comparison
from sensitivity.severity_perturbation import run_severity_perturbation
from sensitivity.window_sensitivity import run_window_sensitivity
from sensitivity.baseline_comparison import run_baseline_comparison

logger = logging.getLogger(__name__)


def run_sensitivity_suite(harmonized_df: pd.DataFrame) -> dict:
    """Run all sensitivity modules and aggregate results.

    Each module is wrapped in try/except so a single failure does not
    abort the entire suite.

    Returns:
        Dict keyed by module name, each value is the module's result
        (DataFrame or dict) or an error dict on failure.
    """
    settings = get_settings()
    weights = get_weights()

    t0 = time.monotonic()
    results: dict[str, Any] = {}

    modules: list[tuple[str, Any, dict]] = [
        (
            "weight_perturbation",
            lambda df: run_weight_perturbation(
                df,
                weights,
                n_iterations=settings.monte_carlo_iterations,
                perturbation_pct=settings.weight_perturbation_pct,
            ),
            {},
        ),
        ("jackknife", run_jackknife, {}),
        ("alt_weighting", run_alt_weighting, {}),
        ("correlation_threshold", run_correlation_threshold_sweep, {}),
        ("normalization", run_normalization_comparison, {}),
        ("severity_perturbation", run_severity_perturbation, {}),
        ("window_sensitivity", run_window_sensitivity, {}),
        ("baseline_comparison", run_baseline_comparison, {}),
    ]

    for name, func, _kwargs in modules:
        logger.info("Running sensitivity module: %s", name)
        t_mod = time.monotonic()
        try:
            result = func(harmonized_df)
            elapsed = time.monotonic() - t_mod
            results[name] = {"result": result, "status": "success", "elapsed_s": elapsed}
            logger.info("  %s completed in %.1fs", name, elapsed)
        except Exception as e:
            elapsed = time.monotonic() - t_mod
            logger.error("  %s FAILED after %.1fs: %s", name, elapsed, e, exc_info=True)
            results[name] = {"status": "failed", "error": str(e), "elapsed_s": elapsed}

    total_elapsed = time.monotonic() - t0
    n_success = sum(1 for v in results.values() if v["status"] == "success")
    logger.info(
        "Sensitivity suite complete: %d/%d modules succeeded in %.1fs",
        n_success,
        len(modules),
        total_elapsed,
    )
    results["_meta"] = {"total_elapsed_s": total_elapsed, "n_modules": len(modules), "n_success": n_success}

    return results


def generate_sensitivity_report(results: dict) -> pd.DataFrame:
    """Format sensitivity results into a summary table.

    Extracts key metrics from each module's output and computes an
    overall robustness assessment.

    Returns:
        DataFrame with columns: test_name, variant, spearman_r_vs_primary,
        max_rank_shift, n_flagged_counties, status.
    """
    rows: list[dict] = []

    extractors: dict[str, Any] = {
        "weight_perturbation": _extract_weight_perturbation,
        "jackknife": _extract_jackknife,
        "alt_weighting": _extract_multi_row,
        "correlation_threshold": _extract_multi_row,
        "normalization": _extract_multi_row,
        "severity_perturbation": _extract_multi_row,
        "window_sensitivity": _extract_multi_row,
        "baseline_comparison": _extract_baseline,
    }

    for name, extractor in extractors.items():
        entry = results.get(name)
        if entry is None:
            rows.append(_failed_row(name, "not_run", "Module not found in results"))
            continue
        if entry.get("status") == "failed":
            rows.append(_failed_row(name, "failed", entry.get("error", "")))
            continue
        try:
            extracted = extractor(name, entry["result"])
            rows.extend(extracted)
        except Exception as e:
            logger.warning("Failed to extract results for %s: %s", name, e)
            rows.append(_failed_row(name, "extraction_failed", str(e)))

    summary = pd.DataFrame(rows)

    # Compute overall robustness assessment
    robustness = _assess_robustness(summary)
    meta = results.get("_meta", {})
    summary.attrs["robustness"] = robustness
    summary.attrs["meta"] = meta

    # Save outputs
    _save_outputs(results, summary, robustness, meta)

    return summary


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_weight_perturbation(name: str, result: Any) -> list[dict]:
    """Weight perturbation returns (detail_df, summary_dict)."""
    if isinstance(result, tuple):
        _detail_df, summary = result
    else:
        summary = result

    median_r = summary.get("spearman_r_median", np.nan) if isinstance(summary, dict) else np.nan
    return [
        {
            "test_name": name,
            "variant": "monte_carlo",
            "spearman_r_vs_primary": median_r,
            "max_rank_shift": summary.get("max_rank_shift_p95", np.nan) if isinstance(summary, dict) else np.nan,
            "n_flagged_counties": summary.get("n_unstable_counties", 0) if isinstance(summary, dict) else 0,
            "status": "success",
        }
    ]


def _extract_jackknife(name: str, result: Any) -> list[dict]:
    """Jackknife returns a DataFrame with one row per excluded component."""
    if not isinstance(result, pd.DataFrame) or result.empty:
        return [_failed_row(name, "empty_result", "")]

    rows = []
    for _, row in result.iterrows():
        r_col = "spearman_r" if "spearman_r" in row.index else "spearman_r_vs_primary"
        rows.append(
            {
                "test_name": name,
                "variant": f"drop_{row.get('excluded_component', 'unknown')}",
                "spearman_r_vs_primary": row.get(r_col, np.nan),
                "max_rank_shift": row.get("max_rank_shift", np.nan),
                "n_flagged_counties": row.get("n_shifted_gt_10", 0),
                "status": "success",
            }
        )
    return rows


def _extract_multi_row(name: str, result: Any) -> list[dict]:
    """Generic extractor for modules returning a DataFrame with multiple variants."""
    if not isinstance(result, pd.DataFrame) or result.empty:
        return [_failed_row(name, "empty_result", "")]

    # Detect the variant column
    variant_col = None
    for candidate in ("scheme", "threshold", "normalization", "perturbation_factor", "window_years", "baseline"):
        if candidate in result.columns:
            variant_col = candidate
            break

    r_col = None
    for candidate in ("spearman_r_vs_primary", "spearman_r"):
        if candidate in result.columns:
            r_col = candidate
            break

    rows = []
    for _, row in result.iterrows():
        variant = str(row[variant_col]) if variant_col else "default"
        # Skip primary/reference rows (r=1.0)
        r_val = row.get(r_col, np.nan) if r_col else np.nan
        rows.append(
            {
                "test_name": name,
                "variant": variant,
                "spearman_r_vs_primary": r_val,
                "max_rank_shift": row.get("max_rank_shift", np.nan),
                "n_flagged_counties": row.get("n_shifted_gt_10", 0),
                "status": row.get("status", "success"),
            }
        )
    return rows


def _extract_baseline(name: str, result: Any) -> list[dict]:
    """Baseline comparison returns a DataFrame with baseline column."""
    return _extract_multi_row(name, result)


def _failed_row(name: str, status: str, error: str) -> dict:
    return {
        "test_name": name,
        "variant": "n/a",
        "spearman_r_vs_primary": np.nan,
        "max_rank_shift": np.nan,
        "n_flagged_counties": 0,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Robustness assessment
# ---------------------------------------------------------------------------

def _assess_robustness(summary: pd.DataFrame) -> str:
    """Compute overall robustness label based on minimum Spearman r.

    Excludes Monte Carlo (weight_perturbation) since it reports a distribution,
    and excludes data_unavailable/failed rows.
    """
    deterministic = summary[
        (summary["test_name"] != "weight_perturbation")
        & (summary["status"] == "success")
    ]

    r_values = deterministic["spearman_r_vs_primary"].dropna()

    # Exclude reference rows (r == 1.0 exactly)
    r_values = r_values[r_values < 1.0]

    if r_values.empty:
        return "insufficient_data"

    min_r = r_values.min()
    if min_r > 0.90:
        return "robust"
    elif min_r > 0.80:
        return "moderately_robust"
    else:
        return "fragile — methodological review needed"


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def _save_outputs(results: dict, summary: pd.DataFrame, robustness: str, meta: dict) -> None:
    """Save sensitivity outputs to data/validation/sensitivity/."""
    settings = get_settings()
    sens_dir = settings.validation_dir / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    # Save weight_perturbation detail if available
    wp = results.get("weight_perturbation")
    if wp and wp.get("status") == "success":
        result = wp["result"]
        if isinstance(result, tuple) and len(result) >= 1:
            detail_df = result[0]
            if isinstance(detail_df, pd.DataFrame):
                detail_df.to_parquet(sens_dir / "weight_perturbation_detail.parquet")
                logger.info("Saved weight perturbation detail to %s", sens_dir)

    # Build metadata
    module_statuses = {
        k: v.get("status", "unknown")
        for k, v in results.items()
        if k != "_meta"
    }
    metadata = {
        "robustness": robustness,
        "module_statuses": module_statuses,
        "total_elapsed_s": meta.get("total_elapsed_s"),
        "n_modules": meta.get("n_modules"),
        "n_success": meta.get("n_success"),
        "monte_carlo_iterations": settings.monte_carlo_iterations,
        "weight_perturbation_pct": settings.weight_perturbation_pct,
        "overlap_correlation_threshold": settings.overlap_correlation_threshold,
    }
    metadata_path = sens_dir / "sensitivity_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved sensitivity metadata to %s", metadata_path)
