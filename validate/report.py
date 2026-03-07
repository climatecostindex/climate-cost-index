"""Generate validation summary report."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from config.settings import Settings, get_settings
from validate.external_criteria import run_external_validation
from validate.convergent_divergent import run_convergent_divergent
from validate.stability import run_stability_analysis

logger = logging.getLogger(__name__)


def run_validation_suite(
    scored_df: pd.DataFrame,
    harmonized_df: pd.DataFrame | None = None,
    multi_year_scores: pd.DataFrame | None = None,
    multi_year_components: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run all validation modules and aggregate results.

    Args:
        scored_df: DataFrame with fips index and 'cci_score' column.
        harmonized_df: Multi-year harmonized data.
        multi_year_scores: CCI scores for multiple years (for stability).
        multi_year_components: Per-component scores for multiple years.
        settings: Pipeline settings.

    Returns:
        Dict keyed by module name, each value is the module's result
        DataFrame or an error dict on failure.
    """
    if settings is None:
        settings = get_settings()

    results: dict = {}

    # 1. External criteria validation
    try:
        results["external_criteria"] = run_external_validation(
            scored_df, harmonized_df, settings,
        )
        logger.info("External criteria validation complete")
    except Exception as e:
        logger.error("External criteria validation failed: %s", e)
        results["external_criteria"] = {"status": "failed", "error": str(e)}

    # 2. Convergent/divergent validation
    try:
        results["convergent_divergent"] = run_convergent_divergent(scored_df, settings)
        logger.info("Convergent/divergent validation complete")
    except Exception as e:
        logger.error("Convergent/divergent validation failed: %s", e)
        results["convergent_divergent"] = {"status": "failed", "error": str(e)}

    # 3. Stability analysis
    if multi_year_scores is not None:
        try:
            # If multi_year_components provided, merge component cols in
            if multi_year_components is not None:
                stability_input = multi_year_scores.merge(
                    multi_year_components, on=["fips", "year"], how="left",
                    suffixes=("", "_comp"),
                )
            else:
                stability_input = multi_year_scores
            results["stability"] = run_stability_analysis(stability_input)
            logger.info("Stability analysis complete")
        except Exception as e:
            logger.error("Stability analysis failed: %s", e)
            results["stability"] = {"status": "failed", "error": str(e)}
    else:
        results["stability"] = pd.DataFrame([{
            "metric": "all",
            "status": "no_multi_year_data",
            "note": "Stability requires multi-year scores. Pass multi_year_scores.",
        }])

    return results


def generate_validation_report(results: dict) -> pd.DataFrame:
    """Format validation results into a summary table.

    Args:
        results: Dict from run_validation_suite.

    Returns:
        Summary DataFrame with columns: test_category, test_name, result,
        threshold, passes, status.
    """
    summary_rows: list[dict] = []

    for category, data in results.items():
        if isinstance(data, dict):
            # Error or simple status
            summary_rows.append({
                "test_category": category,
                "test_name": "all",
                "result": None,
                "threshold": None,
                "passes": None,
                "status": data.get("status", "unknown"),
            })
            continue

        if not isinstance(data, pd.DataFrame) or data.empty:
            summary_rows.append({
                "test_category": category,
                "test_name": "all",
                "result": None,
                "threshold": None,
                "passes": None,
                "status": "empty",
            })
            continue

        if category == "external_criteria":
            for _, row in data.iterrows():
                summary_rows.append({
                    "test_category": category,
                    "test_name": row.get("indicator", "unknown"),
                    "result": row.get("pearson_r"),
                    "threshold": f"[{row.get('expected_r_low')}, {row.get('expected_r_high')}]"
                                 if row.get("expected_r_low") is not None else None,
                    "passes": row.get("within_expected"),
                    "status": row.get("status", "unknown"),
                })

        elif category == "convergent_divergent":
            for _, row in data.iterrows():
                summary_rows.append({
                    "test_category": category,
                    "test_name": f"{row.get('benchmark')}_{row.get('validity_type')}",
                    "result": row.get("pearson_r"),
                    "threshold": f"[{row.get('target_r_low')}, {row.get('target_r_high')}]"
                                 if row.get("target_r_low") is not None else None,
                    "passes": row.get("within_target"),
                    "status": row.get("status", "unknown"),
                })

        elif category == "stability":
            for _, row in data.iterrows():
                summary_rows.append({
                    "test_category": category,
                    "test_name": row.get("metric", "unknown"),
                    "result": row.get("value"),
                    "threshold": row.get("threshold"),
                    "passes": row.get("passes"),
                    "status": row.get("status", "unknown"),
                })

        else:
            # Generic fallback
            for _, row in data.iterrows():
                summary_rows.append({
                    "test_category": category,
                    "test_name": str(row.to_dict()),
                    "result": None,
                    "threshold": None,
                    "passes": None,
                    "status": row.get("status", "unknown") if "status" in row.index else "unknown",
                })

    summary = pd.DataFrame(summary_rows)

    # Log overall counts
    n_passed = summary["passes"].sum() if "passes" in summary.columns else 0
    n_failed = ((summary["passes"] == False) & (summary["status"] == "success")).sum()  # noqa: E712
    n_skipped = (summary["status"].isin(["data_unavailable", "no_multi_year_data",
                                          "no_scored_data", "insufficient_years",
                                          "no_component_data"])).sum()
    n_errored = (summary["status"] == "failed").sum()

    logger.info(
        "Validation summary: %d passed, %d failed, %d skipped, %d errored",
        n_passed, n_failed, n_skipped, n_errored,
    )

    return summary


def save_validation_outputs(
    results: dict,
    summary: pd.DataFrame,
    settings: Settings | None = None,
) -> None:
    """Save all validation outputs to the validation directory.

    Args:
        results: Dict from run_validation_suite.
        summary: Summary DataFrame from generate_validation_report.
        settings: Pipeline settings.
    """
    if settings is None:
        settings = get_settings()

    val_dir = settings.validation_dir
    val_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary.to_parquet(val_dir / "validation_summary.parquet", index=False)

    # Detail files
    for key, fname in [
        ("external_criteria", "external_criteria_detail.parquet"),
        ("convergent_divergent", "convergent_divergent_detail.parquet"),
        ("stability", "stability_detail.parquet"),
    ]:
        data = results.get(key)
        if isinstance(data, pd.DataFrame) and not data.empty:
            data.to_parquet(val_dir / fname, index=False)

    # Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modules": {},
    }
    for key, data in results.items():
        if isinstance(data, dict):
            metadata["modules"][key] = data
        elif isinstance(data, pd.DataFrame):
            statuses = data["status"].value_counts().to_dict() if "status" in data.columns else {}
            metadata["modules"][key] = {
                "n_rows": len(data),
                "statuses": {str(k): int(v) for k, v in statuses.items()},
            }

    with open(val_dir / "validation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Validation outputs saved to %s", val_dir)
