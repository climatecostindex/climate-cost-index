"""Validation phase: run sensitivity and validation suites."""

from __future__ import annotations

import logging
import sys

import pandas as pd

from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_validate() -> None:
    settings = get_settings()
    logger.info("=== Phase 4-5: Sensitivity + Validation (year=%d) ===", settings.scoring_year)

    # Load harmonized input
    input_path = settings.harmonized_dir / f"cci_input_{settings.scoring_year}.parquet"
    if not input_path.exists():
        logger.error("Harmonized input not found: %s", input_path)
        sys.exit(1)
    harmonized_df = pd.read_parquet(input_path)

    # Phase 4: Sensitivity Suite
    from sensitivity.report import run_sensitivity_suite, generate_sensitivity_report

    logger.info("--- Phase 4: Sensitivity Suite ---")
    sensitivity_results = run_sensitivity_suite(harmonized_df)
    sensitivity_summary = generate_sensitivity_report(sensitivity_results)

    # Save sensitivity outputs
    sens_dir = settings.validation_dir / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_summary.to_parquet(sens_dir / "sensitivity_summary.parquet")
    logger.info("Sensitivity summary saved to %s", sens_dir)

    # Phase 5: Validation Suite
    from validate.report import run_validation_suite, generate_validation_report, save_validation_outputs

    logger.info("--- Phase 5: Validation Suite ---")

    scored_path = settings.scored_dir / "cci_scores.parquet"
    if scored_path.exists():
        scored_df = pd.read_parquet(scored_path)

        validation_results = run_validation_suite(
            scored_df=scored_df,
            harmonized_df=harmonized_df,
            settings=settings,
        )
        validation_summary = generate_validation_report(validation_results)
        save_validation_outputs(validation_results, validation_summary, settings)
        logger.info("Validation summary: %d rows", len(validation_summary))
    else:
        logger.error("Scored data not found: %s — skipping validation", scored_path)

    logger.info("=== Phase 4-5 complete ===")


if __name__ == "__main__":
    try:
        run_validate()
    except Exception:
        logger.exception("Validation failed")
        sys.exit(1)
