"""Scoring phase: load harmonized data → compute CCI scores."""

from __future__ import annotations

import logging
import sys

import pandas as pd

from config.components import get_weights
from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_score() -> None:
    settings = get_settings()
    logger.info("=== Phase 3: Score (year=%d) ===", settings.scoring_year)

    # Load harmonized input
    input_path = settings.harmonized_dir / f"cci_input_{settings.scoring_year}.parquet"
    if not input_path.exists():
        logger.error("Harmonized input not found: %s", input_path)
        sys.exit(1)

    harmonized_df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows from %s", len(harmonized_df), input_path)

    # Run deterministic scoring pipeline
    from score.pipeline import compute_cci

    weights = get_weights()
    results = compute_cci(harmonized_df, weights, settings)

    # Save outputs
    results.save(settings.scored_dir)
    logger.info("=== Phase 3 complete ===")


if __name__ == "__main__":
    try:
        run_score()
    except Exception:
        logger.exception("Scoring failed")
        sys.exit(1)
