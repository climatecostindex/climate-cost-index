"""Full end-to-end pipeline: ingest → transform → score → validate."""

from __future__ import annotations

import logging
import sys

from pipeline.run_ingest import run_ingest
from pipeline.run_transform import run_transform
from pipeline.run_score import run_score
from pipeline.run_validate import run_validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_full() -> None:
    logger.info("=== CCI Full Pipeline ===")
    run_ingest()
    run_transform()
    run_score()
    run_validate()
    logger.info("=== CCI Full Pipeline complete ===")


if __name__ == "__main__":
    try:
        run_full()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
