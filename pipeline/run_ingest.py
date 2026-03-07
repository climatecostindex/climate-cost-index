"""Data collection phase: run all ingesters."""

from __future__ import annotations

import logging
import sys

from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingest() -> None:
    settings = get_settings()
    logger.info("=== Phase 1: Data Ingestion (year=%d) ===", settings.scoring_year)

    # TODO: instantiate and run each ingester in dependency order
    # Each ingester should be run via `with IngesterClass() as ing: ing.run()`
    # Order:
    #   1. census_acs (needed as denominator for others)
    #   2. noaa_ncei (degree days, extreme heat)
    #   3. ncei_storms
    #   4. epa_airnow
    #   5. fema_nfhl
    #   6. usfs_wildfire
    #   7. usdm_drought
    #   8. eia_energy
    #   9. cdc_epht
    #  10. bls_ce (weights reference)
    #  11. fema_ia

    logger.info("=== Phase 1 complete ===")


if __name__ == "__main__":
    try:
        run_ingest()
    except Exception:
        logger.exception("Ingestion failed")
        sys.exit(1)
