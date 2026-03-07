"""Transform phase: harmonize raw data to county-year grain."""

from __future__ import annotations

import logging
import sys

from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_transform() -> None:
    settings = get_settings()
    logger.info("=== Phase 2: Transform (year=%d) ===", settings.scoring_year)

    # TODO: run transforms in dependency order
    # 1. station_to_county (needs noaa_ncei raw + TIGER)
    # 2. monitor_to_county (needs epa_airnow raw + TIGER)
    # 3. degree_days (needs noaa_ncei raw + station map + normals)
    # 4. extreme_heat (needs noaa_ncei raw + station map)
    # 5. event_severity_tiers (needs ncei_storms raw)
    # 6. storm_severity (needs tiered events + fema_ia + census_acs)
    # 7. drought_scoring (needs usdm raw)
    # 8. flood_zone_scoring (needs fema_nfhl raw + TIGER + census_blocks)
    # 9. wildfire_scoring (needs usfs_wildfire raw + TIGER)
    # 10. air_quality_scoring (needs epa_airnow raw + monitor map + noaa_hms)
    # 11. health_burden (needs cdc_epht raw + census_acs)
    # 12. energy_attribution (needs eia_energy raw + degree_days output)
    # 13. harmonize (merges all above)

    logger.info("=== Phase 2 complete ===")


if __name__ == "__main__":
    try:
        run_transform()
    except Exception:
        logger.exception("Transform failed")
        sys.exit(1)
