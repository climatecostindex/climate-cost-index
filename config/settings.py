"""Global configuration: paths, API endpoints, thresholds."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HARMONIZED_DIR = DATA_DIR / "harmonized"
SCORED_DIR = DATA_DIR / "scored"
VALIDATION_DIR = DATA_DIR / "validation"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
NOAA_CDO_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
NOAA_STORMS_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"
EPA_AQS_BASE_URL = "https://aqs.epa.gov/data/api"
EIA_BASE_URL = "https://api.eia.gov/v2"
CENSUS_ACS_BASE_URL = "https://api.census.gov/data"
CDC_EPHT_BASE_URL = "https://ephtracking.cdc.gov/apigateway/api/v1"
FEMA_OPENFEMA_BASE_URL = "https://www.fema.gov/api/open/v2"


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------
class Settings(BaseModel):
    """Runtime configuration loaded from environment."""

    # API keys
    noaa_api_token: str = os.getenv("NOAA_API_TOKEN", "")
    epa_aqs_email: str = os.getenv("EPA_AQS_EMAIL", "")
    epa_aqs_key: str = os.getenv("EPA_AQS_KEY", "")
    eia_api_key: str = os.getenv("EIA_API_KEY", "")
    census_api_key: str = os.getenv("CENSUS_API_KEY", "")

    # Pipeline parameters
    methodology_version: str = "1.0"
    climate_normal_baseline: str = "1991-2020"
    scoring_year: int = 2024
    trailing_months: int = 12

    # Scoring thresholds
    winsorize_percentile: float = 99.0
    overlap_correlation_threshold: float = 0.6
    overlap_penalty_floor: float = 0.2
    acceleration_bounds: tuple[float, float] = (0.5, 3.0)
    acceleration_denominator_epsilon_factor: float = 0.1
    acceleration_min_completeness: float = 0.8  # 48/60 months
    acceleration_continuous_window: int = 5  # years
    acceleration_event_window: int = 10  # years
    target_iqr: tuple[float, float] = (80.0, 120.0)

    # Sensitivity
    monte_carlo_iterations: int = 10_000
    weight_perturbation_pct: float = 0.30

    # Data storage
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    harmonized_dir: Path = HARMONIZED_DIR
    scored_dir: Path = SCORED_DIR
    validation_dir: Path = VALIDATION_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
