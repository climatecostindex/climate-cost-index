"""Component definitions: names, sources, transforms, weights, attribution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoreTier(str, Enum):
    """Whether a component is required or preferred for scoring universe membership."""

    REQUIRED = "required"    # Must have data to be in scoring universe
    PREFERRED = "preferred"  # Imputed to 0 if missing


class Attribution(str, Enum):
    """Whether a component's climate link is directly attributed or a proxy."""

    ATTRIBUTED = "attributed"
    PROXY = "proxy"


class OverlapPrecedenceTier(int, Enum):
    """Precedence hierarchy for overlap penalty resolution.

    Tier 1 wins over Tier 2, etc. Within a tier: higher confidence wins,
    then larger CE weight, then lower measurement error.
    """

    DIRECT_DOLLAR_ATTRIBUTED = 1
    HAZARD_BURDEN_PROXY = 2
    GENERAL_EXPOSURE = 3


@dataclass(frozen=True)
class ComponentDef:
    """Immutable definition of a CCI component."""

    id: str
    name: str
    source_module: str  # e.g. "ingest.noaa_ncei"
    attribution: Attribution
    confidence: str  # A, B, or C
    precedence_tier: OverlapPrecedenceTier
    base_weight: float  # from BLS CE expenditure shares, normalized
    transform: str  # "log", "identity", "sqrt"
    acceleration_window: int  # years for Theil-Sen slope
    unit: str


# ---------------------------------------------------------------------------
# Component registry
#
# base_weight values are placeholders derived from BLS Consumer Expenditure
# shares for CCI-relevant categories. They will be finalized when
# ingest/bls_ce.py computes actual shares and must sum to 1.0.
# ---------------------------------------------------------------------------
COMPONENTS: dict[str, ComponentDef] = {
    "hdd_anomaly": ComponentDef(
        id="hdd_anomaly",
        name="Heating Degree-Day Anomaly",
        source_module="ingest.noaa_ncei",
        attribution=Attribution.ATTRIBUTED,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.DIRECT_DOLLAR_ATTRIBUTED,
        base_weight=0.10,
        transform="identity",
        acceleration_window=5,
        unit="degree-days",
    ),
    "cdd_anomaly": ComponentDef(
        id="cdd_anomaly",
        name="Cooling Degree-Day Anomaly",
        source_module="ingest.noaa_ncei",
        attribution=Attribution.ATTRIBUTED,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.DIRECT_DOLLAR_ATTRIBUTED,
        base_weight=0.10,
        transform="identity",
        acceleration_window=5,
        unit="degree-days",
    ),
    "extreme_heat_days": ComponentDef(
        id="extreme_heat_days",
        name="Extreme Heat Days",
        source_module="transform.extreme_heat",
        attribution=Attribution.PROXY,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.05,
        transform="sqrt",
        acceleration_window=5,
        unit="days",
    ),
    "storm_severity": ComponentDef(
        id="storm_severity",
        name="Storm Severity Score",
        source_module="transform.storm_severity",
        attribution=Attribution.PROXY,
        confidence="B",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.12,
        transform="log",
        acceleration_window=10,
        unit="severity-weighted events per housing unit",
    ),
    "pm25_annual": ComponentDef(
        id="pm25_annual",
        name="PM2.5 Annual Average",
        source_module="ingest.epa_airnow",
        attribution=Attribution.ATTRIBUTED,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.06,
        transform="identity",
        acceleration_window=5,
        unit="µg/m³",
    ),
    "aqi_unhealthy_days": ComponentDef(
        id="aqi_unhealthy_days",
        name="AQI Unhealthy Days",
        source_module="ingest.epa_airnow",
        attribution=Attribution.ATTRIBUTED,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.05,
        transform="sqrt",
        acceleration_window=5,
        unit="days",
    ),
    "flood_exposure": ComponentDef(
        id="flood_exposure",
        name="Flood Exposure Score",
        source_module="transform.flood_zone_scoring",
        attribution=Attribution.PROXY,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.10,
        transform="identity",
        acceleration_window=10,
        unit="composite score",
    ),
    "wildfire_score": ComponentDef(
        id="wildfire_score",
        name="Wildfire Hazard Score",
        source_module="ingest.usfs_wildfire",
        attribution=Attribution.PROXY,
        confidence="B",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.08,
        transform="identity",
        acceleration_window=10,
        unit="composite score",
    ),
    "drought_score": ComponentDef(
        id="drought_score",
        name="Drought Severity Score",
        source_module="ingest.usdm_drought",
        attribution=Attribution.PROXY,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.08,
        transform="identity",
        acceleration_window=5,
        unit="severity × area × weeks",
    ),
    "energy_cost_attributed": ComponentDef(
        id="energy_cost_attributed",
        name="Climate-Attributed Energy Cost",
        source_module="transform.energy_attribution",
        attribution=Attribution.ATTRIBUTED,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.DIRECT_DOLLAR_ATTRIBUTED,
        base_weight=0.15,
        transform="identity",
        acceleration_window=5,
        unit="$/household/year",
    ),
    "health_burden": ComponentDef(
        id="health_burden",
        name="Heat-Related Health Burden",
        source_module="ingest.cdc_epht",
        attribution=Attribution.PROXY,
        confidence="B",
        precedence_tier=OverlapPrecedenceTier.GENERAL_EXPOSURE,
        base_weight=0.06,
        transform="log",
        acceleration_window=5,
        unit="ED visits per 100k",
    ),
    "fema_ia_burden": ComponentDef(
        id="fema_ia_burden",
        name="FEMA Individual Assistance Burden",
        source_module="transform.fema_ia_burden",
        attribution=Attribution.PROXY,
        confidence="A",
        precedence_tier=OverlapPrecedenceTier.HAZARD_BURDEN_PROXY,
        base_weight=0.05,
        transform="log",
        acceleration_window=10,
        unit="$/housing unit",
    ),
}


def get_weights() -> dict[str, float]:
    """Return {component_id: base_weight} dict. Weights sum to 1.0."""
    raw = {c.id: c.base_weight for c in COMPONENTS.values()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}
