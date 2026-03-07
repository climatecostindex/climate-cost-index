"""Spatial interpolation to fill county-level data gaps.

Uses inverse-distance-weighted (IDW) averaging from neighboring counties
to fill missing component values. County centroids from Census TIGER
boundary files provide the distance basis.

Components eligible for spatial gap-fill:
  - hdd_anomaly, cdd_anomaly (degree-day coverage ~67%)
  - extreme_heat_days (same station source, ~72%)
  - pm25_annual, aqi_unhealthy_days (monitor-limited, ~20%)

Components NOT eligible:
  - flood_exposure, wildfire_score: spatial hazard scores, not point observations
  - drought_score: already near-universal (97%)
  - storm_severity: already near-universal (96%)
  - energy_cost_attributed: regression-based, not interpolable
  - health_burden: state-level data, handled by forward-fill
  - fema_ia_burden: missing = no events = zero burden

All interpolated values are flagged and confidence is downgraded.

Confidence: original grade → one step down (A→B, B→C) for interpolated values.
Attribution: unchanged (interpolation doesn't change the causal status).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)

# Components eligible for spatial IDW interpolation
IDW_COMPONENTS = [
    "hdd_anomaly", "cdd_anomaly",
    "extreme_heat_days",
    "pm25_annual", "aqi_unhealthy_days",
]

# Maximum number of neighbors for IDW
IDW_K = 8
# Maximum distance in km — per-component, reflecting spatial autocorrelation.
# Temperature: well-correlated to ~200 km (Janis et al. 2004, DeGaetano 2001).
# PM2.5: autocorrelation range ~100 km (Yanosky et al. 2009, semivariogram).
IDW_MAX_DISTANCE_KM = {
    "hdd_anomaly": 200,
    "cdd_anomaly": 200,
    "extreme_heat_days": 200,
    "pm25_annual": 100,
    "aqi_unhealthy_days": 100,
}
IDW_MAX_DISTANCE_KM_DEFAULT = 150
# IDW power parameter (higher = more weight to closer neighbors)
IDW_POWER = 2


def load_county_centroids() -> pd.DataFrame:
    """Load county centroids from Census boundary data.

    Returns:
        DataFrame with columns: fips, lat, lon.
    """
    path = RAW_DIR / "census_blocks" / "county_boundaries_2024.parquet"
    if not path.exists():
        logger.error("County boundaries not found: %s", path)
        return pd.DataFrame(columns=["fips", "lat", "lon"])

    df = pd.read_parquet(path)
    # Normalize column names
    fips_col = "county_fips" if "county_fips" in df.columns else "fips"
    result = df[[fips_col, "lat", "lon"]].copy()
    result = result.rename(columns={fips_col: "fips"})
    result["fips"] = result["fips"].astype(str).str.zfill(5)
    return result


def compute_distance_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Haversine distances between county centroids.

    Args:
        centroids: DataFrame with fips, lat, lon.

    Returns:
        DataFrame (n_counties × n_counties) with distances in km,
        indexed and columned by fips.
    """
    lat = np.radians(centroids["lat"].values)
    lon = np.radians(centroids["lon"].values)
    fips = centroids["fips"].values

    # Haversine vectorized (uses broadcasting)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    R = 6371.0  # Earth radius in km
    dist = R * c

    return pd.DataFrame(dist, index=fips, columns=fips)


def spatial_idw_fill(
    df: pd.DataFrame,
    components: list[str] | None = None,
    k: int = IDW_K,
    max_distance_km: dict[str, float] | float | None = None,
    power: float = IDW_POWER,
) -> pd.DataFrame:
    """Fill missing component values using IDW from neighboring counties.

    Operates on a single-year slice. For each county with a missing component
    value, finds the k nearest counties that have a value and computes an
    inverse-distance-weighted average.

    Args:
        df: Single-year DataFrame with fips index and component columns.
        components: Component IDs to interpolate. Defaults to IDW_COMPONENTS.
        k: Maximum number of neighbors.
        max_distance_km: Per-component or uniform max distance. Defaults to
            IDW_MAX_DISTANCE_KM dict (200km temp, 100km AQ).
        power: IDW power parameter.

    Returns:
        Updated DataFrame with interpolated values filled in.
        New columns:
        - '{component}__gap_filled' (bool): True for interpolated rows.
        - '{component}__nearest_monitor_km' (float): distance to nearest
          neighbor with data (for confidence tiering).
    """
    comp_ids = components or [c for c in IDW_COMPONENTS if c in df.columns]
    if not comp_ids:
        return df

    result = df.copy()

    # Load centroids and compute distances
    centroids = load_county_centroids()
    if centroids.empty:
        logger.warning("No county centroids available; skipping spatial gap fill")
        return result

    # Align centroids with the DataFrame's fips
    fips_in_df = set(result.index if result.index.name == "fips" else result["fips"])
    centroids = centroids[centroids["fips"].isin(fips_in_df)].drop_duplicates("fips")

    if len(centroids) < 2:
        logger.warning("Too few centroids for interpolation; skipping")
        return result

    logger.info("Computing pairwise distances for %d counties...", len(centroids))
    dist_matrix = compute_distance_matrix(centroids)

    # Resolve per-component max distance
    if max_distance_km is None:
        dist_limits = IDW_MAX_DISTANCE_KM
    elif isinstance(max_distance_km, (int, float)):
        dist_limits = {c: float(max_distance_km) for c in comp_ids}
    else:
        dist_limits = max_distance_km

    total_filled = 0
    for comp_id in comp_ids:
        if comp_id not in result.columns:
            continue

        comp_max_dist = dist_limits.get(comp_id, IDW_MAX_DISTANCE_KM_DEFAULT)

        is_fips_index = result.index.name == "fips"
        if is_fips_index:
            values = result[comp_id]
        else:
            values = result.set_index("fips")[comp_id]

        missing_fips = values[values.isna()].index
        valid_fips = values[values.notna()].index

        # Restrict to fips in distance matrix
        missing_fips = missing_fips[missing_fips.isin(dist_matrix.index)]
        valid_fips = valid_fips[valid_fips.isin(dist_matrix.index)]

        if len(missing_fips) == 0 or len(valid_fips) == 0:
            result[f"{comp_id}__gap_filled"] = False
            result[f"{comp_id}__nearest_monitor_km"] = np.nan
            continue

        filled_count = 0
        gap_filled_flags = pd.Series(False, index=result.index if is_fips_index else result["fips"])
        nearest_dist = pd.Series(np.nan, index=result.index if is_fips_index else result["fips"])

        for fips in missing_fips:
            # Get distances to all valid counties
            distances = dist_matrix.loc[fips, valid_fips]

            # Filter by max distance
            within_range = distances[distances <= comp_max_dist]
            if within_range.empty:
                continue

            # Take k nearest
            nearest = within_range.nsmallest(k)

            # IDW weights
            weights = 1.0 / (nearest.values ** power)
            neighbor_values = values.loc[nearest.index].values

            interpolated = np.average(neighbor_values, weights=weights)
            min_dist = nearest.iloc[0]

            if is_fips_index:
                result.loc[fips, comp_id] = interpolated
                gap_filled_flags.loc[fips] = True
                nearest_dist.loc[fips] = min_dist
            else:
                mask = result["fips"] == fips
                result.loc[mask, comp_id] = interpolated
                gap_filled_flags.loc[result.loc[mask].index] = True
                nearest_dist.loc[result.loc[mask].index] = min_dist

            filled_count += 1

        result[f"{comp_id}__gap_filled"] = gap_filled_flags.values
        result[f"{comp_id}__nearest_monitor_km"] = nearest_dist.values
        total_filled += filled_count

        if filled_count > 0:
            logger.info(
                "Spatial IDW: filled %d missing %s values (max_dist=%dkm, from %d valid neighbors)",
                filled_count, comp_id, comp_max_dist, len(valid_fips),
            )

    logger.info("Spatial gap fill complete: %d total values interpolated", total_filled)
    return result


def downgrade_interpolated_confidence(df: pd.DataFrame, components: list[str] | None = None) -> pd.DataFrame:
    """Downgrade confidence for spatially interpolated values.

    Distance-tiered for air quality (per literature):
      - <50 km:  A→B
      - 50-100 km: A→C, B→C

    Single-step downgrade for temperature components (spatially smooth):
      - Any distance: A→B, B→C

    Args:
        df: DataFrame with '{component}__gap_filled', '{component}__confidence',
            and '{component}__nearest_monitor_km' columns.
        components: Components to check. Defaults to IDW_COMPONENTS.

    Returns:
        DataFrame with updated confidence columns.
    """
    comp_ids = components or IDW_COMPONENTS
    aq_components = {"pm25_annual", "aqi_unhealthy_days"}
    downgrade_one = {"A": "B", "B": "C", "C": "C"}
    downgrade_two = {"A": "C", "B": "C", "C": "C"}
    result = df.copy()

    for comp_id in comp_ids:
        flag_col = f"{comp_id}__gap_filled"
        conf_col = f"{comp_id}__confidence"
        dist_col = f"{comp_id}__nearest_monitor_km"

        if flag_col not in result.columns or conf_col not in result.columns:
            continue

        mask = result[flag_col] == True  # noqa: E712
        if not mask.any():
            continue

        if comp_id in aq_components and dist_col in result.columns:
            # Distance-tiered downgrade for AQ
            close = mask & (result[dist_col] <= 50)
            far = mask & (result[dist_col] > 50)

            if close.any():
                result.loc[close, conf_col] = result.loc[close, conf_col].map(
                    lambda x: downgrade_one.get(x, x) if pd.notna(x) else x
                )
            if far.any():
                result.loc[far, conf_col] = result.loc[far, conf_col].map(
                    lambda x: downgrade_two.get(x, x) if pd.notna(x) else x
                )
            logger.info(
                "Downgraded confidence for %d interpolated %s values "
                "(%d close/B, %d far/C)",
                mask.sum(), comp_id, close.sum(), far.sum(),
            )
        else:
            # Single-step downgrade for temperature components
            result.loc[mask, conf_col] = result.loc[mask, conf_col].map(
                lambda x: downgrade_one.get(x, x) if pd.notna(x) else x
            )
            logger.info(
                "Downgraded confidence for %d interpolated %s values",
                mask.sum(), comp_id,
            )

    return result
