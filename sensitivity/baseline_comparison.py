"""Compare CCI scores using 1991-2020 vs 1981-2010 climate normals."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.components import get_weights
from config.settings import get_settings
from score.pipeline import compute_cci
from sensitivity.ranking_utils import compare_rankings

logger = logging.getLogger(__name__)

_ALT_COLS = {"hdd_anomaly": "hdd_anomaly__1981_2010", "cdd_anomaly": "cdd_anomaly__1981_2010"}


def run_baseline_comparison(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute with alternative normals baseline, compare rankings.

    Checks for alternative baseline (1981-2010) anomaly columns in the
    harmonized data. If unavailable, returns a placeholder result rather
    than crashing.

    Returns:
        DataFrame with columns: baseline, spearman_r_vs_primary,
        max_rank_shift, n_shifted_gt_10, status.
    """
    settings = get_settings()
    weights = get_weights()

    # Primary baseline row
    rows = [
        {
            "baseline": "1991-2020",
            "spearman_r_vs_primary": 1.0,
            "max_rank_shift": 0,
            "n_shifted_gt_10": 0,
            "status": "primary",
        }
    ]

    # Check for alternative baseline columns in harmonized data
    alt_available = all(col in harmonized_df.columns for col in _ALT_COLS.values())

    # Also check for alternative anomaly files on disk
    if not alt_available:
        alt_paths = [
            settings.harmonized_dir / "degree_days_1981_2010.parquet",
            settings.raw_dir / "noaa_ncei" / "normals_1981_2010.parquet",
        ]
        for p in alt_paths:
            if p.exists():
                logger.info("Found alternative baseline file: %s", p)
                try:
                    alt_df = pd.read_parquet(p)
                    for orig, alt_col in _ALT_COLS.items():
                        if alt_col in alt_df.columns:
                            continue
                        # Try without the prefix
                        if orig in alt_df.columns:
                            alt_df = alt_df.rename(columns={orig: alt_col})
                    if all(c in alt_df.columns for c in _ALT_COLS.values()):
                        merge_keys = [c for c in ("fips", "year") if c in alt_df.columns]
                        if merge_keys:
                            harmonized_df = harmonized_df.merge(
                                alt_df[merge_keys + list(_ALT_COLS.values())],
                                on=merge_keys,
                                how="left",
                            )
                            alt_available = True
                            break
                except Exception:
                    logger.warning("Could not load alternative baseline from %s", p, exc_info=True)

    if not alt_available:
        logger.warning(
            "Alternative baseline (1981-2010) anomaly data not available. "
            "Baseline comparison requires re-running degree_days transform "
            "with alternative normals. Returning placeholder result."
        )
        rows.append(
            {
                "baseline": "1981-2010",
                "spearman_r_vs_primary": np.nan,
                "max_rank_shift": np.nan,
                "n_shifted_gt_10": np.nan,
                "status": "data_unavailable",
            }
        )
        return pd.DataFrame(rows)

    # Compute primary CCI
    logger.info("Computing primary CCI (1991-2020 baseline)")
    primary_output = compute_cci(harmonized_df, weights, settings)

    # Compute alternative CCI with 1981-2010 anomalies
    logger.info("Computing alternative CCI (1981-2010 baseline)")
    alt_harmonized = harmonized_df.copy()
    for orig, alt_col in _ALT_COLS.items():
        alt_harmonized[orig] = alt_harmonized[alt_col]

    alt_output = compute_cci(alt_harmonized, weights, settings)

    # Compare rankings
    comparison = compare_rankings(
        primary_output.scores["cci_score"],
        alt_output.scores["cci_score"],
    )

    rows.append(
        {
            "baseline": "1981-2010",
            "spearman_r_vs_primary": comparison["spearman_r"],
            "max_rank_shift": comparison["max_rank_shift"],
            "n_shifted_gt_10": comparison["n_shifted_gt_10"],
            "status": "computed",
        }
    )

    return pd.DataFrame(rows)
