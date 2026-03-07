"""Construct validity: CCI vs FEMA NRI, First Street.

Convergent: CCI vs FEMA NRI (target r = 0.4-0.7)
Convergent: CCI vs First Street composite (target r = 0.4-0.7)
Divergent: r < 0.85 (CCI is distinct, not a duplicate)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# FEMA NRI common file patterns
_NRI_FILENAMES = [
    "NRI_Table_Counties.csv",
    "nri_counties.parquet",
    "NRI_Table_Counties.parquet",
    "nri_table_counties.csv",
]


def _interpret(r: float) -> str:
    """Return interpretation string based on correlation magnitude."""
    r_abs = abs(r)
    if r_abs < 0.3:
        return "WARNING: CCI may not capture hazard exposure"
    elif r_abs < 0.4:
        return "Below convergent target but shows meaningful correlation"
    elif r_abs <= 0.7:
        return "Within convergent target range"
    elif r_abs < 0.85:
        return "Above convergent target — higher overlap than expected"
    else:
        return "WARNING: CCI may be duplicating existing metric"


def _make_row(
    benchmark: str,
    validity_type: str,
    pearson_r: float | None = None,
    spearman_r: float | None = None,
    target_r_low: float | None = None,
    target_r_high: float | None = None,
    within_target: bool | None = None,
    interpretation: str = "",
    status: str = "success",
    note: str = "",
) -> dict:
    return {
        "benchmark": benchmark,
        "validity_type": validity_type,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "target_r_low": target_r_low,
        "target_r_high": target_r_high,
        "within_target": within_target,
        "interpretation": interpretation,
        "status": status,
        "note": note,
    }


def _load_nri(raw_dir: Path) -> pd.DataFrame | None:
    """Attempt to load FEMA NRI data from common file locations."""
    nri_dir = raw_dir / "fema_nri"
    if not nri_dir.exists():
        return None

    for fname in _NRI_FILENAMES:
        fpath = nri_dir / fname
        if fpath.exists():
            try:
                if fpath.suffix == ".csv":
                    df = pd.read_csv(fpath)
                else:
                    df = pd.read_parquet(fpath)
                logger.info("Loaded NRI data from %s (%d rows)", fpath, len(df))
                return df
            except Exception as e:
                logger.warning("Failed to load NRI from %s: %s", fpath, e)
    return None


def _normalize_nri_fips(nri_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NRI FIPS column to match CCI format."""
    # NRI uses various column names for FIPS
    fips_col = None
    for col in ["STCOFIPS", "stcofips", "FIPS", "fips", "COUNTYCODE"]:
        if col in nri_df.columns:
            fips_col = col
            break
    if fips_col is None:
        return pd.DataFrame()

    # NRI risk score column
    score_col = None
    for col in ["RISK_SCORE", "risk_score", "RISK_VALUE", "risk_value"]:
        if col in nri_df.columns:
            score_col = col
            break
    if score_col is None:
        return pd.DataFrame()

    result = nri_df[[fips_col, score_col]].copy()
    result.columns = ["fips", "nri_score"]
    result["fips"] = result["fips"].astype(str).str.zfill(5)
    result = result.set_index("fips")
    result["nri_score"] = pd.to_numeric(result["nri_score"], errors="coerce")
    return result.dropna()


def _validate_nri(
    scored_df: pd.DataFrame,
    raw_dir: Path,
) -> list[dict]:
    """Validate CCI against FEMA National Risk Index."""
    nri_raw = _load_nri(raw_dir)
    if nri_raw is None:
        return [
            _make_row("fema_nri", "convergent", status="data_unavailable",
                       target_r_low=0.4, target_r_high=0.7,
                       note="FEMA NRI data not found in data/raw/fema_nri/"),
            _make_row("fema_nri", "divergent", status="data_unavailable",
                       target_r_low=0.0, target_r_high=0.85,
                       note="FEMA NRI data not found in data/raw/fema_nri/"),
        ]

    nri = _normalize_nri_fips(nri_raw)
    if nri.empty:
        return [
            _make_row("fema_nri", "convergent", status="failed",
                       target_r_low=0.4, target_r_high=0.7,
                       note="Could not identify FIPS/score columns in NRI data"),
            _make_row("fema_nri", "divergent", status="failed",
                       target_r_low=0.0, target_r_high=0.85,
                       note="Could not identify FIPS/score columns in NRI data"),
        ]

    merged = scored_df[["cci_score"]].join(nri, how="inner")
    mask = merged["cci_score"].notna() & merged["nri_score"].notna()
    merged = merged[mask]
    n = len(merged)

    if n < 3:
        return [
            _make_row("fema_nri", "convergent", status="insufficient_data",
                       target_r_low=0.4, target_r_high=0.7,
                       note=f"Only {n} matched counties"),
            _make_row("fema_nri", "divergent", status="insufficient_data",
                       target_r_low=0.0, target_r_high=0.85,
                       note=f"Only {n} matched counties"),
        ]

    pr, pp = stats.pearsonr(merged["cci_score"], merged["nri_score"])
    sr, sp = stats.spearmanr(merged["cci_score"], merged["nri_score"])
    pr, sr = float(pr), float(sr)

    interp = _interpret(pr)
    convergent_pass = 0.4 <= abs(pr) <= 0.7
    divergent_pass = abs(pr) < 0.85

    return [
        _make_row("fema_nri", "convergent",
                   pearson_r=pr, spearman_r=sr,
                   target_r_low=0.4, target_r_high=0.7,
                   within_target=convergent_pass,
                   interpretation=interp),
        _make_row("fema_nri", "divergent",
                   pearson_r=pr, spearman_r=sr,
                   target_r_low=0.0, target_r_high=0.85,
                   within_target=divergent_pass,
                   interpretation=interp),
    ]


def run_convergent_divergent(
    scored_df: pd.DataFrame,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Compute convergent and divergent validity metrics.

    Args:
        scored_df: DataFrame with fips index and 'cci_score' column.
        settings: Pipeline settings (for file paths).

    Returns:
        DataFrame with two rows per benchmark (convergent + divergent).
    """
    if settings is None:
        settings = get_settings()

    if "fips" in scored_df.columns:
        scored_df = scored_df.set_index("fips")

    rows: list[dict] = []

    if scored_df.empty or "cci_score" not in scored_df.columns:
        logger.warning("scored_df is empty or missing cci_score")
        for bm in ["fema_nri", "first_street"]:
            for vt in ["convergent", "divergent"]:
                rows.append(_make_row(bm, vt, status="no_scored_data"))
        return pd.DataFrame(rows)

    # Benchmark 1: FEMA NRI
    rows.extend(_validate_nri(scored_df, settings.raw_dir))

    # Benchmark 2: First Street Foundation (NOT AVAILABLE)
    rows.append(_make_row(
        "first_street", "convergent",
        status="data_unavailable",
        target_r_low=0.4, target_r_high=0.7,
        note="First Street data requires partnership agreement",
    ))
    rows.append(_make_row(
        "first_street", "divergent",
        status="data_unavailable",
        target_r_low=0.0, target_r_high=0.85,
        note="First Street data requires partnership agreement",
    ))

    logger.info(
        "Convergent/divergent: %d tests computed, %d unavailable",
        sum(1 for r in rows if r["status"] == "success"),
        sum(1 for r in rows if r["status"] == "data_unavailable"),
    )

    return pd.DataFrame(rows)
