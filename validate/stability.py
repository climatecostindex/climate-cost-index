"""Score stability metrics (requires >= 2 years of scores).

- Spearman rank persistence year-over-year (target > 0.90)
- Top/bottom decile turnover (target < 20%)
- Score band stability (5-year window)
- Component contribution rank persistence
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _rank_persistence(scores_df: pd.DataFrame) -> list[dict]:
    """Spearman rank correlation between consecutive annual releases (SSRN 12.1)."""
    years = sorted(scores_df["year"].unique())
    results = []

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        s1 = scores_df.loc[scores_df["year"] == y1].set_index("fips")["cci_score"]
        s2 = scores_df.loc[scores_df["year"] == y2].set_index("fips")["cci_score"]
        common = s1.index.intersection(s2.index)
        n = len(common)

        if n < 3:
            results.append({
                "year_pair": f"{y1}-{y2}",
                "spearman_r": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "passes": None,
                "n": n,
            })
            continue

        sr, _ = stats.spearmanr(s1.loc[common], s2.loc[common])
        sr = float(sr)

        # Fisher z-transform confidence interval
        z = np.arctanh(sr)
        se = 1.0 / np.sqrt(n - 3)
        z_low = z - 1.96 * se
        z_high = z + 1.96 * se
        ci_low = float(np.tanh(z_low))
        ci_high = float(np.tanh(z_high))

        results.append({
            "year_pair": f"{y1}-{y2}",
            "spearman_r": sr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "passes": sr > 0.90,
            "n": n,
        })

    return results


def _decile_turnover(scores_df: pd.DataFrame) -> list[dict]:
    """Percentage of counties exiting top/bottom 10% between years (SSRN 12.2)."""
    years = sorted(scores_df["year"].unique())
    results = []

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        s1 = scores_df.loc[scores_df["year"] == y1].set_index("fips")["cci_score"]
        s2 = scores_df.loc[scores_df["year"] == y2].set_index("fips")["cci_score"]
        common = s1.index.intersection(s2.index)

        if len(common) < 10:
            for decile in ["top", "bottom"]:
                results.append({
                    "year_pair": f"{y1}-{y2}",
                    "decile": decile,
                    "turnover_pct": np.nan,
                    "threshold": 20.0,
                    "passes": None,
                })
            continue

        s1c = s1.loc[common]
        s2c = s2.loc[common]

        for decile in ["top", "bottom"]:
            if decile == "top":
                cutoff1 = s1c.quantile(0.9)
                in_decile_y1 = set(s1c[s1c >= cutoff1].index)
                cutoff2 = s2c.quantile(0.9)
                in_decile_y2 = set(s2c[s2c >= cutoff2].index)
            else:
                cutoff1 = s1c.quantile(0.1)
                in_decile_y1 = set(s1c[s1c <= cutoff1].index)
                cutoff2 = s2c.quantile(0.1)
                in_decile_y2 = set(s2c[s2c <= cutoff2].index)

            if len(in_decile_y1) == 0:
                results.append({
                    "year_pair": f"{y1}-{y2}",
                    "decile": decile,
                    "turnover_pct": np.nan,
                    "threshold": 20.0,
                    "passes": None,
                })
                continue

            exits = len(in_decile_y1 - in_decile_y2)
            turnover = exits / len(in_decile_y1) * 100.0

            results.append({
                "year_pair": f"{y1}-{y2}",
                "decile": decile,
                "turnover_pct": turnover,
                "threshold": 20.0,
                "passes": turnover < 20.0,
            })

    return results


def _score_band_stability(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Score band stability over trailing 5-year window (SSRN 12.3)."""
    years = sorted(scores_df["year"].unique())
    if len(years) < 5:
        return pd.DataFrame([{
            "fips": None,
            "modal_band": None,
            "years_in_modal_band": None,
            "stability_class": None,
            "status": "insufficient_years",
            "note": f"Band stability requires >=5 years. Found {len(years)}.",
        }])

    # Use most recent 5 years
    recent = years[-5:]
    window = scores_df.loc[scores_df["year"].isin(recent)]

    rows = []
    for fips, group in window.groupby("fips"):
        bands = (group["cci_score"] // 10 * 10).astype(int)
        modal_band = int(bands.mode().iloc[0]) if not bands.empty else None
        years_in_modal = int((bands == modal_band).sum()) if modal_band is not None else 0
        n_unique_bands = bands.nunique()

        if years_in_modal >= 4:
            stability_class = "high_stability"
        elif n_unique_bands >= 3:
            stability_class = "low_stability"
        else:
            stability_class = "moderate_stability"

        rows.append({
            "fips": fips,
            "modal_band": modal_band,
            "years_in_modal_band": years_in_modal,
            "stability_class": stability_class,
        })

    return pd.DataFrame(rows)


def _component_contribution_stability(
    multi_year_components: pd.DataFrame,
) -> pd.DataFrame:
    """Track rank ordering of component contributions YoY (SSRN 12.4)."""
    if multi_year_components.empty:
        return pd.DataFrame()

    # Identify component columns (exclude fips, year, metadata)
    meta_cols = {"fips", "year", "cci_score", "raw_composite"}
    comp_cols = [c for c in multi_year_components.columns
                 if c not in meta_cols and not c.endswith("__confidence")
                 and not c.endswith("__attribution")]

    if not comp_cols:
        return pd.DataFrame()

    years = sorted(multi_year_components["year"].unique())
    rows = []

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        df1 = multi_year_components.loc[
            multi_year_components["year"] == y1
        ].set_index("fips")[comp_cols]
        df2 = multi_year_components.loc[
            multi_year_components["year"] == y2
        ].set_index("fips")[comp_cols]
        common = df1.index.intersection(df2.index)

        for fips in common:
            vals1 = df1.loc[fips].abs()
            vals2 = df2.loc[fips].abs()

            if vals1.isna().all() or vals2.isna().all():
                continue

            driver1 = vals1.idxmax()
            driver2 = vals2.idxmax()

            # Spearman correlation of component rank ordering
            ranks1 = vals1.rank(ascending=False)
            ranks2 = vals2.rank(ascending=False)
            valid = ranks1.notna() & ranks2.notna()
            if valid.sum() >= 3:
                sr, _ = stats.spearmanr(ranks1[valid], ranks2[valid])
            else:
                sr = np.nan

            rows.append({
                "fips": fips,
                "year_pair": f"{y1}-{y2}",
                "component_rank_spearman_r": float(sr) if not np.isnan(sr) else np.nan,
                "primary_driver_year1": driver1,
                "primary_driver_year2": driver2,
                "driver_changed": driver1 != driver2,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_stability_analysis(multi_year_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute rank persistence, decile turnover, and band stability.

    Args:
        multi_year_scores: DataFrame with columns: fips, year, cci_score,
            and optionally per-component score columns.

    Returns:
        DataFrame with stability metrics. If fewer than 2 years,
        returns a single row with status='insufficient_years'.
    """
    required = {"fips", "year", "cci_score"}
    missing = required - set(multi_year_scores.columns)
    if missing:
        return pd.DataFrame([{
            "metric": "all",
            "year_pair": "N/A",
            "value": None,
            "threshold": None,
            "passes": None,
            "status": "missing_columns",
            "note": f"Missing required columns: {missing}",
        }])

    n_years = multi_year_scores["year"].nunique()
    if n_years < 2:
        return pd.DataFrame([{
            "metric": "all",
            "year_pair": "N/A",
            "value": None,
            "threshold": None,
            "passes": None,
            "status": "insufficient_years",
            "note": f"Stability analysis requires >=2 years. Found {n_years}.",
        }])

    rows: list[dict] = []

    # Metric 1: Rank persistence
    for rp in _rank_persistence(multi_year_scores):
        rows.append({
            "metric": "rank_persistence",
            "year_pair": rp["year_pair"],
            "value": rp["spearman_r"],
            "threshold": 0.90,
            "passes": rp["passes"],
            "status": "success" if rp["passes"] is not None else "insufficient_data",
            "note": f"CI=[{rp['ci_low']:.3f}, {rp['ci_high']:.3f}]"
                    if rp["ci_low"] is not None and not np.isnan(rp.get("ci_low", np.nan))
                    else "",
        })

    # Metric 2: Decile turnover
    for dt in _decile_turnover(multi_year_scores):
        rows.append({
            "metric": f"decile_turnover_{dt['decile']}",
            "year_pair": dt["year_pair"],
            "value": dt["turnover_pct"],
            "threshold": 20.0,
            "passes": dt["passes"],
            "status": "success" if dt["passes"] is not None else "insufficient_data",
            "note": "",
        })

    # Metric 3: Score band stability
    band_df = _score_band_stability(multi_year_scores)
    if "status" in band_df.columns and band_df.iloc[0].get("status") == "insufficient_years":
        rows.append({
            "metric": "score_band_stability",
            "year_pair": "N/A",
            "value": None,
            "threshold": None,
            "passes": None,
            "status": "insufficient_years",
            "note": band_df.iloc[0].get("note", ""),
        })
    elif not band_df.empty:
        n_high = (band_df["stability_class"] == "high_stability").sum()
        n_low = (band_df["stability_class"] == "low_stability").sum()
        n_total = len(band_df)
        rows.append({
            "metric": "score_band_stability",
            "year_pair": "trailing_5yr",
            "value": n_high / n_total * 100 if n_total > 0 else None,
            "threshold": None,
            "passes": None,
            "status": "success",
            "note": f"{n_high} high-stability, {n_low} low-stability out of {n_total} counties",
        })

    # Metric 4: Component contribution stability
    meta_cols = {"fips", "year", "cci_score", "raw_composite"}
    comp_cols = [c for c in multi_year_scores.columns
                 if c not in meta_cols and not c.endswith("__confidence")
                 and not c.endswith("__attribution")]

    if comp_cols:
        comp_df = _component_contribution_stability(multi_year_scores)
        if not comp_df.empty:
            n_changed = comp_df["driver_changed"].sum()
            n_total = len(comp_df)
            mean_r = comp_df["component_rank_spearman_r"].mean()
            rows.append({
                "metric": "component_contribution_stability",
                "year_pair": "all",
                "value": mean_r,
                "threshold": None,
                "passes": None,
                "status": "success",
                "note": f"{n_changed}/{n_total} counties changed primary driver",
            })
        else:
            rows.append({
                "metric": "component_contribution_stability",
                "year_pair": "N/A",
                "value": None,
                "threshold": None,
                "passes": None,
                "status": "no_component_data",
                "note": "No component data available for stability analysis",
            })
    else:
        rows.append({
            "metric": "component_contribution_stability",
            "year_pair": "N/A",
            "value": None,
            "threshold": None,
            "passes": None,
            "status": "no_component_data",
            "note": "No component columns found in multi_year_scores",
        })

    logger.info("Stability analysis: %d metrics computed", len(rows))
    return pd.DataFrame(rows)
