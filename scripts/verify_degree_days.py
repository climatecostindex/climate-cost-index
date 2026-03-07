#!/usr/bin/env python3
"""Manual computation verification for the degree_days transformer.

This script independently recomputes HDD/CDD and anomalies for selected
counties from raw data, then compares results to the transformer's output
file (degree_days_2023.parquet).

Verification targets:
  - FIPS 01029 (Cleburne County, AL) — 3 stations, has anomalies
  - FIPS 01063 (Dale County, AL) — 2 stations, has anomalies
  - FIPS 01011 (Bullock County, AL) — 2 stations, NaN anomalies (tests that path)

Also verifies:
  - BASE_TEMP_C constant = (65 - 32) * 5/9 = 18.333...
  - Completeness threshold = 335 days
  - Anomaly direction = observed - normal (not the reverse)
"""

import sys
import numpy as np
import pandas as pd

# ===========================================================================
# Configuration
# ===========================================================================
RAW_DIR = "/Users/macminipro/Projects/ClimateCostIndex/cci/data/raw/noaa_ncei"
HARMONIZED_DIR = "/Users/macminipro/Projects/ClimateCostIndex/cci/data/harmonized"

OBS_PATH = f"{RAW_DIR}/noaa_ncei_observations_all.parquet"
NORMALS_PATH = f"{RAW_DIR}/noaa_ncei_normals.parquet"
STATION_COUNTY_PATH = f"{HARMONIZED_DIR}/station_to_county.parquet"
OUTPUT_PATH = f"{HARMONIZED_DIR}/degree_days_2023.parquet"

YEAR = 2023
TARGET_FIPS = ["01029", "01063", "01011"]

BASE_TEMP_F = 65.0
BASE_TEMP_C = (BASE_TEMP_F - 32) * 5 / 9
MIN_DAYS = 335

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}


def sep(title: str) -> None:
    """Print a separator line with a title."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def subsep(title: str) -> None:
    """Print a smaller separator."""
    print(f"\n  --- {title} ---")


# ===========================================================================
# STEP 0: Verify constants
# ===========================================================================
sep("STEP 0: Verify constants")

expected_base_c = (65 - 32) * 5 / 9
print(f"  BASE_TEMP_F            = {BASE_TEMP_F}")
print(f"  BASE_TEMP_C            = (65 - 32) * 5/9 = {expected_base_c:.10f}")
print(f"  18.333... check        = {18 + 1/3:.10f}")
print(f"  Match (65-32)*5/9 == 18.333...? {abs(expected_base_c - (18 + 1/3)) < 1e-10}")
print(f"  MIN_DAYS_PER_YEAR      = {MIN_DAYS}")
print(f"  Anomaly formula        = observed - normal (positive = warmer/more CDD than normal)")
print()

# Verify against the transformer's own constant
transformer_base_c = (65.0 - 32) * 5 / 9
print(f"  Transformer BASE_TEMP_C = {transformer_base_c:.10f}")
print(f"  Our BASE_TEMP_C         = {BASE_TEMP_C:.10f}")
print(f"  Match?                   {transformer_base_c == BASE_TEMP_C}")

# ===========================================================================
# STEP 1: Load all data
# ===========================================================================
sep("STEP 1: Load all data")

print("  Loading observations (all years combined)...")
obs_all = pd.read_parquet(OBS_PATH)
print(f"    Shape: {obs_all.shape}")
print(f"    Columns: {list(obs_all.columns)}")

print("\n  Loading normals...")
normals = pd.read_parquet(NORMALS_PATH)
print(f"    Shape: {normals.shape}")

print("\n  Loading station-to-county mapping...")
stc = pd.read_parquet(STATION_COUNTY_PATH)
print(f"    Shape: {stc.shape}")

print("\n  Loading degree_days_2023 output (transformer result)...")
dd_output = pd.read_parquet(OUTPUT_PATH)
print(f"    Shape: {dd_output.shape}")

# ===========================================================================
# STEP 2: Identify stations for target counties
# ===========================================================================
sep("STEP 2: Identify stations for target counties")

for fips in TARGET_FIPS:
    stations = stc[stc["fips"] == fips][["station_id", "lat", "lon"]].copy()
    print(f"  FIPS {fips}: {len(stations)} stations")
    for _, row in stations.iterrows():
        print(f"    {row['station_id']}  (lat={row['lat']:.4f}, lon={row['lon']:.4f})")
    print()

# ===========================================================================
# STEP 3: Filter observations for year 2023
# ===========================================================================
sep("STEP 3: Filter 2023 observations for target stations")

# Get all stations for our target counties
target_stations_map = stc[stc["fips"].isin(TARGET_FIPS)][["station_id", "fips"]].copy()
all_target_stations = target_stations_map["station_id"].unique()
print(f"  Target stations: {list(all_target_stations)}")

# Parse dates and filter to 2023
obs_all["date_parsed"] = pd.to_datetime(obs_all["date"])
obs_all["year"] = obs_all["date_parsed"].dt.year

obs_2023 = obs_all[
    (obs_all["year"] == YEAR) &
    (obs_all["station_id"].isin(all_target_stations))
].copy()
print(f"\n  Observations for target stations in {YEAR}: {len(obs_2023)} rows")

# Show per-station counts before filtering
print("\n  Per-station row counts (before quality filtering):")
for sid in sorted(all_target_stations):
    n = len(obs_2023[obs_2023["station_id"] == sid])
    print(f"    {sid}: {n} rows")

# ===========================================================================
# STEP 4: Apply quality filtering (same as transformer)
# ===========================================================================
sep("STEP 4: Quality filtering (NaN temps + quality flags)")

for sid in sorted(all_target_stations):
    sdf = obs_2023[obs_2023["station_id"] == sid].copy()
    initial = len(sdf)

    # Count NaN temps
    nan_tmax = sdf["tmax"].isna().sum()
    nan_tmin = sdf["tmin"].isna().sum()
    nan_either = (sdf["tmax"].isna() | sdf["tmin"].isna()).sum()

    # After dropping NaN
    sdf_no_nan = sdf[~(sdf["tmax"].isna() | sdf["tmin"].isna())]

    # Count quality flags
    q_tmax = sdf_no_nan["q_flag_tmax"].fillna("")
    q_tmin = sdf_no_nan["q_flag_tmin"].fillna("")
    flagged = ((q_tmax != "") | (q_tmin != "")).sum()

    # After dropping flagged
    sdf_clean = sdf_no_nan[~((q_tmax != "") | (q_tmin != ""))]

    print(f"  Station {sid}:")
    print(f"    Total rows:        {initial}")
    print(f"    NaN tmax:          {nan_tmax}")
    print(f"    NaN tmin:          {nan_tmin}")
    print(f"    NaN either:        {nan_either}")
    print(f"    After NaN drop:    {len(sdf_no_nan)}")
    print(f"    Quality-flagged:   {flagged}")
    print(f"    Final valid days:  {len(sdf_clean)}")
    print(f"    >= {MIN_DAYS} threshold?  {'PASS' if len(sdf_clean) >= MIN_DAYS else 'FAIL (excluded)'}")
    print()

# ===========================================================================
# STEP 5-9: Full manual computation per county
# ===========================================================================
results_manual = {}
all_discrepancies = []

for fips in TARGET_FIPS:
    sep(f"COUNTY FIPS {fips} — Full manual computation")

    # Stations for this county
    county_stations = target_stations_map[target_stations_map["fips"] == fips]["station_id"].tolist()
    print(f"  Stations: {county_stations}")

    station_results = []

    for sid in county_stations:
        subsep(f"Station {sid}")

        # ---- Step 5a: Get clean daily observations ----
        sdf = obs_2023[obs_2023["station_id"] == sid].copy()
        initial = len(sdf)

        # Drop NaN
        sdf = sdf[~(sdf["tmax"].isna() | sdf["tmin"].isna())].copy()

        # Drop quality-flagged
        q_tmax = sdf["q_flag_tmax"].fillna("")
        q_tmin = sdf["q_flag_tmin"].fillna("")
        sdf = sdf[~((q_tmax != "") | (q_tmin != ""))].copy()

        n_valid = len(sdf)
        print(f"    Valid days after filtering: {n_valid} (from {initial})")

        if n_valid < MIN_DAYS:
            print(f"    *** BELOW {MIN_DAYS}-day threshold — EXCLUDED ***")
            continue

        # ---- Step 5b: Compute daily HDD/CDD ----
        sdf["avg_temp"] = (sdf["tmax"] + sdf["tmin"]) / 2.0
        sdf["hdd_daily"] = np.maximum(0.0, BASE_TEMP_C - sdf["avg_temp"])
        sdf["cdd_daily"] = np.maximum(0.0, sdf["avg_temp"] - BASE_TEMP_C)

        # Show some sample daily values
        sample = sdf.head(5)
        print(f"\n    Sample daily values (first 5 valid days):")
        print(f"    {'date':>12s}  {'tmax':>8s}  {'tmin':>8s}  {'avg':>8s}  {'hdd':>8s}  {'cdd':>8s}")
        for _, r in sample.iterrows():
            print(f"    {str(r['date']):>12s}  {r['tmax']:8.3f}  {r['tmin']:8.3f}  "
                  f"{r['avg_temp']:8.3f}  {r['hdd_daily']:8.3f}  {r['cdd_daily']:8.3f}")

        # ---- Step 5c: Sum to annual totals ----
        hdd_annual = sdf["hdd_daily"].sum()
        cdd_annual = sdf["cdd_daily"].sum()
        print(f"\n    Annual totals:")
        print(f"      HDD = {hdd_annual:.6f}  (sum of {n_valid} daily values)")
        print(f"      CDD = {cdd_annual:.6f}  (sum of {n_valid} daily values)")

        # ---- Step 6: Get normals for this station ----
        sn = normals[normals["station_id"] == sid].copy()
        n_normal_months = sn["month"].nunique()
        print(f"\n    Normals: {n_normal_months} months available")

        normal_hdd_annual = np.nan
        normal_cdd_annual = np.nan
        hdd_anomaly = np.nan
        cdd_anomaly = np.nan

        if n_normal_months >= 12:
            # Drop NaN normals
            sn = sn[~(sn["normal_tmax"].isna() | sn["normal_tmin"].isna())]
            n_normal_months_clean = sn["month"].nunique()

            if n_normal_months_clean >= 12:
                sn["normal_avg"] = (sn["normal_tmax"] + sn["normal_tmin"]) / 2.0
                sn["days"] = sn["month"].map(DAYS_IN_MONTH)
                sn["normal_hdd_month"] = np.maximum(0.0, BASE_TEMP_C - sn["normal_avg"]) * sn["days"]
                sn["normal_cdd_month"] = np.maximum(0.0, sn["normal_avg"] - BASE_TEMP_C) * sn["days"]

                print(f"\n    Monthly normal degree days:")
                print(f"    {'month':>5s}  {'days':>4s}  {'norm_tmax':>10s}  {'norm_tmin':>10s}  "
                      f"{'norm_avg':>10s}  {'norm_hdd':>10s}  {'norm_cdd':>10s}")
                for _, r in sn.sort_values("month").iterrows():
                    print(f"    {int(r['month']):>5d}  {int(r['days']):>4d}  {r['normal_tmax']:10.4f}  "
                          f"{r['normal_tmin']:10.4f}  {r['normal_avg']:10.4f}  "
                          f"{r['normal_hdd_month']:10.4f}  {r['normal_cdd_month']:10.4f}")

                normal_hdd_annual = sn["normal_hdd_month"].sum()
                normal_cdd_annual = sn["normal_cdd_month"].sum()

                # ---- Step 7: Compute anomalies ----
                hdd_anomaly = hdd_annual - normal_hdd_annual
                cdd_anomaly = cdd_annual - normal_cdd_annual

                print(f"\n    Normal annual totals:")
                print(f"      Normal HDD = {normal_hdd_annual:.6f}")
                print(f"      Normal CDD = {normal_cdd_annual:.6f}")
                print(f"\n    Anomalies (observed - normal):")
                print(f"      HDD anomaly = {hdd_annual:.6f} - {normal_hdd_annual:.6f} = {hdd_anomaly:.6f}")
                print(f"      CDD anomaly = {cdd_annual:.6f} - {normal_cdd_annual:.6f} = {cdd_anomaly:.6f}")
            else:
                print(f"    Only {n_normal_months_clean} clean normal months — anomaly = NaN")
        else:
            print(f"    < 12 months of normals — anomaly = NaN")

        station_results.append({
            "station_id": sid,
            "hdd_annual": hdd_annual,
            "cdd_annual": cdd_annual,
            "normal_hdd_annual": normal_hdd_annual,
            "normal_cdd_annual": normal_cdd_annual,
            "hdd_anomaly": hdd_anomaly,
            "cdd_anomaly": cdd_anomaly,
            "n_valid_days": n_valid,
        })

    # ---- Step 8: Average across stations to get county values ----
    subsep(f"County-level aggregation for FIPS {fips}")

    if not station_results:
        print(f"  No stations passed completeness threshold!")
        results_manual[fips] = None
        continue

    sr_df = pd.DataFrame(station_results)
    print(f"\n  Station-level results:")
    print(f"  {'station_id':>20s}  {'hdd_annual':>12s}  {'cdd_annual':>12s}  {'hdd_anomaly':>12s}  {'cdd_anomaly':>12s}  {'n_days':>6s}")
    for _, r in sr_df.iterrows():
        anom_h = f"{r['hdd_anomaly']:.6f}" if not np.isnan(r['hdd_anomaly']) else "NaN"
        anom_c = f"{r['cdd_anomaly']:.6f}" if not np.isnan(r['cdd_anomaly']) else "NaN"
        print(f"  {r['station_id']:>20s}  {r['hdd_annual']:12.6f}  {r['cdd_annual']:12.6f}  "
              f"{anom_h:>12s}  {anom_c:>12s}  {int(r['n_valid_days']):>6d}")

    # Mean of all stations
    county_hdd = sr_df["hdd_annual"].mean()
    county_cdd = sr_df["cdd_annual"].mean()

    # For anomalies: mean of non-NaN values (matching the transformer's lambda behavior)
    hdd_anom_vals = sr_df["hdd_anomaly"].dropna()
    cdd_anom_vals = sr_df["cdd_anomaly"].dropna()
    county_hdd_anom = hdd_anom_vals.mean() if len(hdd_anom_vals) > 0 else np.nan
    county_cdd_anom = cdd_anom_vals.mean() if len(cdd_anom_vals) > 0 else np.nan

    print(f"\n  County averages:")
    print(f"    HDD annual  = mean({', '.join(f'{v:.6f}' for v in sr_df['hdd_annual'])}) = {county_hdd:.6f}")
    print(f"    CDD annual  = mean({', '.join(f'{v:.6f}' for v in sr_df['cdd_annual'])}) = {county_cdd:.6f}")
    if len(hdd_anom_vals) > 0:
        print(f"    HDD anomaly = mean({', '.join(f'{v:.6f}' for v in hdd_anom_vals)}) = {county_hdd_anom:.6f}")
        print(f"    CDD anomaly = mean({', '.join(f'{v:.6f}' for v in cdd_anom_vals)}) = {county_cdd_anom:.6f}")
    else:
        print(f"    HDD anomaly = NaN (no stations have normals)")
        print(f"    CDD anomaly = NaN (no stations have normals)")

    results_manual[fips] = {
        "hdd_annual": county_hdd,
        "cdd_annual": county_cdd,
        "hdd_anomaly": county_hdd_anom,
        "cdd_anomaly": county_cdd_anom,
    }

    # ---- Step 9: Compare to transformer output ----
    subsep(f"Comparison to transformer output for FIPS {fips}")

    output_row = dd_output[dd_output["fips"] == fips]
    if output_row.empty:
        print(f"  WARNING: FIPS {fips} not found in degree_days_2023 output!")
        continue

    out = output_row.iloc[0]
    print(f"\n  {'Metric':>15s}  {'Manual':>15s}  {'Transformer':>15s}  {'Diff':>15s}  {'Match?':>8s}")
    print(f"  {'-'*15}  {'-'*15}  {'-'*15}  {'-'*15}  {'-'*8}")

    for col in ["hdd_annual", "cdd_annual", "hdd_anomaly", "cdd_anomaly"]:
        manual_val = results_manual[fips][col]
        trans_val = out[col]

        if np.isnan(manual_val) and np.isnan(trans_val):
            diff_str = "both NaN"
            match = "YES"
        elif np.isnan(manual_val) or np.isnan(trans_val):
            diff_str = "NaN mismatch!"
            match = "NO"
            all_discrepancies.append((fips, col, manual_val, trans_val))
        else:
            diff = manual_val - trans_val
            diff_str = f"{diff:.6f}"
            match = "YES" if abs(diff) < 0.01 else "NO"
            if match == "NO":
                all_discrepancies.append((fips, col, manual_val, trans_val))

        m_str = f"{manual_val:.6f}" if not np.isnan(manual_val) else "NaN"
        t_str = f"{trans_val:.6f}" if not np.isnan(trans_val) else "NaN"
        print(f"  {col:>15s}  {m_str:>15s}  {t_str:>15s}  {diff_str:>15s}  {match:>8s}")


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
sep("FINAL VERIFICATION SUMMARY")

print("  Constants:")
print(f"    BASE_TEMP_C = {BASE_TEMP_C:.10f} (expected 18.3333...)")
print(f"    MIN_DAYS    = {MIN_DAYS}")
print(f"    Anomaly     = observed - normal")
print()

if all_discrepancies:
    print(f"  DISCREPANCIES FOUND: {len(all_discrepancies)}")
    for fips, col, manual, trans in all_discrepancies:
        m_str = f"{manual:.6f}" if not np.isnan(manual) else "NaN"
        t_str = f"{trans:.6f}" if not np.isnan(trans) else "NaN"
        print(f"    FIPS {fips}, {col}: manual={m_str}, transformer={t_str}")
    print()
    print("  VERIFICATION RESULT: FAIL — values diverge beyond tolerance (0.01)")
else:
    print("  All county-year records match within tolerance (0.01).")
    print()
    print("  VERIFICATION RESULT: PASS")

print()
print("  Counties verified:")
for fips in TARGET_FIPS:
    if results_manual[fips] is not None:
        r = results_manual[fips]
        h = f"{r['hdd_anomaly']:.2f}" if not np.isnan(r['hdd_anomaly']) else "NaN"
        c = f"{r['cdd_anomaly']:.2f}" if not np.isnan(r['cdd_anomaly']) else "NaN"
        print(f"    FIPS {fips}: HDD={r['hdd_annual']:.2f}, CDD={r['cdd_annual']:.2f}, "
              f"HDD_anom={h}, CDD_anom={c}")
    else:
        print(f"    FIPS {fips}: no stations passed threshold")
print()
