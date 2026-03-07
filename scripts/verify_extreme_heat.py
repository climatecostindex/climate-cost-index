"""Manual computation verification for the extreme_heat transformer.

This script independently recomputes extreme heat day counts for selected
counties and compares them against the transformer's output files.
"""

import sys
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)

SEPARATOR = "=" * 80

# ---------------------------------------------------------------------------
# 0. Verify threshold constants
# ---------------------------------------------------------------------------
print(SEPARATOR)
print("STEP 0: Verify threshold constants")
print(SEPARATOR)

THRESHOLD_95F_C = (95 - 32) * 5 / 9
THRESHOLD_100F_C = (100 - 32) * 5 / 9

print(f"  THRESHOLD_95F_C  = (95 - 32) * 5/9 = {THRESHOLD_95F_C}")
print(f"  Expected: 35.0")
print(f"  Match: {THRESHOLD_95F_C == 35.0}")
print()
print(f"  THRESHOLD_100F_C = (100 - 32) * 5/9 = {THRESHOLD_100F_C}")
print(f"  Full precision: {THRESHOLD_100F_C!r}")
print(f"  Expected: 37.7777... recurring")
print(f"  Is 37.0 + 7/9: {abs(THRESHOLD_100F_C - (37 + 7/9)) < 1e-15}")
print()

# Verify strict inequality semantics
test_val_exactly_35 = 35.0
test_val_slightly_above = 35.0 + 1e-10
print(f"  Strict inequality check (tmax > threshold, NOT >=):")
print(f"    35.0 > 35.0 = {test_val_exactly_35 > THRESHOLD_95F_C}  (should be False)")
print(f"    35.0+eps > 35.0 = {test_val_slightly_above > THRESHOLD_95F_C}  (should be True)")
print()

MIN_DAYS_PER_YEAR = 335
print(f"  MIN_DAYS_PER_YEAR = {MIN_DAYS_PER_YEAR}")
print()

# ---------------------------------------------------------------------------
# 1. Load raw daily observations
# ---------------------------------------------------------------------------
print(SEPARATOR)
print("STEP 1: Load raw daily observations for 2023")
print(SEPARATOR)

RAW_PATH = "/Users/macminipro/Projects/ClimateCostIndex/cci/data/raw/noaa_ncei/noaa_ncei_2023.parquet"
print(f"  Loading: {RAW_PATH}")
obs_2023 = pd.read_parquet(RAW_PATH)
print(f"  Shape: {obs_2023.shape}")
print(f"  Columns: {list(obs_2023.columns)}")
print(f"  dtypes:")
for col in obs_2023.columns:
    print(f"    {col}: {obs_2023[col].dtype}")
print(f"  Sample rows:")
print(obs_2023.head(3).to_string(index=False))
print()

# Check for q_flag_tmax
if "q_flag_tmax" in obs_2023.columns:
    print(f"  q_flag_tmax unique values (first 20): {obs_2023['q_flag_tmax'].unique()[:20]}")
    print(f"  q_flag_tmax non-null non-empty count: {(obs_2023['q_flag_tmax'].fillna('') != '').sum()}")
else:
    print("  WARNING: q_flag_tmax column not found!")
print()

# ---------------------------------------------------------------------------
# 2. Load station-to-county mapping
# ---------------------------------------------------------------------------
print(SEPARATOR)
print("STEP 2: Load station-to-county mapping")
print(SEPARATOR)

STC_PATH = "/Users/macminipro/Projects/ClimateCostIndex/cci/data/harmonized/station_to_county.parquet"
print(f"  Loading: {STC_PATH}")
stc = pd.read_parquet(STC_PATH)
print(f"  Shape: {stc.shape}")
print(f"  Columns: {list(stc.columns)}")
print(f"  Unique stations: {stc['station_id'].nunique()}")
print(f"  Unique counties (fips): {stc['fips'].nunique()}")
print(f"  Sample rows:")
print(stc.head(5).to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 3. Load transformer output for 2023
# ---------------------------------------------------------------------------
print(SEPARATOR)
print("STEP 3: Load extreme_heat_2023 output")
print(SEPARATOR)

OUT_PATH = "/Users/macminipro/Projects/ClimateCostIndex/cci/data/harmonized/extreme_heat_2023.parquet"
print(f"  Loading: {OUT_PATH}")
eh_2023 = pd.read_parquet(OUT_PATH)
print(f"  Shape: {eh_2023.shape}")
print(f"  Columns: {list(eh_2023.columns)}")
print(f"  dtypes:")
for col in eh_2023.columns:
    print(f"    {col}: {eh_2023[col].dtype}")
print()
print(f"  Summary statistics:")
print(eh_2023[["days_above_95f", "days_above_100f"]].describe().to_string())
print()

# Verify days_above_100f <= days_above_95f for ALL records
violation_mask = eh_2023["days_above_100f"] > eh_2023["days_above_95f"]
n_violations = violation_mask.sum()
print(f"  GLOBAL CHECK: days_above_100f <= days_above_95f for all records?")
print(f"    Violations: {n_violations} out of {len(eh_2023)} records")
if n_violations > 0:
    print("    VIOLATION DETAILS:")
    print(eh_2023[violation_mask].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 4. Select counties for verification
# ---------------------------------------------------------------------------
print(SEPARATOR)
print("STEP 4: Select counties for verification")
print(SEPARATOR)

# Find counties with some extreme heat and 2-5 stations
county_station_counts = stc.groupby("fips")["station_id"].nunique().reset_index()
county_station_counts.columns = ["fips", "n_stations"]
county_station_counts = county_station_counts[
    (county_station_counts["n_stations"] >= 2) &
    (county_station_counts["n_stations"] <= 5)
]

# Merge with extreme heat output
merged_info = county_station_counts.merge(eh_2023, on="fips", how="inner")
print(f"  Counties with 2-5 stations and extreme heat output: {len(merged_info)}")

# Pick one hot county (high days_above_95f)
hot_counties = merged_info.sort_values("days_above_95f", ascending=False)
print(f"\n  Top 10 hottest counties (by days_above_95f):")
print(hot_counties.head(10).to_string(index=False))

# Pick one zero-heat county
zero_counties = merged_info[merged_info["days_above_95f"] == 0.0]
print(f"\n  Counties with 0 extreme heat days: {len(zero_counties)}")
if len(zero_counties) > 0:
    print(zero_counties.head(5).to_string(index=False))

# Select verification targets
# Target 1: Hottest county with 2-5 stations
target1_fips = hot_counties.iloc[0]["fips"]
# Target 2: A county with moderate heat
moderate = merged_info[(merged_info["days_above_95f"] > 5) & (merged_info["days_above_95f"] < 50)]
if len(moderate) > 0:
    target2_fips = moderate.iloc[len(moderate) // 2]["fips"]
else:
    target2_fips = hot_counties.iloc[len(hot_counties) // 2]["fips"]
# Target 3: A county with 0 extreme heat days
if len(zero_counties) > 0:
    target3_fips = zero_counties.iloc[0]["fips"]
else:
    # Fallback: county with lowest non-zero days
    target3_fips = merged_info.sort_values("days_above_95f").iloc[0]["fips"]

targets = [target1_fips, target2_fips, target3_fips]
print(f"\n  Selected target counties: {targets}")
print()

# ---------------------------------------------------------------------------
# 5. Manual verification for each target county
# ---------------------------------------------------------------------------
print(SEPARATOR)
print("STEP 5: Manual verification for each target county")
print(SEPARATOR)

# Pre-filter observations: drop NaN tmax, drop quality-flagged rows
print("\n  Pre-filtering observations (same as transformer)...")
obs = obs_2023.copy()
initial_count = len(obs)

# Drop NaN tmax
nan_mask = obs["tmax"].isna()
nan_count = nan_mask.sum()
print(f"    NaN tmax rows: {nan_count}")
obs = obs[~nan_mask]

# Drop quality-flagged rows
q_tmax = obs["q_flag_tmax"].fillna("")
flag_mask = q_tmax != ""
flag_count = flag_mask.sum()
print(f"    Quality-flagged rows: {flag_count}")
obs = obs[~flag_mask]

total_dropped = initial_count - len(obs)
print(f"    Total: {initial_count} -> {len(obs)} rows ({total_dropped} dropped)")
print()

# Extract year
obs = obs.copy()
obs["year"] = pd.to_datetime(obs["date"]).dt.year
# Keep only 2023
obs = obs[obs["year"] == 2023].copy()
print(f"  Observations for year 2023 after filtering: {len(obs)}")
print()

all_pass = True

for idx, fips in enumerate(targets, 1):
    print("-" * 80)
    print(f"  COUNTY {idx}: FIPS = {fips}")
    print("-" * 80)

    # 5a. Identify stations for this county
    county_stations = stc[stc["fips"] == fips]["station_id"].unique()
    print(f"  5a. Stations mapped to county: {len(county_stations)}")
    for sid in county_stations:
        print(f"      {sid}")
    print()

    # 5b. Get daily observations for each station in 2023
    station_obs = obs[obs["station_id"].isin(county_stations)].copy()
    print(f"  5b. Total daily observations across all stations: {len(station_obs)}")
    print()

    # Per-station breakdown
    station_results = []
    for sid in county_stations:
        s_obs = station_obs[station_obs["station_id"] == sid].copy()
        n_valid_days = len(s_obs)

        # 5c. These are already filtered (NaN and q_flag removed above)
        # 5d. Count threshold exceedances
        days_95 = (s_obs["tmax"] > THRESHOLD_95F_C).sum()
        days_100 = (s_obs["tmax"] > THRESHOLD_100F_C).sum()

        # 5e. Check completeness
        passes_completeness = n_valid_days >= MIN_DAYS_PER_YEAR

        print(f"  Station {sid}:")
        print(f"    Valid days: {n_valid_days} (threshold: {MIN_DAYS_PER_YEAR}, passes: {passes_completeness})")
        print(f"    Days above 95F (>35.0C): {days_95}")
        print(f"    Days above 100F (>37.778C): {days_100}")
        print(f"    Check days_100 <= days_95: {days_100 <= days_95}")

        if passes_completeness:
            station_results.append({
                "station_id": sid,
                "days_above_95f": days_95,
                "days_above_100f": days_100,
                "n_valid_days": n_valid_days,
            })

            # Show some tmax values near threshold for 95F
            near_95 = s_obs[(s_obs["tmax"] >= 34.5) & (s_obs["tmax"] <= 35.5)]
            if len(near_95) > 0:
                print(f"    Sample observations near 95F threshold (34.5-35.5C), first 5:")
                for _, row in near_95.head(5).iterrows():
                    exceeds = row["tmax"] > THRESHOLD_95F_C
                    print(f"      date={row['date']}, tmax={row['tmax']:.4f}C, exceeds_95F={exceeds}")

            # Show some tmax values near threshold for 100F
            near_100 = s_obs[(s_obs["tmax"] >= 37.2) & (s_obs["tmax"] <= 38.3)]
            if len(near_100) > 0:
                print(f"    Sample observations near 100F threshold (37.2-38.3C), first 5:")
                for _, row in near_100.head(5).iterrows():
                    exceeds = row["tmax"] > THRESHOLD_100F_C
                    print(f"      date={row['date']}, tmax={row['tmax']:.4f}C, exceeds_100F={exceeds}")
        else:
            print(f"    EXCLUDED (below completeness threshold)")
        print()

    # 5f. Average across stations to get county values
    if len(station_results) == 0:
        print(f"  5f. No stations passed completeness threshold for county {fips}")
        manual_95 = None
        manual_100 = None
    else:
        sdf = pd.DataFrame(station_results)
        manual_95 = sdf["days_above_95f"].mean()
        manual_100 = sdf["days_above_100f"].mean()
        print(f"  5f. County-level average (across {len(station_results)} stations):")
        print(f"    Per-station days_above_95f: {sdf['days_above_95f'].tolist()}")
        print(f"    Per-station days_above_100f: {sdf['days_above_100f'].tolist()}")
        print(f"    Mean days_above_95f:  {manual_95}")
        print(f"    Mean days_above_100f: {manual_100}")
    print()

    # 5g. Compare to output file
    output_row = eh_2023[eh_2023["fips"] == fips]
    if len(output_row) == 0:
        print(f"  5g. WARNING: FIPS {fips} not found in extreme_heat_2023 output!")
        if manual_95 is None:
            print(f"       This is expected (no stations passed completeness).")
        else:
            print(f"       MISMATCH: Manual computation found values but output has no row!")
            all_pass = False
    else:
        output_95 = output_row.iloc[0]["days_above_95f"]
        output_100 = output_row.iloc[0]["days_above_100f"]
        print(f"  5g. Comparison:")
        print(f"    Output  days_above_95f:  {output_95}")
        print(f"    Manual  days_above_95f:  {manual_95}")
        if manual_95 is not None:
            diff_95 = abs(output_95 - manual_95)
            match_95 = diff_95 < 1e-10
            print(f"    Difference: {diff_95}")
            print(f"    Match: {match_95}")
        else:
            match_95 = False
            print(f"    CANNOT COMPARE (no manual result)")

        print(f"    Output  days_above_100f: {output_100}")
        print(f"    Manual  days_above_100f: {manual_100}")
        if manual_100 is not None:
            diff_100 = abs(output_100 - manual_100)
            match_100 = diff_100 < 1e-10
            print(f"    Difference: {diff_100}")
            print(f"    Match: {match_100}")
        else:
            match_100 = False
            print(f"    CANNOT COMPARE (no manual result)")

        # Check days_above_100f <= days_above_95f in output
        print(f"    Output: days_above_100f ({output_100}) <= days_above_95f ({output_95}): {output_100 <= output_95}")

        if manual_95 is not None and manual_100 is not None:
            if not (match_95 and match_100):
                all_pass = False
                print(f"    >>> VERIFICATION FAILED <<<")
            else:
                print(f"    >>> VERIFICATION PASSED <<<")
    print()

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print(SEPARATOR)
print("STEP 6: Summary")
print(SEPARATOR)

print(f"\n  Threshold verification:")
print(f"    THRESHOLD_95F_C  = {THRESHOLD_95F_C} (expected 35.0): {'PASS' if THRESHOLD_95F_C == 35.0 else 'FAIL'}")
print(f"    THRESHOLD_100F_C = {THRESHOLD_100F_C!r} (expected 37.777...): {'PASS' if abs(THRESHOLD_100F_C - 37.77777777777778) < 1e-14 else 'FAIL'}")
print(f"    Strict inequality (> not >=): PASS (verified in Step 0)")
print(f"    days_above_100f <= days_above_95f (global): {'PASS' if n_violations == 0 else 'FAIL'}")
print()
print(f"  County verification: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
print()

if not all_pass:
    sys.exit(1)
else:
    print("  All manual computations match the transformer output exactly.")
    sys.exit(0)
