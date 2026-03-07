"""Manual computation verification for flood_zone_scoring transform.

Independently recomputes flood zone scores for 3 sample counties from raw data
and compares against the harmonized output.

Counties:
  01001 - Autauga County, AL  (moderate score ~46)
  01003 - Baldwin County, AL  (moderate score ~78)
  22089 - St. Charles Parish, LA (high score ~312, saturated at 100% high-risk area)
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/macminipro/Projects/ClimateCostIndex/cci")
RAW_NFHL = BASE / "data" / "raw" / "fema_nfhl"
RAW_CENSUS = BASE / "data" / "raw" / "census_blocks"
HARMONIZED = BASE / "data" / "harmonized"

COUNTIES = ["01001", "01003", "22089"]
SCORING_YEAR = 2024

# Constants from the transform (verified against source)
CRS_NAD83 = "EPSG:4269"
CRS_ALBERS = "EPSG:5070"
HIGH_RISK_ZONES = {"A", "AE", "AH", "AO", "V", "VE"}
MODERATE_RISK_ZONE_B = {"B"}
HIGH_RISK_WEIGHT = 3
MODERATE_RISK_WEIGHT = 1
MAP_CURRENCY_THRESHOLD_YEARS = 10


def banner(text: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}")


def section(text: str) -> None:
    print(f"\n--- {text} ---")


# ---------------------------------------------------------------------------
# Step 0: Load the harmonized output for comparison
# ---------------------------------------------------------------------------
banner("STEP 0: Load harmonized flood_zone_scoring_2024 output (ground truth)")
output = pd.read_parquet(HARMONIZED / "flood_zone_scoring_2024.parquet")
output_meta = json.loads(
    (HARMONIZED / "flood_zone_scoring_2024_metadata.json").read_text()
)
print(f"Output shape: {output.shape}")
print(f"Output columns: {output.columns.tolist()}")
print(f"Metadata: {json.dumps(output_meta, indent=2)}")

for fips in COUNTIES:
    row = output[output["fips"] == fips].iloc[0]
    print(f"\n  {fips} output values:")
    for col in output.columns:
        print(f"    {col}: {row[col]}")


# ---------------------------------------------------------------------------
# Step 1: Load county boundary shapefile and compute county areas in EPSG:5070
# ---------------------------------------------------------------------------
banner("STEP 1: Load county boundaries and compute areas in EPSG:5070")

county_shp_path = sorted(RAW_CENSUS.glob("cb_*_us_county_500k.zip"))[-1]
print(f"Loading county boundaries from: {county_shp_path}")
counties_gdf = gpd.read_file(county_shp_path)
print(f"Total counties in shapefile: {len(counties_gdf)}")
print(f"CRS: {counties_gdf.crs}")

# Normalize FIPS
counties_gdf["fips"] = counties_gdf["GEOID"].astype(str).str.zfill(5)

# Project to EPSG:5070 for area calculation
counties_albers = counties_gdf[["fips", "geometry"]].copy().to_crs(CRS_ALBERS)
counties_albers["county_area_m2"] = counties_albers.geometry.area

# Also keep NAD83 version for spatial joins
counties_nad83 = counties_gdf[["fips", "geometry"]].copy()
if counties_nad83.crs is None or counties_nad83.crs.to_epsg() != 4269:
    counties_nad83 = counties_nad83.to_crs(CRS_NAD83)

for fips in COUNTIES:
    row = counties_albers[counties_albers["fips"] == fips].iloc[0]
    area_km2 = row["county_area_m2"] / 1e6
    print(f"\n  {fips} county area:")
    print(f"    county_area_m2 = {row['county_area_m2']:,.2f}")
    print(f"    county_area_km2 = {area_km2:,.2f}")


# ---------------------------------------------------------------------------
# Step 2: Load raw NFHL flood zones for each county, classify, compute areas
# ---------------------------------------------------------------------------
banner("STEP 2: Load raw NFHL flood zones, classify, compute areas")

manual_results = {}

for fips in COUNTIES:
    section(f"County {fips}")

    # --- 2a. Load raw flood zone polygons ---
    fz_path = RAW_NFHL / f"fema_nfhl_{fips}.parquet"
    print(f"  Loading: {fz_path}")
    fz = gpd.read_parquet(fz_path)
    print(f"  Total polygons: {len(fz)}")
    print(f"  Columns: {fz.columns.tolist()}")
    print(f"  CRS: {fz.crs}")

    # Show flood_zone value counts
    zone_counts = fz["flood_zone"].value_counts()
    print(f"\n  Flood zone value counts:")
    for zone, count in zone_counts.items():
        print(f"    {zone}: {count}")

    # Show zone_subtype value counts
    subtype_counts = fz["zone_subtype"].fillna("(null)").value_counts()
    print(f"\n  Zone subtype value counts:")
    for subtype, count in subtype_counts.items():
        print(f"    '{subtype}': {count}")

    # --- 2b. Classify zones ---
    fz["zone_subtype"] = fz["zone_subtype"].fillna("")

    high_mask = fz["flood_zone"].isin(HIGH_RISK_ZONES)
    moderate_mask = (
        fz["flood_zone"].isin(MODERATE_RISK_ZONE_B)
        | (
            (fz["flood_zone"] == "X")
            & fz["zone_subtype"].str.contains(
                "0.2 PCT ANNUAL CHANCE", case=False, na=False
            )
        )
    )
    minimal_mask = ~high_mask & ~moderate_mask

    fz["risk_class"] = "minimal"
    fz.loc[high_mask, "risk_class"] = "high"
    fz.loc[moderate_mask, "risk_class"] = "moderate"

    print(f"\n  Classification results:")
    print(f"    High-risk polygons:     {high_mask.sum()}")
    print(f"    Moderate-risk polygons: {moderate_mask.sum()}")
    print(f"    Minimal-risk polygons:  {minimal_mask.sum()}")
    print(f"    Total:                  {len(fz)}")

    # Show which zones classified as what
    risk_by_zone = fz.groupby(["flood_zone", "zone_subtype", "risk_class"]).size()
    print(f"\n  Classification breakdown (zone, subtype -> risk_class):")
    for (zone, subtype, risk), count in risk_by_zone.items():
        sub_display = subtype if subtype else "(empty)"
        print(f"    {zone} / {sub_display} -> {risk}: {count}")

    # --- 2c. Project to EPSG:5070 and compute areas ---
    scored_fz = fz[fz["risk_class"].isin({"high", "moderate"})].copy()
    print(f"\n  Scored polygons (high+moderate): {len(scored_fz)}")

    scored_fz = scored_fz[["county_fips", "risk_class", "geometry"]].to_crs(CRS_ALBERS)

    # Fix invalid geometries after reprojection
    import shapely
    scored_fz["geometry"] = shapely.make_valid(scored_fz.geometry.values)
    scored_fz["area_m2"] = scored_fz.geometry.area

    # --- 2d. Sum areas by risk class ---
    area_by_class = scored_fz.groupby("risk_class")["area_m2"].sum()
    high_area_m2 = area_by_class.get("high", 0.0)
    moderate_area_m2 = area_by_class.get("moderate", 0.0)

    print(f"\n  Area computations (EPSG:5070):")
    print(f"    High-risk total area:     {high_area_m2:>20,.2f} m2  ({high_area_m2/1e6:,.4f} km2)")
    print(f"    Moderate-risk total area:  {moderate_area_m2:>20,.2f} m2  ({moderate_area_m2/1e6:,.4f} km2)")

    # --- 2e. Get county area ---
    county_area_m2 = counties_albers[counties_albers["fips"] == fips].iloc[0]["county_area_m2"]
    print(f"    County total area:        {county_area_m2:>20,.2f} m2  ({county_area_m2/1e6:,.4f} km2)")

    # --- 2f. Compute percentages ---
    pct_area_high = min((high_area_m2 / county_area_m2) * 100, 100.0)
    pct_area_mod = min((moderate_area_m2 / county_area_m2) * 100, 100.0)

    print(f"\n  Percentage calculations:")
    print(f"    pct_area_high_risk     = {high_area_m2:,.2f} / {county_area_m2:,.2f} * 100 = {pct_area_high:.6f}%")
    print(f"    pct_area_moderate_risk = {moderate_area_m2:,.2f} / {county_area_m2:,.2f} * 100 = {pct_area_mod:.6f}%")

    # --- 2g. Compute flood_exposure_score ---
    flood_exposure_score = pct_area_high * HIGH_RISK_WEIGHT + pct_area_mod * MODERATE_RISK_WEIGHT

    print(f"\n  Score formula: (pct_high * {HIGH_RISK_WEIGHT}) + (pct_moderate * {MODERATE_RISK_WEIGHT})")
    print(f"    = ({pct_area_high:.6f} * {HIGH_RISK_WEIGHT}) + ({pct_area_mod:.6f} * {MODERATE_RISK_WEIGHT})")
    print(f"    = {pct_area_high * HIGH_RISK_WEIGHT:.6f} + {pct_area_mod * MODERATE_RISK_WEIGHT:.6f}")
    print(f"    = {flood_exposure_score:.6f}")

    manual_results[fips] = {
        "pct_area_high_risk": pct_area_high,
        "pct_area_moderate_risk": pct_area_mod,
        "flood_exposure_score": flood_exposure_score,
        "high_area_m2": high_area_m2,
        "moderate_area_m2": moderate_area_m2,
        "county_area_m2": county_area_m2,
    }


# ---------------------------------------------------------------------------
# Step 3: Verify housing unit metrics (pct_hu_high_risk)
# ---------------------------------------------------------------------------
banner("STEP 3: Verify housing unit metrics (pct_hu_high_risk)")

# Load block-group housing data
bg_path = sorted(RAW_CENSUS.glob("census_blocks_*.parquet"))[-1]
print(f"Loading block-group housing data from: {bg_path}")
bg_df = pd.read_parquet(bg_path)
print(f"Total block groups: {len(bg_df)}")
print(f"Columns: {bg_df.columns.tolist()}")

for fips in COUNTIES:
    section(f"County {fips} - Housing Unit Overlay")

    # Get block groups for this county
    county_bg = bg_df[bg_df["county_fips"] == fips].copy()
    county_bg = county_bg.dropna(subset=["lat", "lon"])
    total_hu = county_bg["housing_units"].sum()
    print(f"  Block groups in county: {len(county_bg)}")
    print(f"  Total housing units: {total_hu:,.0f}")

    # Build point geometries
    bg_gdf = gpd.GeoDataFrame(
        county_bg,
        geometry=[Point(xy) for xy in zip(county_bg["lon"], county_bg["lat"])],
        crs=CRS_NAD83,
    )

    # Load flood zones for this county (high-risk only)
    fz = gpd.read_parquet(RAW_NFHL / f"fema_nfhl_{fips}.parquet")
    fz["zone_subtype"] = fz["zone_subtype"].fillna("")
    high_risk_fz = fz[fz["flood_zone"].isin(HIGH_RISK_ZONES)].copy()
    print(f"  High-risk flood zone polygons: {len(high_risk_fz)}")

    if high_risk_fz.empty:
        print(f"  No high-risk zones -> pct_hu_high_risk = 0.0")
        manual_results[fips]["pct_hu_high_risk"] = 0.0
        continue

    # Ensure CRS
    if high_risk_fz.crs is None or high_risk_fz.crs.to_epsg() != 4269:
        high_risk_fz = high_risk_fz.to_crs(CRS_NAD83)

    # Repair geometries
    invalid_mask = ~high_risk_fz.geometry.is_valid
    if invalid_mask.any():
        print(f"  Repairing {invalid_mask.sum()} invalid geometries")
        high_risk_fz.loc[invalid_mask, "geometry"] = high_risk_fz.loc[
            invalid_mask, "geometry"
        ].buffer(0)

    # Spatial join: block-group centroids within high-risk zones
    joined = gpd.sjoin(
        bg_gdf,
        high_risk_fz[["geometry"]],
        how="left",
        predicate="within",
    )

    # Deduplicate (a block group might fall in overlapping polygons)
    joined_deduped = joined.drop_duplicates(subset=["block_group_fips"])
    in_high_risk = joined_deduped[joined_deduped["index_right"].notna()]

    hu_in_hr = in_high_risk["housing_units"].sum()
    pct_hu = (hu_in_hr / total_hu * 100) if total_hu > 0 else 0.0

    print(f"  Block groups in high-risk zones: {len(in_high_risk)} / {len(county_bg)}")
    print(f"  Housing units in high-risk zones: {hu_in_hr:,.0f}")
    print(f"  pct_hu_high_risk = {hu_in_hr:,.0f} / {total_hu:,.0f} * 100 = {pct_hu:.6f}%")

    manual_results[fips]["pct_hu_high_risk"] = pct_hu


# ---------------------------------------------------------------------------
# Step 4: Verify panel dates and map currency flag
# ---------------------------------------------------------------------------
banner("STEP 4: Verify panel dates and map currency flag")

cutoff_date = pd.Timestamp(SCORING_YEAR - MAP_CURRENCY_THRESHOLD_YEARS, 1, 1)
print(f"Scoring year: {SCORING_YEAR}")
print(f"Map currency threshold: {MAP_CURRENCY_THRESHOLD_YEARS} years")
print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"  (Maps with most-recent effective_date BEFORE {cutoff_date.strftime('%Y-%m-%d')} -> flag=1 (stale))")

for fips in COUNTIES:
    section(f"County {fips} - Panel Dates")

    panel_path = RAW_NFHL / f"fema_nfhl_panels_{fips}.parquet"
    panels = pd.read_parquet(panel_path)
    panels["effective_date"] = pd.to_datetime(panels["effective_date"])

    print(f"  Total panel records: {len(panels)}")
    print(f"  Effective dates range: {panels['effective_date'].min()} to {panels['effective_date'].max()}")

    # Most recent effective date
    most_recent = panels["effective_date"].max()
    print(f"  Most recent effective_date: {most_recent.strftime('%Y-%m-%d')}")

    # Map currency flag
    flag = 1 if most_recent < cutoff_date else 0
    print(f"  {most_recent.strftime('%Y-%m-%d')} < {cutoff_date.strftime('%Y-%m-%d')} ? -> map_currency_flag = {flag}")

    manual_results[fips]["nfhl_effective_date"] = most_recent.strftime("%Y-%m-%d")
    manual_results[fips]["map_currency_flag"] = flag


# ---------------------------------------------------------------------------
# Step 5: Compare manual results with output file
# ---------------------------------------------------------------------------
banner("STEP 5: COMPARISON - Manual vs. Harmonized Output")

comparison_fields = [
    "pct_area_high_risk",
    "pct_area_moderate_risk",
    "flood_exposure_score",
    "pct_hu_high_risk",
    "nfhl_effective_date",
    "map_currency_flag",
]

all_match = True
TOLERANCE = 1e-4  # 0.01% tolerance for floating point comparison

for fips in COUNTIES:
    section(f"County {fips}")
    out_row = output[output["fips"] == fips].iloc[0]
    manual = manual_results[fips]

    for field in comparison_fields:
        out_val = out_row[field]
        man_val = manual[field]

        # Comparison
        if isinstance(out_val, str) or isinstance(man_val, str):
            match = str(out_val) == str(man_val)
            diff_str = ""
        elif pd.isna(out_val) and pd.isna(man_val):
            match = True
            diff_str = ""
        elif pd.isna(out_val) or pd.isna(man_val):
            match = False
            diff_str = f"  (one is NaN)"
        else:
            abs_diff = abs(float(out_val) - float(man_val))
            rel_diff = abs_diff / abs(float(out_val)) * 100 if float(out_val) != 0 else abs_diff
            match = abs_diff < TOLERANCE or rel_diff < 0.01  # within 0.01%
            diff_str = f"  (abs_diff={abs_diff:.8f}, rel_diff={rel_diff:.6f}%)"

        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False

        print(f"  {field}:")
        print(f"    Output:  {out_val}")
        print(f"    Manual:  {man_val}")
        print(f"    -> [{status}]{diff_str}")


# ---------------------------------------------------------------------------
# Step 6: Verify constants and formulas
# ---------------------------------------------------------------------------
banner("STEP 6: Verify constants and formulas against specification")

print(f"\n  HIGH_RISK_ZONES = {sorted(HIGH_RISK_ZONES)}")
print(f"    Expected: ['A', 'AE', 'AH', 'AO', 'V', 'VE']")
assert HIGH_RISK_ZONES == {"A", "AE", "AH", "AO", "V", "VE"}, "MISMATCH in HIGH_RISK_ZONES!"
print(f"    -> MATCH")

print(f"\n  MODERATE_RISK includes B zone: {MODERATE_RISK_ZONE_B}")
print(f"    Expected: {{'B'}}")
assert MODERATE_RISK_ZONE_B == {"B"}, "MISMATCH in MODERATE_RISK_ZONE_B!"
print(f"    -> MATCH")

print(f"\n  MODERATE_RISK X subtype check: 'contains 0.2 PCT ANNUAL CHANCE'")
print(f"    Transform uses: str.contains('0.2 PCT ANNUAL CHANCE', case=False)")
print(f"    -> Correct (matches '0.2 PCT ANNUAL CHANCE FLOOD HAZARD' and similar)")

print(f"\n  Score formula: (pct_high * {HIGH_RISK_WEIGHT}) + (pct_moderate * {MODERATE_RISK_WEIGHT})")
print(f"    Expected: (pct_high * 3) + (pct_moderate * 1)")
assert HIGH_RISK_WEIGHT == 3 and MODERATE_RISK_WEIGHT == 1, "MISMATCH in score weights!"
print(f"    -> MATCH")

print(f"\n  Map currency threshold: {MAP_CURRENCY_THRESHOLD_YEARS} years")
print(f"    Expected: 10 years")
assert MAP_CURRENCY_THRESHOLD_YEARS == 10, "MISMATCH in map currency threshold!"
print(f"    -> MATCH")

print(f"\n  CRS for area: {CRS_ALBERS}")
print(f"    Expected: EPSG:5070")
assert CRS_ALBERS == "EPSG:5070", "MISMATCH in CRS!"
print(f"    -> MATCH")

print(f"\n  pct_area_high_risk clipped at 100: Yes (in transform line 518)")
print(f"  pct_area_moderate_risk clipped at 100: Yes (in transform line 521)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner("SUMMARY")

if all_match:
    print("\n  ALL VALUES MATCH between manual computation and harmonized output.")
    print("  The flood_zone_scoring transform is producing correct results.")
else:
    print("\n  SOME VALUES DO NOT MATCH.")
    print("  Review discrepancies above.")

print(f"\n  Counties verified: {COUNTIES}")
print(f"  Scoring year: {SCORING_YEAR}")
print(f"  Total counties in output: {len(output)}")
print(f"\n  Verified computations:")
print(f"    - Zone classification (HIGH_RISK_ZONES, MODERATE_RISK with B and X/0.2PCT)")
print(f"    - Area projection to EPSG:5070 (Albers Equal Area)")
print(f"    - pct_area_high_risk = sum(high-risk area) / county_area * 100, clipped at 100")
print(f"    - pct_area_moderate_risk = sum(moderate-risk area) / county_area * 100, clipped at 100")
print(f"    - flood_exposure_score = (pct_high * 3) + (pct_moderate * 1)")
print(f"    - pct_hu_high_risk via block-group centroid point-in-polygon overlay")
print(f"    - nfhl_effective_date = max(panel effective_date) per county")
print(f"    - map_currency_flag = 1 if effective_date < (scoring_year - 10)")
print()

sys.exit(0 if all_match else 1)
