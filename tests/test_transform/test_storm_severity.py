"""Tests for transform/storm_severity.py — severity-weighted storm scores."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from transform.storm_severity import (
    FEMA_NCEI_RATIO_THRESHOLD,
    FEMA_NCEI_SENTINEL,
    METADATA_ATTRIBUTION,
    METADATA_CONFIDENCE,
    METADATA_SOURCE,
    MISSING_DAMAGE_THRESHOLD_PCT,
    OUTPUT_COLUMNS,
    _build_county_fips_crosswalk,
    _enrich_ia_with_ha,
    _normalize_county_name,
    _write_metadata,
    compute_storm_severity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_tiered_events(rows: list[dict]) -> pd.DataFrame:
    """Create a tiered events DataFrame with sensible defaults."""
    base = {
        "event_id": "E001",
        "fips": "01001",
        "date": pd.Timestamp("2020-06-15"),
        "event_type": "Thunderstorm Wind",
        "total_damage": 10_000.0,
        "severity_tier": 1.0,
        "tier_weight": 1.0,
    }
    records = []
    for i, row in enumerate(rows):
        record = {**base, "event_id": f"E{i+1:03d}", **row}
        records.append(record)
    return pd.DataFrame(records)


def _make_fema_ia(rows: list[dict]) -> pd.DataFrame:
    """Create a FEMA IA DataFrame."""
    records = []
    for row in rows:
        record = {
            "disaster_number": "DR-0001",
            "fips": "01001",
            "year": 2020,
            "ia_amount": 0.0,
            "registrant_count": 10,
            "disaster_type": "DR",
            "declaration_date": pd.Timestamp("2020-07-01"),
            **row,
        }
        records.append(record)
    return pd.DataFrame(records)


def _make_housing(rows: list[dict]) -> pd.DataFrame:
    """Create a Census ACS housing units DataFrame."""
    records = []
    for row in rows:
        record = {
            "fips": "01001",
            "year": 2020,
            "total_housing_units": 10_000,
            "owner_occupied_units": 7_000,
            "population": 25_000,
            "median_household_income": 55_000.0,
            **row,
        }
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------
class TestRawSeverityAggregation:
    def test_raw_severity_sum(self):
        """Two Tier 1 + one Tier 3 → raw = 1+1+7 = 9."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": 7.0, "severity_tier": 3.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        row = result[result["fips"] == "01001"].iloc[0]
        # raw = 9, housing = 10000, score = 9/10000 = 0.0009
        assert result.iloc[0]["storm_severity_score"] == pytest.approx(9 / 10_000)

    def test_housing_unit_normalization(self):
        """raw = 9, housing = 1000 → score = 0.009."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": 7.0, "severity_tier": 3.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020, "total_housing_units": 1_000}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["storm_severity_score"] == pytest.approx(0.009)


class TestEventCount:
    def test_event_count_includes_unreported(self):
        """5 events total (3 tiered + 2 unreported) → event_count = 5."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0, "total_damage": 10_000.0},
            {"fips": "01001", "tier_weight": 3.0, "severity_tier": 2.0, "total_damage": 100_000.0},
            {"fips": "01001", "tier_weight": 7.0, "severity_tier": 3.0, "total_damage": 1_000_000.0},
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["event_count"] == 5


class TestTotalDamage:
    def test_total_damage_sum(self):
        events = _make_tiered_events([
            {"fips": "01001", "total_damage": 10_000.0, "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "total_damage": 50_000.0, "tier_weight": 2.0, "severity_tier": 2.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["total_damage"] == 60_000.0


class TestPctMissingDamage:
    def test_pct_missing(self):
        """4 events, 1 unreported → 25%."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["pct_missing_damage"] == pytest.approx(25.0)


class TestFemaNceiRatio:
    def test_fema_ncei_ratio(self):
        """$100K NCEI, $400K FEMA → ratio 4.0."""
        events = _make_tiered_events([
            {"fips": "01001", "total_damage": 100_000.0, "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        fema = _make_fema_ia([{"fips": "01001", "year": 2020, "ia_amount": 400_000.0}])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, fema, housing)
        assert result.iloc[0]["fema_ncei_ratio"] == pytest.approx(4.0)


class TestReliabilityFlag:
    def test_high_fema_ratio_flagged(self):
        events = _make_tiered_events([
            {"fips": "01001", "total_damage": 10_000.0, "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        # FEMA ratio = 50000 / 10000 = 5.0 > 3.0
        fema = _make_fema_ia([{"fips": "01001", "year": 2020, "ia_amount": 50_000.0}])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, fema, housing)
        assert result.iloc[0]["severity_reliability_flag"] == 1

    def test_high_pct_missing_flagged(self):
        """3 out of 4 events unreported → 75% > 50%."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["severity_reliability_flag"] == 1

    def test_clean_data_not_flagged(self):
        events = _make_tiered_events([
            {"fips": "01001", "total_damage": 100_000.0, "tier_weight": 3.0, "severity_tier": 2.0},
            {"fips": "01001", "total_damage": 50_000.0, "tier_weight": 2.0, "severity_tier": 2.0},
        ])
        # FEMA ratio = 100000 / 150000 = 0.67 < 3.0, pct_missing = 0%
        fema = _make_fema_ia([{"fips": "01001", "year": 2020, "ia_amount": 100_000.0}])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, fema, housing)
        assert result.iloc[0]["severity_reliability_flag"] == 0


class TestOnlyUnreportedEvents:
    def test_all_unreported(self):
        """All events have NaN tier_weight → raw = 0, pct_missing = 100."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
            {"fips": "01001", "tier_weight": np.nan, "severity_tier": np.nan, "total_damage": 0.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["storm_severity_score"] == pytest.approx(0.0)
        assert result.iloc[0]["pct_missing_damage"] == pytest.approx(100.0)


class TestMultiCounty:
    def test_independent_scores(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0, "total_damage": 10_000.0},
            {"fips": "01003", "tier_weight": 15.0, "severity_tier": 4.0, "total_damage": 10_000_000.0},
        ])
        housing = _make_housing([
            {"fips": "01001", "year": 2020, "total_housing_units": 1_000},
            {"fips": "01003", "year": 2020, "total_housing_units": 2_000},
        ])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 2
        r1 = result[result["fips"] == "01001"].iloc[0]
        r2 = result[result["fips"] == "01003"].iloc[0]
        assert r1["storm_severity_score"] == pytest.approx(1.0 / 1_000)
        assert r2["storm_severity_score"] == pytest.approx(15.0 / 2_000)


class TestMultiYear:
    def test_separate_year_rows(self):
        events = _make_tiered_events([
            {"fips": "01001", "date": pd.Timestamp("2019-06-15"), "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "date": pd.Timestamp("2020-06-15"), "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        housing = _make_housing([
            {"fips": "01001", "year": 2019, "total_housing_units": 1_000},
            {"fips": "01001", "year": 2020, "total_housing_units": 1_000},
        ])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 2
        years = sorted(result["year"].tolist())
        assert years == [2019, 2020]


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------
class TestOutputSchema:
    def test_column_presence(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_no_extra_columns(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_column_types(self):
        events = _make_tiered_events([
            {"fips": "01001", "total_damage": 100_000.0, "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        fema = _make_fema_ia([{"fips": "01001", "year": 2020, "ia_amount": 50_000.0}])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, fema, housing)
        row = result.iloc[0]
        assert isinstance(row["fips"], str)
        assert isinstance(row["storm_severity_score"], float)
        assert isinstance(row["total_damage"], float)
        assert isinstance(row["pct_missing_damage"], float)
        assert isinstance(row["fema_ncei_ratio"], float)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_no_fema_ia_data(self):
        """No FEMA IA → fema_ncei_ratio is NaN, flag based on pct_missing only."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert pd.isna(result.iloc[0]["fema_ncei_ratio"])
        assert result.iloc[0]["severity_reliability_flag"] == 0

    def test_ncei_zero_fema_positive(self):
        """NCEI damage = 0 but FEMA > 0 → sentinel triggers flag, then replaced with NaN."""
        events = _make_tiered_events([
            {"fips": "01001", "total_damage": 0.0, "tier_weight": np.nan, "severity_tier": np.nan},
        ])
        fema = _make_fema_ia([{"fips": "01001", "year": 2020, "ia_amount": 500_000.0}])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, fema, housing)
        # Sentinel 999.0 is replaced with NaN in output to prevent downstream distortion
        assert pd.isna(result.iloc[0]["fema_ncei_ratio"])
        # But the reliability flag was set BEFORE sentinel cleanup
        assert result.iloc[0]["severity_reliability_flag"] == 1

    def test_zero_housing_units(self):
        """0 housing units → score NaN, no crash."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020, "total_housing_units": 0}])
        result = compute_storm_severity(events, None, housing)
        assert pd.isna(result.iloc[0]["storm_severity_score"])

    def test_housing_temporal_fallback(self):
        """Housing for 2022 but storm year 2023 → uses 2022 fallback."""
        events = _make_tiered_events([
            {"fips": "01001", "date": pd.Timestamp("2023-06-15"), "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2022, "total_housing_units": 5_000}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["storm_severity_score"] == pytest.approx(3.0 / 5_000)

    def test_no_fema_file_still_computes(self):
        """Module should compute scores even without FEMA IA data."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 7.0, "severity_tier": 3.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020, "total_housing_units": 1_000}])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 1
        assert result.iloc[0]["storm_severity_score"] == pytest.approx(7.0 / 1_000)
        assert pd.isna(result.iloc[0]["fema_ncei_ratio"])

    def test_county_with_no_events_absent(self):
        """Counties with zero events don't appear in output."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = _make_housing([
            {"fips": "01001", "year": 2020},
            {"fips": "01003", "year": 2020},
        ])
        result = compute_storm_severity(events, None, housing)
        assert "01003" not in result["fips"].values


# ---------------------------------------------------------------------------
# Empty / missing input tests
# ---------------------------------------------------------------------------
class TestEmptyMissing:
    def test_empty_tiered_events(self):
        events = pd.DataFrame(columns=[
            "event_id", "fips", "date", "event_type",
            "total_damage", "severity_tier", "tier_weight",
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_empty_housing_units(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = pd.DataFrame(columns=["fips", "year", "total_housing_units"])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 0

    def test_empty_fema_ia(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        fema = pd.DataFrame(columns=["fips", "year", "ia_amount"])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, fema, housing)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["fema_ncei_ratio"])

    def test_missing_tiered_columns_raises(self):
        events = pd.DataFrame({"event_id": ["E1"], "fips": ["01001"]})
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        with pytest.raises(ValueError, match="missing columns"):
            compute_storm_severity(events, None, housing)

    def test_missing_housing_columns_raises(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = pd.DataFrame({"fips": ["01001"]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_storm_severity(events, None, housing)

    def test_partial_data_produces_partial_output(self):
        """Some counties have housing data, others don't."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01003", "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        # Only 01001 has housing data
        housing = _make_housing([{"fips": "01001", "year": 2020, "total_housing_units": 1_000}])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 2
        r1 = result[result["fips"] == "01001"].iloc[0]
        r2 = result[result["fips"] == "01003"].iloc[0]
        assert r1["storm_severity_score"] == pytest.approx(1.0 / 1_000)
        assert pd.isna(r2["storm_severity_score"])


class TestFipsNormalization:
    def test_fips_zero_padded(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert result.iloc[0]["fips"] == "01001"


class TestFileLoading:
    def test_missing_tiered_events_file_raises(self):
        from transform.storm_severity import _load_tiered_events

        with patch(
            "transform.storm_severity.TIERED_EVENTS_PATH",
            Path("/nonexistent/event_severity_tiers.parquet"),
        ):
            with pytest.raises(FileNotFoundError):
                _load_tiered_events()

    def test_missing_housing_file_raises(self):
        from transform.storm_severity import _load_housing_units

        with patch(
            "transform.storm_severity.CENSUS_ACS_COMBINED_PATH",
            Path("/nonexistent/census_acs_all.parquet"),
        ), patch(
            "transform.storm_severity.CENSUS_ACS_DIR",
            Path("/nonexistent"),
        ):
            with pytest.raises(FileNotFoundError):
                _load_housing_units()

    def test_missing_fema_ia_returns_none(self):
        from transform.storm_severity import _load_fema_ia

        with patch(
            "transform.storm_severity.FEMA_IA_COMBINED_PATH",
            Path("/nonexistent/fema_ia_all.parquet"),
        ), patch(
            "transform.storm_severity.FEMA_IA_DIR",
            Path("/nonexistent"),
        ):
            result = _load_fema_ia()
            assert result is None


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_reproducibility(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0, "total_damage": 10_000.0},
            {"fips": "01001", "tier_weight": 7.0, "severity_tier": 3.0, "total_damage": 1_000_000.0},
            {"fips": "01003", "tier_weight": 15.0, "severity_tier": 4.0, "total_damage": 10_000_000.0},
        ])
        fema = _make_fema_ia([
            {"fips": "01001", "year": 2020, "ia_amount": 200_000.0},
        ])
        housing = _make_housing([
            {"fips": "01001", "year": 2020, "total_housing_units": 5_000},
            {"fips": "01003", "year": 2020, "total_housing_units": 3_000},
        ])
        r1 = compute_storm_severity(events.copy(), fema.copy(), housing.copy())
        r2 = compute_storm_severity(events.copy(), fema.copy(), housing.copy())
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Transform purity test
# ---------------------------------------------------------------------------
class TestTransformPurity:
    def test_no_scoring_metrics(self):
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        forbidden = {
            "percentile", "rank", "national_rank", "composite",
            "acceleration", "overlap", "penalty", "cci_score",
        }
        for col in result.columns:
            assert col.lower() not in forbidden, f"Scoring metric found: {col}"

    def test_county_year_grain(self):
        """Output has exactly one row per (fips, year)."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
            {"fips": "01001", "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 1  # Two events aggregated to one county-year


# ---------------------------------------------------------------------------
# Metadata test
# ---------------------------------------------------------------------------
class TestMetadata:
    def test_metadata_sidecar(self, tmp_path: Path):
        meta_path = tmp_path / "test_metadata.json"
        _write_metadata(meta_path, 2020)
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == METADATA_SOURCE
        assert meta["confidence"] == METADATA_CONFIDENCE
        assert meta["attribution"] == METADATA_ATTRIBUTION
        assert meta["data_vintage"] == "2020"
        assert "retrieved_at" in meta


# ---------------------------------------------------------------------------
# FEMA HA integration helpers
# ---------------------------------------------------------------------------
def _make_fema_ha(rows: list[dict]) -> pd.DataFrame:
    """Create a FEMA HA DataFrame."""
    records = []
    for row in rows:
        record = {
            "disaster_number": "4337",
            "state": "AL",
            "county": "Baldwin (County)",
            "ia_amount": 100_000.0,
            "registrant_count": 50.0,
            **row,
        }
        records.append(record)
    return pd.DataFrame(records)


def _make_crosswalk_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Create a mock county_boundaries parquet for crosswalk building."""
    records = []
    for row in rows:
        record = {
            "county_fips": "01003",
            "state_fips": "01",
            "county_name": "Baldwin",
            "lat": 30.66,
            "lon": -87.75,
            "land_area_sqm": 1e9,
            "water_area_sqm": 1e8,
            **row,
        }
        records.append(record)
    df = pd.DataFrame(records)
    path = tmp_path / "county_boundaries_2024.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# County name normalization tests
# ---------------------------------------------------------------------------
class TestCountyNameNormalization:
    def test_strip_county_suffix(self):
        assert _normalize_county_name("Baldwin (County)") == "baldwin"

    def test_strip_parish_suffix(self):
        assert _normalize_county_name("Jefferson (Parish)") == "jefferson"

    def test_strip_borough_suffix(self):
        assert _normalize_county_name("Denali (Borough)") == "denali"

    def test_strip_municipio_suffix(self):
        assert _normalize_county_name("Bayamón (Municipio)") == "bayamon"

    def test_strip_municipality_suffix(self):
        assert _normalize_county_name("Anchorage (Municipality)") == "anchorage"

    def test_saint_normalization(self):
        assert _normalize_county_name("St. Lucie (County)") == "saint lucie"

    def test_multiword_name(self):
        assert _normalize_county_name("Fort Bend (County)") == "fort bend"

    def test_hyphenated_name(self):
        assert _normalize_county_name("Miami-Dade (County)") == "miami-dade"

    def test_no_suffix(self):
        assert _normalize_county_name("Baldwin") == "baldwin"

    def test_whitespace_handling(self):
        assert _normalize_county_name("  Baldwin (County)  ") == "baldwin"


# ---------------------------------------------------------------------------
# Crosswalk building tests
# ---------------------------------------------------------------------------
class TestBuildCrosswalk:
    def test_crosswalk_from_boundaries(self, tmp_path: Path):
        path = _make_crosswalk_parquet(tmp_path, [
            {"county_fips": "01003", "state_fips": "01", "county_name": "Baldwin"},
            {"county_fips": "01001", "state_fips": "01", "county_name": "Autauga"},
        ])
        with patch("transform.storm_severity.COUNTY_BOUNDARIES_PATH", path):
            cw = _build_county_fips_crosswalk()
        assert len(cw) == 2
        row = cw[cw["fips"] == "01003"].iloc[0]
        assert row["state_abbr"] == "AL"
        assert row["county_norm"] == "baldwin"

    def test_crosswalk_missing_file(self, tmp_path: Path):
        with patch(
            "transform.storm_severity.COUNTY_BOUNDARIES_PATH",
            tmp_path / "nonexistent.parquet",
        ):
            cw = _build_county_fips_crosswalk()
        assert len(cw) == 0


# ---------------------------------------------------------------------------
# FEMA HA merge / enrichment tests
# ---------------------------------------------------------------------------
class TestFemaHaEnrichment:
    def test_ha_enriches_ia_amount(self, tmp_path: Path):
        """HA dollar amounts replace NaN ia_amount in IA records."""
        ia = _make_fema_ia([
            {"disaster_number": "4337", "fips": "01003", "year": 2020, "ia_amount": np.nan},
        ])
        ha = _make_fema_ha([
            {"disaster_number": "4337", "state": "AL", "county": "Baldwin (County)",
             "ia_amount": 250_000.0, "registrant_count": 100.0},
        ])
        cb_path = _make_crosswalk_parquet(tmp_path, [
            {"county_fips": "01003", "state_fips": "01", "county_name": "Baldwin"},
        ])
        with patch("transform.storm_severity.COUNTY_BOUNDARIES_PATH", cb_path):
            enriched = _enrich_ia_with_ha(ia, ha)

        row = enriched[enriched["fips"] == "01003"].iloc[0]
        assert row["ia_amount"] == pytest.approx(250_000.0)
        assert row["registrant_count"] == pytest.approx(100.0)

    def test_county_name_to_fips_resolution(self, tmp_path: Path):
        """HA county name 'Baldwin (County)' maps to FIPS 01003 via crosswalk."""
        ia = _make_fema_ia([
            {"disaster_number": "4337", "fips": "01003", "year": 2020, "ia_amount": np.nan},
        ])
        ha = _make_fema_ha([
            {"disaster_number": "4337", "state": "AL", "county": "Baldwin (County)",
             "ia_amount": 500_000.0, "registrant_count": 200.0},
        ])
        cb_path = _make_crosswalk_parquet(tmp_path, [
            {"county_fips": "01003", "state_fips": "01", "county_name": "Baldwin"},
        ])
        with patch("transform.storm_severity.COUNTY_BOUNDARIES_PATH", cb_path):
            enriched = _enrich_ia_with_ha(ia, ha)

        assert enriched.iloc[0]["ia_amount"] == pytest.approx(500_000.0)

    def test_unmatched_ha_records_graceful(self, tmp_path: Path):
        """HA records with unknown county names don't crash; IA stays NaN."""
        ia = _make_fema_ia([
            {"disaster_number": "4337", "fips": "01003", "year": 2020, "ia_amount": np.nan},
        ])
        ha = _make_fema_ha([
            {"disaster_number": "4337", "state": "AL", "county": "Nonexistent (County)",
             "ia_amount": 100_000.0},
        ])
        cb_path = _make_crosswalk_parquet(tmp_path, [
            {"county_fips": "01003", "state_fips": "01", "county_name": "Baldwin"},
        ])
        with patch("transform.storm_severity.COUNTY_BOUNDARIES_PATH", cb_path):
            enriched = _enrich_ia_with_ha(ia, ha)

        # ia_amount should remain NaN since the HA record didn't match
        assert pd.isna(enriched.iloc[0]["ia_amount"])

    def test_ha_data_missing_entirely(self):
        """Module works when fema_ha_all.parquet doesn't exist."""
        events = _make_tiered_events([
            {"fips": "01001", "tier_weight": 1.0, "severity_tier": 1.0},
        ])
        housing = _make_housing([{"fips": "01001", "year": 2020}])
        result = compute_storm_severity(events, None, housing)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["fema_ncei_ratio"])

    def test_disaster_year_resolution(self, tmp_path: Path):
        """HA records get year from IA declarations via disaster_number."""
        ia = _make_fema_ia([
            {"disaster_number": "4337", "fips": "01003", "year": 2019, "ia_amount": np.nan},
            {"disaster_number": "4500", "fips": "01003", "year": 2021, "ia_amount": np.nan},
        ])
        ha = _make_fema_ha([
            {"disaster_number": "4337", "state": "AL", "county": "Baldwin (County)",
             "ia_amount": 100_000.0, "registrant_count": 50.0},
            {"disaster_number": "4500", "state": "AL", "county": "Baldwin (County)",
             "ia_amount": 200_000.0, "registrant_count": 80.0},
        ])
        cb_path = _make_crosswalk_parquet(tmp_path, [
            {"county_fips": "01003", "state_fips": "01", "county_name": "Baldwin"},
        ])
        with patch("transform.storm_severity.COUNTY_BOUNDARIES_PATH", cb_path):
            enriched = _enrich_ia_with_ha(ia, ha)

        # Both years should be enriched
        row_2019 = enriched[enriched["year"] == 2019].iloc[0]
        row_2021 = enriched[enriched["year"] == 2021].iloc[0]
        # 2019: $100K from disaster 4337
        assert row_2019["ia_amount"] == pytest.approx(100_000.0)
        # 2021: $200K from disaster 4500
        assert row_2021["ia_amount"] == pytest.approx(200_000.0)

    def test_multiple_counties_per_disaster(self, tmp_path: Path):
        """Multiple HA counties under one disaster each get own FIPS."""
        ia = _make_fema_ia([
            {"disaster_number": "4337", "fips": "01003", "year": 2020, "ia_amount": np.nan},
            {"disaster_number": "4337", "fips": "01001", "year": 2020, "ia_amount": np.nan},
            {"disaster_number": "4337", "fips": "12001", "year": 2020, "ia_amount": np.nan},
        ])
        ha = _make_fema_ha([
            {"disaster_number": "4337", "state": "AL", "county": "Baldwin (County)",
             "ia_amount": 100_000.0, "registrant_count": 50.0},
            {"disaster_number": "4337", "state": "AL", "county": "Autauga (County)",
             "ia_amount": 75_000.0, "registrant_count": 30.0},
            {"disaster_number": "4337", "state": "FL", "county": "Alachua (County)",
             "ia_amount": 200_000.0, "registrant_count": 100.0},
        ])
        cb_path = _make_crosswalk_parquet(tmp_path, [
            {"county_fips": "01003", "state_fips": "01", "county_name": "Baldwin"},
            {"county_fips": "01001", "state_fips": "01", "county_name": "Autauga"},
            {"county_fips": "12001", "state_fips": "12", "county_name": "Alachua"},
        ])
        with patch("transform.storm_severity.COUNTY_BOUNDARIES_PATH", cb_path):
            enriched = _enrich_ia_with_ha(ia, ha)

        r_01003 = enriched[enriched["fips"] == "01003"].iloc[0]
        r_01001 = enriched[enriched["fips"] == "01001"].iloc[0]
        r_12001 = enriched[enriched["fips"] == "12001"].iloc[0]
        assert r_01003["ia_amount"] == pytest.approx(100_000.0)
        assert r_01001["ia_amount"] == pytest.approx(75_000.0)
        assert r_12001["ia_amount"] == pytest.approx(200_000.0)

    def test_ha_enriched_fema_ratio_in_pipeline(self, tmp_path: Path):
        """End-to-end: HA amounts produce a real FEMA/NCEI ratio."""
        events = _make_tiered_events([
            {"fips": "01003", "total_damage": 100_000.0,
             "tier_weight": 3.0, "severity_tier": 2.0},
        ])
        # Pre-merged FEMA IA with HA amounts (simulating what _load_fema_ia returns)
        fema = _make_fema_ia([
            {"fips": "01003", "year": 2020, "ia_amount": 400_000.0},
        ])
        housing = _make_housing([{"fips": "01003", "year": 2020}])
        result = compute_storm_severity(events, fema, housing)
        # ratio = 400K / 100K = 4.0
        assert result.iloc[0]["fema_ncei_ratio"] == pytest.approx(4.0)

    def test_ha_empty_crosswalk_no_crash(self, tmp_path: Path):
        """If crosswalk is empty, HA merge is a no-op and IA stays intact."""
        ia = _make_fema_ia([
            {"disaster_number": "4337", "fips": "01003", "year": 2020, "ia_amount": np.nan},
        ])
        ha = _make_fema_ha([
            {"disaster_number": "4337", "state": "AL", "county": "Baldwin (County)",
             "ia_amount": 100_000.0},
        ])
        with patch(
            "transform.storm_severity.COUNTY_BOUNDARIES_PATH",
            tmp_path / "nonexistent.parquet",
        ):
            enriched = _enrich_ia_with_ha(ia, ha)

        # Should return IA unchanged
        assert pd.isna(enriched.iloc[0]["ia_amount"])
