"""Tests for transform/event_severity_tiers.py — storm event tier classification."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from transform.event_severity_tiers import (
    METADATA_ATTRIBUTION,
    METADATA_CONFIDENCE,
    METADATA_SOURCE,
    OUTPUT_COLUMNS,
    TIER_CUTOFFS,
    TIER_WEIGHTS,
    _write_metadata,
    classify_event_severity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_events(**overrides) -> pd.DataFrame:
    """Create a single-row storm event DataFrame with sensible defaults."""
    defaults = {
        "event_id": "E001",
        "fips": "01001",
        "date": pd.Timestamp("2020-06-15"),
        "event_type": "Thunderstorm Wind",
        "property_damage": 10_000.0,
        "crop_damage": 0.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _make_multi_events(rows: list[dict]) -> pd.DataFrame:
    """Create a multi-row storm event DataFrame."""
    base = {
        "event_id": "E001",
        "fips": "01001",
        "date": pd.Timestamp("2020-06-15"),
        "event_type": "Thunderstorm Wind",
        "property_damage": 0.0,
        "crop_damage": 0.0,
    }
    records = []
    for i, row in enumerate(rows):
        record = {**base, "event_id": f"E{i+1:03d}", **row}
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------
class TestTotalDamageCalculation:
    def test_total_damage_sum(self):
        df = _make_events(property_damage=30_000.0, crop_damage=20_000.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["total_damage"] == 50_000.0

    def test_property_only(self):
        df = _make_events(property_damage=75_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["total_damage"] == 75_000.0

    def test_crop_only(self):
        df = _make_events(property_damage=0.0, crop_damage=75_000.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["total_damage"] == 75_000.0

    def test_one_field_nan_treats_as_zero(self):
        df = _make_events(property_damage=100_000.0, crop_damage=np.nan)
        result = classify_event_severity(df)
        assert result.iloc[0]["total_damage"] == 100_000.0
        assert result.iloc[0]["severity_tier"] == 2

    def test_both_nan_gives_nan_total(self):
        df = _make_events(property_damage=np.nan, crop_damage=np.nan)
        result = classify_event_severity(df)
        assert pd.isna(result.iloc[0]["total_damage"])
        assert pd.isna(result.iloc[0]["severity_tier"])
        assert pd.isna(result.iloc[0]["tier_weight"])


class TestTierClassification:
    def test_tier1(self):
        df = _make_events(property_damage=10_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 1
        assert result.iloc[0]["tier_weight"] == 1

    def test_tier2(self):
        df = _make_events(property_damage=100_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 2
        assert result.iloc[0]["tier_weight"] == 3

    def test_tier3(self):
        df = _make_events(property_damage=1_000_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 3
        assert result.iloc[0]["tier_weight"] == 7

    def test_tier4(self):
        df = _make_events(property_damage=10_000_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 4
        assert result.iloc[0]["tier_weight"] == 15

    def test_boundary_exactly_50k_is_tier2(self):
        df = _make_events(property_damage=50_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 2

    def test_boundary_exactly_500k_is_tier3(self):
        df = _make_events(property_damage=500_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 3

    def test_boundary_exactly_5m_is_tier4(self):
        df = _make_events(property_damage=5_000_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 4

    def test_boundary_just_below_50k_is_tier1(self):
        df = _make_events(property_damage=49_999.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 1

    def test_zero_damage_unreported(self):
        df = _make_events(property_damage=0.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert pd.isna(result.iloc[0]["severity_tier"])
        assert pd.isna(result.iloc[0]["tier_weight"])

    def test_multiple_events_all_tiers(self):
        rows = [
            {"property_damage": 10_000.0, "crop_damage": 0.0},       # Tier 1
            {"property_damage": 100_000.0, "crop_damage": 0.0},      # Tier 2
            {"property_damage": 1_000_000.0, "crop_damage": 0.0},    # Tier 3
            {"property_damage": 10_000_000.0, "crop_damage": 0.0},   # Tier 4
            {"property_damage": 0.0, "crop_damage": 0.0},            # Unreported
        ]
        df = _make_multi_events(rows)
        result = classify_event_severity(df)
        tiers = result["severity_tier"].tolist()
        assert tiers[0] == 1
        assert tiers[1] == 2
        assert tiers[2] == 3
        assert tiers[3] == 4
        assert pd.isna(tiers[4])


class TestEventTypePreservation:
    def test_preserves_event_type_exactly(self):
        types = [
            "Thunderstorm Wind",
            "Hurricane/Typhoon",
            "Heavy Rain",
            "Winter Storm",
            "Debris Flow",
        ]
        rows = [
            {"event_type": t, "property_damage": 10_000.0}
            for t in types
        ]
        df = _make_multi_events(rows)
        result = classify_event_severity(df)
        for i, expected_type in enumerate(types):
            assert result.iloc[i]["event_type"] == expected_type


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------
class TestOutputSchema:
    def test_column_presence(self):
        df = _make_events()
        result = classify_event_severity(df)
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_no_extra_columns(self):
        df = _make_events()
        result = classify_event_severity(df)
        assert set(result.columns) == set(OUTPUT_COLUMNS)

    def test_column_types(self):
        df = _make_events(property_damage=100_000.0)
        result = classify_event_severity(df)
        row = result.iloc[0]
        assert isinstance(row["fips"], str)
        assert isinstance(row["event_type"], str)
        assert isinstance(row["total_damage"], float)
        assert isinstance(row["severity_tier"], float)
        assert isinstance(row["tier_weight"], float)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_negative_damage_clamped_to_zero(self):
        df = _make_events(property_damage=-5_000.0, crop_damage=10_000.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["total_damage"] == 10_000.0
        assert result.iloc[0]["severity_tier"] == 1

    def test_very_large_damage_classifies_tier4(self):
        df = _make_events(property_damage=5_000_000_000.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert result.iloc[0]["severity_tier"] == 4
        assert result.iloc[0]["tier_weight"] == 15

    def test_nan_fips_excluded(self):
        df = _make_events(fips=np.nan)
        result = classify_event_severity(df)
        assert len(result) == 0

    def test_valid_fips_with_zero_damage(self):
        df = _make_events(property_damage=0.0, crop_damage=0.0)
        result = classify_event_severity(df)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["severity_tier"])

    def test_empty_input(self):
        df = pd.DataFrame(columns=[
            "event_id", "fips", "date", "event_type",
            "property_damage", "crop_damage",
        ])
        result = classify_event_severity(df)
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"event_id": ["E1"], "fips": ["01001"]})
        with pytest.raises(ValueError, match="missing columns"):
            classify_event_severity(df)

    def test_partial_data_produces_partial_output(self):
        rows = [
            {"fips": "01001", "property_damage": 100_000.0},
            {"fips": np.nan, "property_damage": 100_000.0},
        ]
        df = _make_multi_events(rows)
        # Manually set one fips to NaN (override the default)
        df.loc[1, "fips"] = np.nan
        result = classify_event_severity(df)
        assert len(result) == 1
        assert result.iloc[0]["fips"] == "01001"

    def test_fips_normalization(self):
        df = _make_events(fips="1001")  # Should become "01001"
        result = classify_event_severity(df)
        assert result.iloc[0]["fips"] == "01001"


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_reproducibility(self):
        rows = [
            {"property_damage": 10_000.0, "crop_damage": 5_000.0},
            {"property_damage": 200_000.0, "crop_damage": 100_000.0},
            {"property_damage": 0.0, "crop_damage": 0.0},
            {"property_damage": 8_000_000.0, "crop_damage": 0.0},
        ]
        df = _make_multi_events(rows)
        result1 = classify_event_severity(df.copy())
        result2 = classify_event_severity(df.copy())
        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Transform purity test
# ---------------------------------------------------------------------------
class TestTransformPurity:
    def test_no_scoring_metrics(self):
        df = _make_events(property_damage=100_000.0)
        result = classify_event_severity(df)
        forbidden = {
            "percentile", "rank", "national_rank", "composite",
            "acceleration", "overlap", "penalty", "cci_score",
        }
        for col in result.columns:
            assert col.lower() not in forbidden, f"Scoring metric found: {col}"

    def test_no_county_year_aggregation(self):
        rows = [
            {"fips": "01001", "property_damage": 10_000.0},
            {"fips": "01001", "property_damage": 50_000.0},
        ]
        df = _make_multi_events(rows)
        result = classify_event_severity(df)
        # Should have 2 rows — not aggregated to 1 county
        assert len(result) == 2

    def test_no_duplicate_events(self):
        rows = [
            {"property_damage": 10_000.0},
            {"property_damage": 50_000.0},
            {"property_damage": 500_000.0},
        ]
        df = _make_multi_events(rows)
        result = classify_event_severity(df)
        assert result["event_id"].is_unique


# ---------------------------------------------------------------------------
# Metadata test
# ---------------------------------------------------------------------------
class TestMetadata:
    def test_metadata_sidecar(self, tmp_path: Path):
        meta_path = tmp_path / "test_metadata.json"
        _write_metadata(meta_path)
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["source"] == METADATA_SOURCE
        assert meta["confidence"] == METADATA_CONFIDENCE
        assert meta["attribution"] == METADATA_ATTRIBUTION
        assert "retrieved_at" in meta


# ---------------------------------------------------------------------------
# File loading tests
# ---------------------------------------------------------------------------
class TestFileLoading:
    def test_missing_input_file_raises(self):
        from transform.event_severity_tiers import _load_storm_events

        with patch(
            "transform.event_severity_tiers.STORMS_COMBINED_PATH",
            Path("/nonexistent/storms_all.parquet"),
        ), patch(
            "transform.event_severity_tiers.STORMS_DIR",
            Path("/nonexistent"),
        ):
            with pytest.raises(FileNotFoundError):
                _load_storm_events()
