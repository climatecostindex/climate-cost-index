"""Smoke test that verifies the project scaffold imports correctly."""


def test_config_imports():
    from config.settings import Settings, get_settings
    from config.components import COMPONENTS, ComponentDef, get_weights
    from config.confidence_grades import ConfidenceGrade, CONFIDENCE_GRADES
    from config.fips_codes import is_valid_fips, state_abbr, STATE_FIPS

    assert len(COMPONENTS) > 0
    assert len(CONFIDENCE_GRADES) == 3
    assert len(STATE_FIPS) == 51  # 50 states + DC


def test_settings_defaults():
    from config.settings import get_settings

    s = get_settings()
    assert s.methodology_version == "1.0"
    assert s.climate_normal_baseline == "1991-2020"
    assert s.winsorize_percentile == 99.0
    assert s.overlap_penalty_floor == 0.2


def test_component_weights_sum_to_one():
    from config.components import get_weights

    weights = get_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_component_definitions_complete():
    from config.components import COMPONENTS, Attribution

    for comp_id, comp in COMPONENTS.items():
        assert comp.id == comp_id
        assert comp.attribution in (Attribution.ATTRIBUTED, Attribution.PROXY)
        assert comp.confidence in ("A", "B", "C")
        assert comp.base_weight > 0
        assert comp.acceleration_window in (5, 10)


def test_fips_validation():
    from config.fips_codes import is_valid_fips

    assert is_valid_fips("12086")   # Miami-Dade, FL
    assert is_valid_fips("06037")   # Los Angeles, CA
    assert not is_valid_fips("99999")  # invalid state
    assert not is_valid_fips("1234")   # too short
    assert not is_valid_fips("abcde")  # not digits


def test_base_ingester_schema_validation():
    """Per-subclass validate_output accepts a DataFrame matching required_columns."""
    import pandas as pd
    from ingest.usdm_drought import USDMDroughtIngester
    from datetime import date

    ing = USDMDroughtIngester()
    df = pd.DataFrame({
        "fips": ["12086"],
        "date": [date(2024, 1, 2)],
        "d0_pct": [20.0],
        "d1_pct": [15.0],
        "d2_pct": [10.0],
        "d3_pct": [5.0],
        "d4_pct": [0.0],
        "none_pct": [50.0],
    })
    ing.validate_output(df)  # should not raise


def test_base_ingester_rejects_bad_schema():
    """Per-subclass validate_output rejects a DataFrame with missing columns."""
    import pandas as pd
    import pytest
    from ingest.usdm_drought import USDMDroughtIngester

    ing = USDMDroughtIngester()
    df = pd.DataFrame({"fips": ["12086"]})  # missing most columns
    with pytest.raises(ValueError, match="missing required columns"):
        ing.validate_output(df)
