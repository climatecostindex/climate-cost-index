"""Shared fixtures for sensitivity tests."""

import numpy as np
import pandas as pd
import pytest

from config.components import COMPONENTS


@pytest.fixture
def component_ids():
    """All 12 component IDs."""
    return list(COMPONENTS.keys())


@pytest.fixture
def synthetic_harmonized(component_ids):
    """Create a synthetic harmonized DataFrame suitable for compute_cci.

    20 counties, 6 years (2019-2024), all 12 components with varied values.
    Required core (hdd/cdd_anomaly, drought_score, storm_severity) always present.
    """
    rng = np.random.default_rng(123)
    n_counties = 20
    years = list(range(2019, 2025))
    fips_codes = [f"{i:05d}" for i in range(1, n_counties + 1)]

    rows = []
    for year in years:
        for fips in fips_codes:
            row = {"fips": fips, "year": year}
            for comp in component_ids:
                # Use positive values to avoid issues with log/sqrt transforms
                row[comp] = rng.uniform(1, 100)
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def base_weights():
    """Primary weight vector from config."""
    from config.components import get_weights
    return get_weights()
