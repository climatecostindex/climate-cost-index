"""County FIPS reference utilities.

Full FIPS-to-name mapping is loaded lazily from Census data or a bundled
CSV. This module provides lookup helpers and validation.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config.settings import RAW_DIR

_FIPS_PATTERN = re.compile(r"^\d{5}$")

# State FIPS codes (2-digit) for the 50 states + DC
STATE_FIPS: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

FIPS_TO_STATE = {v: k for k, v in STATE_FIPS.items()}


def is_valid_fips(fips: str) -> bool:
    """Check if a string is a valid 5-digit county FIPS code."""
    if not _FIPS_PATTERN.match(fips):
        return False
    return fips[:2] in STATE_FIPS


def state_fips(fips: str) -> str:
    """Extract the 2-digit state FIPS from a 5-digit county FIPS."""
    return fips[:2]


def state_abbr(fips: str) -> str:
    """Return 2-letter state abbreviation for a county FIPS."""
    return STATE_FIPS.get(fips[:2], "??")


@lru_cache(maxsize=1)
def load_fips_table() -> pd.DataFrame:
    """Load the full FIPS-to-county-name lookup table.

    Returns a DataFrame with columns: fips, state_fips, state_abbr,
    county_name, full_name.

    Expects a CSV at data/raw/fips_counties.csv. If not present,
    raises FileNotFoundError — run ingest/census_acs.py first to
    download it.
    """
    path = RAW_DIR / "fips_counties.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"FIPS lookup table not found at {path}. "
            "Run the Census ACS ingester to download county reference data."
        )
    df = pd.read_csv(path, dtype={"fips": str, "state_fips": str})
    return df
