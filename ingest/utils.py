"""Shared ingestion utilities: FIPS normalization, HTTP helpers, constants."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from tqdm import tqdm

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)

# Re-export so ingesters can do ``from ingest.utils import DATA_RAW_DIR``
DATA_RAW_DIR: Path = RAW_DIR


def normalize_fips(raw: str | int | float) -> str:
    """Normalize a FIPS code to a 5-digit zero-padded string.

    Handles common raw formats:
        - int:   1001   → "01001"
        - float: 1001.0 → "01001"
        - str:   "1001" → "01001", "01001" → "01001"

    Args:
        raw: Raw FIPS code in any common format.

    Returns:
        5-digit zero-padded string.

    Raises:
        ValueError: If the value cannot be converted to a valid 5-digit FIPS.
    """
    if isinstance(raw, float):
        raw = int(raw)
    code = str(raw).strip().lstrip("0") or "0"
    numeric = int(code)
    result = f"{numeric:05d}"
    if len(result) != 5:
        raise ValueError(f"FIPS code out of range: {raw!r} → {result}")
    return result


def fips_5digit(state_fips: str | int, county_fips: str | int) -> str:
    """Combine state and county FIPS into a zero-padded 5-digit code."""
    return f"{int(state_fips):02d}{int(county_fips):03d}"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file with progress bar. Skips if dest already exists."""
    if dest.exists():
        logger.info("Already downloaded: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    with httpx.stream("GET", url, timeout=300.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in resp.iter_bytes(chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest
