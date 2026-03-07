"""Abstract base ingester with rate limiting, retry, caching, and schema validation."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)

# Approximate number of U.S. counties (used for completeness reporting)
TOTAL_US_COUNTIES = 3_143

# HTTP status codes that trigger retry
RETRYABLE_STATUS_CODES = {429, 500, 503}


class BaseIngester(ABC):
    """Abstract base class for all data ingesters.

    Subclasses must define:
        source_name: Short identifier (e.g. "usdm").
        confidence: Data confidence grade ("A", "B", or "C").
        attribution: Attribution class ("attributed", "proxy", or "none").
        required_columns: Dict mapping column name → expected dtype.

    Subclasses implement ``fetch()`` which returns a DataFrame conforming to
    the subclass-defined schema. The base class provides:
        - HTTP client with rate limiting and exponential-backoff retry
        - Raw data caching to disk (parquet + metadata JSON sidecar)
        - Output schema validation
        - Completeness logging (% of ~3,100 U.S. counties covered)
    """

    source_name: str = ""
    confidence: str = ""
    attribution: str = ""
    required_columns: dict[str, type] = {}
    calls_per_second: float = 5.0
    max_retries: int = 3
    retry_backoff_base: float = 2.0

    def __init__(self) -> None:
        self._last_call_time: float = 0.0
        self._client: httpx.Client | None = None

    # -- HTTP client with rate limiting ----------------------------------------

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialized HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=60.0, follow_redirects=True)
        return self._client

    def rate_limit(self) -> None:
        """Enforce minimum interval between HTTP calls."""
        if self.calls_per_second <= 0:
            return
        min_interval = 1.0 / self.calls_per_second
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.monotonic()

    def api_get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """GET with rate limiting and exponential-backoff retry.

        Retries on HTTP 429, 500, 503 and transport errors (timeouts, etc.).
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.rate_limit()
            try:
                resp = self.client.get(url, params=params, headers=headers)
                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    wait = self.retry_backoff_base ** attempt
                    logger.warning(
                        "%s: HTTP %d (attempt %d/%d), retrying in %.1fs",
                        self.source_name,
                        resp.status_code,
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise
                wait = self.retry_backoff_base ** attempt
                logger.warning(
                    "%s: transport error (attempt %d/%d), retrying in %.1fs: %s",
                    self.source_name,
                    attempt + 1,
                    self.max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
        # Should not reach here, but satisfy type checker
        raise last_exc  # type: ignore[misc]

    # -- Caching ---------------------------------------------------------------

    def cache_dir(self) -> Path:
        """Return ``data/raw/{source_name}/``, creating it if needed."""
        d = RAW_DIR / self.source_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cache_raw(
        self,
        df: pd.DataFrame,
        label: str,
        data_vintage: str,
    ) -> Path:
        """Save DataFrame as parquet and write a metadata JSON sidecar.

        Args:
            df: The data to persist.
            label: File stem (e.g. ``"usdm_2024"``).
            data_vintage: Human-readable date range of the source data.

        Returns:
            Path to the saved parquet file.
        """
        parquet_path = self.cache_dir() / f"{label}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("%s: saved %d rows to %s", self.source_name, len(df), parquet_path)

        metadata = {
            "source": self.source_name.upper(),
            "confidence": self.confidence,
            "attribution": self.attribution,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "data_vintage": data_vintage,
        }
        meta_path = self.cache_dir() / f"{label}_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info("%s: wrote metadata to %s", self.source_name, meta_path)

        return parquet_path

    def load_raw(self, label: str) -> pd.DataFrame | None:
        """Load cached raw parquet if it exists."""
        path = self.cache_dir() / f"{label}.parquet"
        if path.exists():
            logger.info("%s: loading cached data from %s", self.source_name, path)
            return pd.read_parquet(path)
        return None

    # -- Schema validation -----------------------------------------------------

    def validate_output(self, df: pd.DataFrame) -> None:
        """Check that all required columns exist and have correct types.

        Uses the subclass-defined ``required_columns`` mapping.
        Raises ``ValueError`` on schema violations.
        """
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"{self.source_name}: output missing required columns: {missing}")

        extra = set(df.columns) - set(self.required_columns)
        if extra:
            raise ValueError(
                f"{self.source_name}: output has unexpected columns: {extra}. "
                "Raw ingest output must contain ONLY the defined schema columns."
            )

        for col, expected_type in self.required_columns.items():
            if expected_type is str and not pd.api.types.is_string_dtype(df[col]):
                raise ValueError(
                    f"{self.source_name}: column '{col}' should be string, "
                    f"got {df[col].dtype}"
                )
            if expected_type is float and not pd.api.types.is_float_dtype(df[col]):
                raise ValueError(
                    f"{self.source_name}: column '{col}' should be float, "
                    f"got {df[col].dtype}"
                )

    # -- Completeness logging --------------------------------------------------

    def log_completeness(self, df: pd.DataFrame) -> None:
        """Log what percentage of ~3,100 U.S. counties have data."""
        n_counties = df["fips"].nunique() if "fips" in df.columns else 0
        pct = (n_counties / TOTAL_US_COUNTIES) * 100
        n_rows = len(df)
        logger.info(
            "%s completeness: %d rows, %d counties (%.1f%% of %d U.S. counties)",
            self.source_name,
            n_rows,
            n_counties,
            pct,
            TOTAL_US_COUNTIES,
        )

    # -- Template method -------------------------------------------------------

    @abstractmethod
    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch raw data from the source.

        Returns a DataFrame conforming to this ingester's ``required_columns``.
        """
        ...

    def run(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch, validate, log completeness, cache, and return data."""
        logger.info("%s: starting ingestion", self.source_name)
        df = self.fetch(years=years)
        self.validate_output(df)
        self.log_completeness(df)
        return df

    # -- Cleanup ---------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> BaseIngester:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
