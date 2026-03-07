"""Database connection and query layer.

Uses DuckDB for local development. Can be swapped to PostgreSQL for production
by changing the connection factory.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import duckdb

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "cci.duckdb"


@contextmanager
def get_connection(read_only: bool = False):
    """Yield a DuckDB connection, closing on exit."""
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()
