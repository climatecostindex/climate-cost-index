"""Fetch Consumer Expenditure Survey tables from BLS.

Source: BLS Consumer Expenditure Survey
URL: https://www.bls.gov/cex/tables.htm
Format: Published tables (Excel/CSV)

Fetches raw expenditure tables ONLY. Does NOT compute CCI weights or
map CE categories to CCI components — those belong in config or transform.

Preserve CE item codes so the CE-to-CCI category mapping is traceable.

Output columns: item_code, category, annual_expenditure, pct_of_total, year
Confidence: A
Attribution: none (reference data for weighting, not a scored component)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BLS CE configuration
# ---------------------------------------------------------------------------

# Base URL for CE multi-year average tables (all consumer units).
BLS_CE_BASE_URL = "https://www.bls.gov/cex/tables/calendar-year/mean"

# BLS publishes multi-year tables as
# ``cu-all-multi-year-{start}-{end}.xlsx``.  As of 2025, two files
# cover the full available history:
BLS_CE_TABLE_URLS = [
    f"{BLS_CE_BASE_URL}/cu-all-multi-year-2013-2020.xlsx",
    f"{BLS_CE_BASE_URL}/cu-all-multi-year-2021-2024.xlsx",
]

# BLS blocks requests without a browser-like User-Agent header.
BLS_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Rate limit: static file downloads, be polite (1 req/sec)
BLS_CE_CALLS_PER_SECOND = 1.0

# BLS footnote markers that indicate non-numeric / suppressed data
BLS_FOOTNOTE_MARKERS = {"b/", "c/", "f/", "n.a.", "n.a"}

# ---------------------------------------------------------------------------
# Expected Excel structure markers
# ---------------------------------------------------------------------------

# The CE table uses a hierarchical layout. The first column contains
# item descriptions (with indentation indicating hierarchy). A separate
# column (or the same row) contains item codes. Expenditure values
# appear in subsequent columns, one per year.
#
# We look for these markers to validate the table structure.
EXPECTED_HEADER_MARKER = "Item"
EXPECTED_TOTAL_ROW_MARKER = "Average annual expenditures"


class BLSCEIngester(BaseIngester):
    """Ingest BLS Consumer Expenditure Survey tables.

    Downloads published CE annual expenditure tables (Excel format) from
    bls.gov. Parses the hierarchical category structure, preserving item
    codes and computing pct_of_total as a direct property of the source.

    This is reference data for CCI component weighting — downloaded once
    per methodology cycle, not a recurring pull.
    """

    source_name = "bls_ce"
    confidence = "A"
    attribution = "none"
    calls_per_second = BLS_CE_CALLS_PER_SECOND

    required_columns: dict[str, type] = {
        "item_code": str,
        "category": str,
        "annual_expenditure": float,
        "pct_of_total": float,
        "year": int,
    }

    def _download_table(self, url: str) -> bytes:
        """Download the CE Excel table from BLS.

        BLS blocks requests without a browser-like User-Agent, so we
        include one in the request headers.

        Args:
            url: Full URL to the Excel file.

        Returns:
            Raw bytes of the downloaded file.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        logger.info("BLS_CE: downloading table from %s", url)
        resp = self.api_get(url, headers={"User-Agent": BLS_USER_AGENT})
        return resp.content

    def _parse_excel(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse BLS CE Excel table into a flat DataFrame.

        The CE multi-year table has a complex layout:
        - Row 0-N: header rows (title, blank lines, column headers)
        - A column with item descriptions (hierarchical, indented)
        - One or more columns with annual expenditure values by year

        This method finds the data region, extracts item codes, category
        names, and expenditure values for each available year.

        Args:
            raw_bytes: Raw Excel file content.

        Returns:
            DataFrame with columns: item_code, category, annual_expenditure,
            pct_of_total, year.

        Raises:
            ValueError: If the Excel structure doesn't match expectations.
        """
        # Read raw Excel — header=None so we can inspect the full layout
        xls = pd.read_excel(
            io.BytesIO(raw_bytes),
            header=None,
            dtype=str,
            engine="openpyxl",
        )

        # Find the header row containing "Item" (case-insensitive)
        header_row_idx = self._find_header_row(xls)
        if header_row_idx is None:
            raise ValueError(
                f"BLS_CE: could not find header row containing '{EXPECTED_HEADER_MARKER}' "
                f"in downloaded Excel file. The table structure may have changed. "
                f"Columns found in first 20 rows: {xls.head(20).to_string()}"
            )

        # Extract column headers from the header row
        headers = xls.iloc[header_row_idx].tolist()
        logger.info("BLS_CE: found header row at index %d: %s", header_row_idx, headers[:6])

        # Data rows start after the header
        data = xls.iloc[header_row_idx + 1:].copy()
        data.columns = range(len(data.columns))

        # Identify which columns contain the item description and year values.
        # Column 0 is typically the item description. Year columns contain
        # 4-digit year strings in the header row.
        item_col = 0
        year_cols = self._find_year_columns(headers)

        if not year_cols:
            raise ValueError(
                "BLS_CE: could not find any year columns in the header row. "
                f"Headers: {headers}"
            )

        logger.info(
            "BLS_CE: found %d year columns: %s",
            len(year_cols),
            {col: yr for col, yr in year_cols},
        )

        # Extract only the expenditure section of the table.
        # The multi-year table starts with demographic characteristics,
        # then has the expenditure section starting at "Average annual
        # expenditures", followed by income sources and footnotes.
        # We start from the total row and stop at known section breaks.
        #
        # If no total row is found (e.g. simpler table format), fall back
        # to including all rows.
        rows = []
        total_expenditure_by_year: dict[int, float] = {}

        # Pre-scan: does the table have the total row?
        has_total_row = any(
            self._is_total_row(str(row[item_col]).strip())
            for _, row in data.iterrows()
            if pd.notna(row[item_col])
        )
        in_expenditure_section = not has_total_row  # start open if no total row

        for _, row in data.iterrows():
            item_text = str(row[item_col]).strip() if pd.notna(row[item_col]) else ""
            if not item_text or item_text == "nan":
                continue

            # Detect start of expenditure section
            if self._is_total_row(item_text):
                in_expenditure_section = True

            # Detect end of expenditure section (income/tax/addenda rows)
            if in_expenditure_section and self._is_post_expenditure_row(item_text):
                break

            if not in_expenditure_section:
                continue

            category = item_text
            item_code = self._extract_item_code(category)

            for col_idx, year in year_cols:
                raw_val = row[col_idx] if col_idx < len(row) else None
                expenditure = self._parse_expenditure(raw_val)
                if expenditure is None:
                    continue

                rows.append({
                    "item_code": item_code,
                    "category": category,
                    "annual_expenditure": expenditure,
                    "year": year,
                })

                # Track the total for pct_of_total computation
                if self._is_total_row(category):
                    total_expenditure_by_year[year] = expenditure

        if not rows:
            raise ValueError(
                "BLS_CE: no expenditure data rows found in the Excel file. "
                "The table structure may have changed."
            )

        df = pd.DataFrame(rows)

        # Compute pct_of_total
        df["pct_of_total"] = 0.0
        for year, total in total_expenditure_by_year.items():
            if total > 0:
                mask = df["year"] == year
                df.loc[mask, "pct_of_total"] = (
                    df.loc[mask, "annual_expenditure"] / total * 100.0
                )

        if not total_expenditure_by_year:
            logger.warning(
                "BLS_CE: could not find total expenditure row ('%s'). "
                "pct_of_total will be 0 for all rows.",
                EXPECTED_TOTAL_ROW_MARKER,
            )

        # Enforce dtypes
        df["item_code"] = df["item_code"].astype(str)
        df["category"] = df["category"].astype(str)
        df["annual_expenditure"] = df["annual_expenditure"].astype(float)
        df["pct_of_total"] = df["pct_of_total"].astype(float)
        df["year"] = df["year"].astype(int)

        return df

    def _find_header_row(self, xls: pd.DataFrame) -> int | None:
        """Find the row index containing the column headers.

        Searches the first 30 rows for one containing 'Item' (case-insensitive).

        Args:
            xls: Raw Excel DataFrame (no header).

        Returns:
            Row index, or None if not found.
        """
        search_limit = min(30, len(xls))
        for idx in range(search_limit):
            row_vals = [str(v).strip().lower() for v in xls.iloc[idx] if pd.notna(v)]
            if any(EXPECTED_HEADER_MARKER.lower() in val for val in row_vals):
                return idx
        return None

    def _find_year_columns(self, headers: list) -> list[tuple[int, int]]:
        """Identify columns containing year values (e.g. '2022', '2023').

        Args:
            headers: List of header cell values.

        Returns:
            List of (column_index, year) tuples.
        """
        year_cols = []
        for i, h in enumerate(headers):
            h_str = str(h).strip()
            if h_str.isdigit() and 1990 <= int(h_str) <= 2099:
                year_cols.append((i, int(h_str)))
        return year_cols

    @staticmethod
    def _extract_item_code(category: str) -> str:
        """Generate a normalized item code from the category name.

        BLS CE tables don't always include explicit item codes in the
        published Excel files. We preserve the category name and create
        a normalized code by uppercasing, replacing spaces and special
        characters, and truncating.

        Args:
            category: Raw category text from the CE table.

        Returns:
            Normalized item code string.
        """
        # Strip leading whitespace (hierarchy indentation) for the code,
        # but preserve the full category text as-is.
        stripped = category.strip()
        # Create code: uppercase, replace non-alphanumeric with underscore
        code = ""
        for ch in stripped.upper():
            if ch.isalnum():
                code += ch
            elif code and code[-1] != "_":
                code += "_"
        # Remove trailing underscore
        code = code.rstrip("_")
        return code if code else "UNKNOWN"

    @staticmethod
    def _parse_expenditure(raw_val: object) -> float | None:
        """Parse an expenditure cell value to float.

        Handles common formats: plain numbers, numbers with commas,
        dollar signs. Returns None for BLS footnote markers (b/, c/, f/,
        n.a.) and other non-numeric cells.

        Args:
            raw_val: Raw cell value from Excel.

        Returns:
            Float value, or None if not parseable as a number.
        """
        if raw_val is None or pd.isna(raw_val):
            return None
        s = str(raw_val).strip()
        # BLS footnote markers indicate suppressed / unavailable data
        if s.lower() in BLS_FOOTNOTE_MARKERS:
            return None
        # Remove common formatting: $, commas
        s = s.replace("$", "").replace(",", "").strip()
        # Remove trailing footnote references (e.g., "1234.56 1" or "1234.56(2)")
        # We keep only the leading numeric portion
        parts = s.split()
        if parts:
            s = parts[0]
        s = s.rstrip("()")
        try:
            val = float(s)
            return val
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _is_total_row(category: str) -> bool:
        """Check if a category row represents the total expenditure.

        Args:
            category: Category text from the CE table.

        Returns:
            True if this is the total expenditure row.
        """
        normalized = category.strip().lower()
        return EXPECTED_TOTAL_ROW_MARKER.lower() in normalized

    @staticmethod
    def _is_post_expenditure_row(category: str) -> bool:
        """Check if a row marks the end of the expenditure section.

        The BLS CE table has income sources, tax data, and addenda
        after the expenditure categories. We stop parsing at these.

        Args:
            category: Category text from the CE table.

        Returns:
            True if this row is past the expenditure section.
        """
        normalized = category.strip().lower()
        post_markers = [
            "sources of pretax income",
            "income before taxes",
            "income after taxes",
            "personal taxes",
            "addenda",
            "other financial information",
        ]
        return any(marker in normalized for marker in post_markers)

    def fetch(self, years: list[int] | None = None) -> pd.DataFrame:
        """Fetch BLS Consumer Expenditure Survey data.

        Downloads all available CE multi-year average tables from BLS,
        parses them, and merges into a single DataFrame covering the
        full history (currently 2013-2024).

        Args:
            years: If provided, return only these survey years. If None,
                   return all years found across all tables.

        Returns:
            DataFrame with columns: item_code, category, annual_expenditure,
            pct_of_total, year.
        """
        frames: list[pd.DataFrame] = []
        for url in BLS_CE_TABLE_URLS:
            try:
                raw_bytes = self._download_table(url)
                frame = self._parse_excel(raw_bytes)
                frames.append(frame)
                logger.info("BLS_CE: parsed %d rows from %s", len(frame), url)
            except Exception:
                logger.warning("BLS_CE: failed to fetch/parse %s, skipping", url, exc_info=True)

        if not frames:
            logger.warning("BLS_CE: no data parsed from any CE table")
            return pd.DataFrame(columns=list(self.required_columns))

        df = pd.concat(frames, ignore_index=True)

        # Filter to requested years if specified
        if years is not None:
            df = df[df["year"].isin(years)].copy()
            if df.empty:
                logger.warning("BLS_CE: no records match requested years %s", years)
                return pd.DataFrame(columns=list(self.required_columns))

        # Cache per year
        for yr in sorted(df["year"].unique()):
            year_df = df[df["year"] == yr]
            self.cache_raw(
                year_df,
                label=f"bls_ce_{yr}",
                data_vintage=f"BLS CE Survey {yr}",
            )

        # Cache combined
        year_min, year_max = df["year"].min(), df["year"].max()
        self.cache_raw(
            df,
            label="bls_ce_all",
            data_vintage=f"BLS CE Survey {year_min} to {year_max}",
        )

        logger.info(
            "BLS_CE: parsed %d rows across %d years, %d categories",
            len(df),
            df["year"].nunique(),
            df["item_code"].nunique(),
        )

        return df
