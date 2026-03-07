"""Tests for the BLS Consumer Expenditure Survey ingester (ingest/bls_ce.py).

All HTTP calls are mocked — no real requests to bls.gov.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from ingest.bls_ce import (
    BLS_CE_TABLE_URLS,
    BLSCEIngester,
    EXPECTED_HEADER_MARKER,
    EXPECTED_TOTAL_ROW_MARKER,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _build_ce_excel(
    years: list[int] | None = None,
    include_total: bool = True,
    extra_header_rows: int = 2,
) -> bytes:
    """Build a synthetic CE-style Excel file in memory.

    Mimics the BLS CE multi-year table layout:
    - A few blank/title rows at the top
    - A header row with "Item" and year columns
    - Data rows with category names and expenditure values
    - A total row ("Average annual expenditures")

    Args:
        years: Year columns to include. Defaults to [2022, 2023].
        include_total: Whether to include the total expenditure row.
        extra_header_rows: Number of blank/title rows before the header.

    Returns:
        Excel file bytes.
    """
    if years is None:
        years = [2022, 2023]

    # Build the spreadsheet row by row
    rows: list[list[str | float | None]] = []

    # Extra header rows (title, blank lines)
    for i in range(extra_header_rows):
        row = ["Table 1100. Consumer unit characteristics" if i == 0 else None]
        row.extend([None] * len(years))
        rows.append(row)

    # Column header row
    header_row: list[str | float | None] = ["Item"]
    header_row.extend([str(y) for y in years])
    rows.append(header_row)

    # Category data (simplified CE structure with hierarchy)
    categories = [
        ("Average annual expenditures", {2022: 72967.0, 2023: 77280.0}),
        ("Food", {2022: 9343.0, 2023: 9826.0}),
        ("  Food at home", {2022: 5703.0, 2023: 5955.0}),
        ("  Food away from home", {2022: 3639.0, 2023: 3871.0}),
        ("Housing", {2022: 24298.0, 2023: 25432.0}),
        ("  Shelter", {2022: 14531.0, 2023: 15329.0}),
        ("  Utilities, fuels, and public services", {2022: 4714.0, 2023: 4891.0}),
        ("  Household operations", {2022: 1580.0, 2023: 1641.0}),
        ("  Housekeeping supplies", {2022: 797.0, 2023: 823.0}),
        ("  Household furnishings and equipment", {2022: 2177.0, 2023: 2234.0}),
        ("Transportation", {2022: 12295.0, 2023: 13116.0}),
        ("  Vehicle purchases", {2022: 5152.0, 2023: 5527.0}),
        ("  Gasoline, other fuels, and motor oil", {2022: 2707.0, 2023: 2647.0}),
        ("Healthcare", {2022: 5850.0, 2023: 6191.0}),
        ("Entertainment", {2022: 3458.0, 2023: 3659.0}),
        ("Personal insurance and pensions", {2022: 8617.0, 2023: 9139.0}),
    ]

    if not include_total:
        # Remove the total row
        categories = [c for c in categories if EXPECTED_TOTAL_ROW_MARKER not in c[0]]

    for cat_name, expenditures in categories:
        row: list[str | float | None] = [cat_name]
        for yr in years:
            row.append(expenditures.get(yr))
        rows.append(row)

    # Convert to DataFrame and write to Excel
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, header=False, engine="openpyxl")
    return buf.getvalue()


def _make_httpx_response(content: bytes, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response with binary content."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# Prebuilt fixture data
SAMPLE_EXCEL_BYTES = _build_ce_excel(years=[2022, 2023])
SINGLE_YEAR_EXCEL_BYTES = _build_ce_excel(years=[2023])


def _mock_download_single(excel_bytes: bytes):
    """Return a side_effect function for _download_table that serves one table.

    The first URL gets the fixture bytes; subsequent URLs raise 404 so
    fetch() skips them (matching real behaviour when only one table exists).
    """
    call_count = 0

    def _side_effect(url: str) -> bytes:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return excel_bytes
        raise httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock(status_code=404),
        )

    return _side_effect


@pytest.fixture
def ingester():
    """Return a fresh BLSCEIngester instance."""
    return BLSCEIngester()


@pytest.fixture
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """Redirect RAW_DIR to a temp directory."""
    monkeypatch.setattr("ingest.base.RAW_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Test: Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify output DataFrame has all expected columns with correct dtypes."""

    def test_columns_present(self, ingester, tmp_raw_dir):
        """Output contains exactly the required columns."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        expected = {"item_code", "category", "annual_expenditure", "pct_of_total", "year"}
        assert set(df.columns) == expected

    def test_item_code_is_string(self, ingester, tmp_raw_dir):
        """item_code column is string dtype."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert pd.api.types.is_string_dtype(df["item_code"])

    def test_category_is_string(self, ingester, tmp_raw_dir):
        """category column is string dtype."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert pd.api.types.is_string_dtype(df["category"])

    def test_annual_expenditure_is_float(self, ingester, tmp_raw_dir):
        """annual_expenditure column is float dtype."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert pd.api.types.is_float_dtype(df["annual_expenditure"])

    def test_pct_of_total_is_float(self, ingester, tmp_raw_dir):
        """pct_of_total column is float dtype."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert pd.api.types.is_float_dtype(df["pct_of_total"])

    def test_year_is_int(self, ingester, tmp_raw_dir):
        """year column is integer dtype."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert pd.api.types.is_integer_dtype(df["year"])


# ---------------------------------------------------------------------------
# Test: Item codes preserved
# ---------------------------------------------------------------------------

class TestItemCodesPreserved:
    """Verify CE item codes are present and unchanged from source."""

    def test_item_codes_present(self, ingester, tmp_raw_dir):
        """Every row has a non-empty item_code."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert (df["item_code"].str.len() > 0).all()

    def test_item_codes_are_normalized(self, ingester, tmp_raw_dir):
        """Item codes are uppercase, alphanumeric + underscores."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        for code in df["item_code"]:
            assert code == code.upper(), f"Item code not uppercase: {code}"
            assert all(
                ch.isalnum() or ch == "_" for ch in code
            ), f"Item code has invalid chars: {code}"

    def test_known_categories_have_codes(self, ingester, tmp_raw_dir):
        """Key categories (food, housing, transport, healthcare) have item codes."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        codes = set(df["item_code"].values)
        # These should exist based on our synthetic data
        assert "FOOD" in codes
        assert "HOUSING" in codes
        assert "TRANSPORTATION" in codes
        assert "HEALTHCARE" in codes


# ---------------------------------------------------------------------------
# Test: Category parsing
# ---------------------------------------------------------------------------

class TestCategoryParsing:
    """Verify key categories are extracted from the table."""

    def test_key_categories_present(self, ingester, tmp_raw_dir):
        """Utilities/energy, housing, food, healthcare, transportation appear."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        categories_lower = set(df["category"].str.strip().str.lower())
        assert "food" in categories_lower
        assert "housing" in categories_lower
        assert "healthcare" in categories_lower
        assert "transportation" in categories_lower

    def test_utilities_subcategory_present(self, ingester, tmp_raw_dir):
        """Utilities/fuels subcategory appears under housing."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        # Look for the utilities subcategory (indented in the CE table)
        assert any(
            "utilities" in cat.lower()
            for cat in df["category"].values
        )


# ---------------------------------------------------------------------------
# Test: Percentage computation
# ---------------------------------------------------------------------------

class TestPercentageComputation:
    """Verify pct_of_total values are correct."""

    def test_pct_of_total_sums_approximately_100(self, ingester, tmp_raw_dir):
        """pct_of_total for all non-total categories sums near 100% per year.

        Note: In the CE table, subcategories are included separately from
        their parents, so the raw sum of all rows' pct_of_total will exceed
        100%. The total row itself should be ~100%.
        """
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        # The total row itself should have pct_of_total == 100
        for year in df["year"].unique():
            year_df = df[df["year"] == year]
            total_rows = year_df[
                year_df["category"].str.strip().str.lower().str.contains("average annual expenditures")
            ]
            if not total_rows.empty:
                assert abs(total_rows.iloc[0]["pct_of_total"] - 100.0) < 0.01

    def test_pct_of_total_positive(self, ingester, tmp_raw_dir):
        """All pct_of_total values are non-negative."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert (df["pct_of_total"] >= 0).all()

    def test_pct_of_total_computed_correctly(self, ingester, tmp_raw_dir):
        """pct_of_total = (category / total) * 100 for a known category."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        # Check Food for year 2022: 9343 / 72967 * 100 ≈ 12.81
        food_2022 = df[
            (df["category"].str.strip() == "Food") & (df["year"] == 2022)
        ]
        assert len(food_2022) == 1
        expected_pct = 9343.0 / 72967.0 * 100.0
        assert abs(food_2022.iloc[0]["pct_of_total"] - expected_pct) < 0.01


# ---------------------------------------------------------------------------
# Test: Ingest purity check
# ---------------------------------------------------------------------------

class TestIngestPurity:
    """Verify NO CCI-derived metrics in output."""

    def test_no_cci_columns(self, ingester, tmp_raw_dir):
        """Output must not contain CCI-derived columns."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        forbidden = {
            "cci_weight", "component_mapping", "normalized_share",
            "weight", "cci_component", "score", "percentile",
        }
        assert forbidden.isdisjoint(set(df.columns))

    def test_exact_column_set(self, ingester, tmp_raw_dir):
        """Output columns are EXACTLY the required set."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        expected = {"item_code", "category", "annual_expenditure", "pct_of_total", "year"}
        assert set(df.columns) == expected

    def test_validate_output_rejects_extra_columns(self, ingester):
        """Base class validate_output() raises on unexpected columns."""
        df = pd.DataFrame({
            "item_code": ["FOOD"],
            "category": ["Food"],
            "annual_expenditure": [9343.0],
            "pct_of_total": [12.81],
            "year": [2022],
            "cci_weight": [0.15],  # FORBIDDEN derived column
        })
        with pytest.raises(ValueError, match="unexpected columns"):
            ingester.validate_output(df)


# ---------------------------------------------------------------------------
# Test: Metadata sidecar
# ---------------------------------------------------------------------------

class TestMetadataSidecar:
    """Verify metadata JSON is written alongside parquet files."""

    def test_metadata_file_created(self, ingester, tmp_raw_dir):
        """A _metadata.json file is written next to each year's parquet."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            ingester.fetch()

        assert (tmp_raw_dir / "bls_ce" / "bls_ce_2022_metadata.json").exists()
        assert (tmp_raw_dir / "bls_ce" / "bls_ce_2023_metadata.json").exists()

    def test_metadata_content(self, ingester, tmp_raw_dir):
        """Metadata JSON has correct source, confidence, attribution."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            ingester.fetch()

        meta_path = tmp_raw_dir / "bls_ce" / "bls_ce_2022_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["source"] == "BLS_CE"
        assert meta["confidence"] == "A"
        assert meta["attribution"] == "none"
        assert "retrieved_at" in meta
        assert "data_vintage" in meta

    def test_combined_metadata_written(self, ingester, tmp_raw_dir):
        """The combined bls_ce_all file gets a metadata sidecar."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            ingester.fetch()

        assert (tmp_raw_dir / "bls_ce" / "bls_ce_all.parquet").exists()
        assert (tmp_raw_dir / "bls_ce" / "bls_ce_all_metadata.json").exists()


# ---------------------------------------------------------------------------
# Test: Hierarchical preservation
# ---------------------------------------------------------------------------

class TestHierarchicalPreservation:
    """Verify subcategories are retained as distinct rows."""

    def test_subcategories_present(self, ingester, tmp_raw_dir):
        """Subcategories like 'Food at home' appear as distinct rows."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        categories = df["category"].str.strip().tolist()
        assert "Food at home" in categories
        assert "Food away from home" in categories

    def test_utilities_subcategory_distinct_from_housing(self, ingester, tmp_raw_dir):
        """Utilities appears as a separate row from Housing."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        categories = df["category"].str.strip().tolist()
        assert "Housing" in categories
        assert any("Utilities" in c for c in categories)

    def test_parent_and_child_both_present(self, ingester, tmp_raw_dir):
        """Both parent categories and their subcategories exist in output."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        categories = set(df["category"].str.strip().values)
        # Transportation and its subcategory
        assert "Transportation" in categories
        assert "Vehicle purchases" in categories
        assert "Gasoline, other fuels, and motor oil" in categories


# ---------------------------------------------------------------------------
# Test: Malformed Excel handling
# ---------------------------------------------------------------------------

class TestMalformedExcelHandling:
    """Verify clear error on unexpected table structure.

    These test _parse_excel() directly since fetch() catches exceptions
    per-URL and logs warnings instead of raising.
    """

    def test_missing_header_row_raises(self, ingester):
        """Excel with no 'Item' header row raises ValueError."""
        df_bad = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        buf = io.BytesIO()
        df_bad.to_excel(buf, index=False, engine="openpyxl")
        bad_bytes = buf.getvalue()

        with pytest.raises(ValueError, match="could not find header row"):
            ingester._parse_excel(bad_bytes)

    def test_no_year_columns_raises(self, ingester):
        """Excel with 'Item' header but no year columns raises ValueError."""
        rows = [
            ["Title Row"],
            ["Item", "SomeColumn", "AnotherColumn"],
            ["Food", "abc", "def"],
        ]
        df_bad = pd.DataFrame(rows)
        buf = io.BytesIO()
        df_bad.to_excel(buf, index=False, header=False, engine="openpyxl")
        bad_bytes = buf.getvalue()

        with pytest.raises(ValueError, match="could not find any year columns"):
            ingester._parse_excel(bad_bytes)

    def test_no_data_rows_raises(self, ingester):
        """Excel with header row but no data rows raises ValueError."""
        rows = [
            ["Item", "2022"],
            # No data rows follow
        ]
        df_bad = pd.DataFrame(rows)
        buf = io.BytesIO()
        df_bad.to_excel(buf, index=False, header=False, engine="openpyxl")
        bad_bytes = buf.getvalue()

        with pytest.raises(ValueError, match="no expenditure data rows"):
            ingester._parse_excel(bad_bytes)


# ---------------------------------------------------------------------------
# Test: Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify retry logic triggers on HTTP 500/503."""

    def test_retries_on_500(self, ingester, tmp_raw_dir):
        """Retries after HTTP 500 and succeeds on subsequent attempt."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500

        ok_resp = _make_httpx_response(SAMPLE_EXCEL_BYTES)

        call_count = 0

        def mock_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fail_resp
            return ok_resp

        with patch.object(ingester, "_client", MagicMock()):
            ingester._client.get = mock_get
            ingester._last_call_time = 0.0
            resp = ingester.api_get(BLS_CE_TABLE_URLS[0])

        assert call_count == 2

    def test_retries_on_503(self, ingester, tmp_raw_dir):
        """Retries after HTTP 503."""
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503

        ok_resp = _make_httpx_response(SAMPLE_EXCEL_BYTES)

        call_count = 0

        def mock_get(url, params=None, headers=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return fail_resp
            return ok_resp

        with patch.object(ingester, "_client", MagicMock()):
            ingester._client.get = mock_get
            ingester._last_call_time = 0.0
            resp = ingester.api_get(BLS_CE_TABLE_URLS[0])

        assert call_count == 3


# ---------------------------------------------------------------------------
# Test: Year filtering
# ---------------------------------------------------------------------------

class TestYearFiltering:
    """Verify that the years parameter filters results."""

    def test_filter_to_single_year(self, ingester, tmp_raw_dir):
        """Passing years=[2022] returns only 2022 records."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch(years=[2022])

        assert set(df["year"].unique()) == {2022}

    def test_filter_no_match_returns_empty(self, ingester, tmp_raw_dir):
        """Passing years with no match returns empty DataFrame."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch(years=[1999])

        assert df.empty
        assert set(df.columns) == set(ingester.required_columns)

    def test_no_year_filter_returns_all(self, ingester, tmp_raw_dir):
        """No year filter returns all years in the table."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        assert set(df["year"].unique()) == {2022, 2023}


# ---------------------------------------------------------------------------
# Test: Class attributes
# ---------------------------------------------------------------------------

class TestClassAttributes:
    """Verify ingester metadata class attributes."""

    def test_source_name(self, ingester):
        assert ingester.source_name == "bls_ce"

    def test_confidence(self, ingester):
        assert ingester.confidence == "A"

    def test_attribution(self, ingester):
        assert ingester.attribution == "none"

    def test_calls_per_second(self, ingester):
        """Rate limit is polite (<=1 req/sec for static downloads)."""
        assert ingester.calls_per_second <= 1.0


# ---------------------------------------------------------------------------
# Test: Caching
# ---------------------------------------------------------------------------

class TestCaching:
    """Verify per-year and combined caching behavior."""

    def test_per_year_parquet_files(self, ingester, tmp_raw_dir):
        """Each year in the data gets its own parquet file."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            ingester.fetch()

        assert (tmp_raw_dir / "bls_ce" / "bls_ce_2022.parquet").exists()
        assert (tmp_raw_dir / "bls_ce" / "bls_ce_2023.parquet").exists()

    def test_combined_parquet_file(self, ingester, tmp_raw_dir):
        """A combined bls_ce_all.parquet is written."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            ingester.fetch()

        assert (tmp_raw_dir / "bls_ce" / "bls_ce_all.parquet").exists()

    def test_cached_data_matches_returned(self, ingester, tmp_raw_dir):
        """Cached parquet data matches what fetch() returned."""
        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(SAMPLE_EXCEL_BYTES)):
            df = ingester.fetch()

        cached = pd.read_parquet(tmp_raw_dir / "bls_ce" / "bls_ce_all.parquet")
        assert len(cached) == len(df)
        assert set(cached.columns) == set(df.columns)


# ---------------------------------------------------------------------------
# Test: Missing total row
# ---------------------------------------------------------------------------

class TestMissingTotalRow:
    """Verify behavior when the total expenditure row is missing."""

    def test_pct_of_total_zero_without_total(self, ingester, tmp_raw_dir):
        """Without total row, pct_of_total is 0 (logged as warning)."""
        no_total_bytes = _build_ce_excel(years=[2023], include_total=False)

        with patch.object(ingester, "_download_table", side_effect=_mock_download_single(no_total_bytes)):
            df = ingester.fetch()

        assert (df["pct_of_total"] == 0.0).all()

    def test_warning_logged_without_total(self, ingester, tmp_raw_dir, caplog):
        """A warning is logged when the total row is missing."""
        import logging

        no_total_bytes = _build_ce_excel(years=[2023], include_total=False)

        with caplog.at_level(logging.WARNING):
            with patch.object(ingester, "_download_table", side_effect=_mock_download_single(no_total_bytes)):
                ingester.fetch()

        assert any("total expenditure" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Test: Expenditure parsing edge cases
# ---------------------------------------------------------------------------

class TestExpenditureParsing:
    """Verify expenditure value parsing handles edge cases."""

    def test_parse_expenditure_with_commas(self, ingester):
        """Values with commas are parsed correctly."""
        assert ingester._parse_expenditure("72,967") == 72967.0

    def test_parse_expenditure_with_dollar_sign(self, ingester):
        """Values with $ are parsed correctly."""
        assert ingester._parse_expenditure("$9,343") == 9343.0

    def test_parse_expenditure_none_returns_none(self, ingester):
        """None input returns None."""
        assert ingester._parse_expenditure(None) is None

    def test_parse_expenditure_nan_returns_none(self, ingester):
        """NaN input returns None."""
        assert ingester._parse_expenditure(float("nan")) is None

    def test_parse_expenditure_non_numeric_returns_none(self, ingester):
        """Non-numeric text returns None."""
        assert ingester._parse_expenditure("n.a.") is None
        assert ingester._parse_expenditure("") is None
        assert ingester._parse_expenditure("-") is None

    def test_parse_expenditure_plain_number(self, ingester):
        """Plain numeric string is parsed."""
        assert ingester._parse_expenditure("5703.0") == 5703.0
