"""Tests for parser module."""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.parser import (
    load_cp_database,
    prepare_search_corpus,
    _safe_json_parse,
    _build_search_text,
)


class TestSafeJsonParse:
    """Tests for _safe_json_parse function."""

    def test_parse_valid_json(self):
        """Should parse valid JSON string."""
        result = _safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        """Should return empty dict for invalid JSON."""
        result = _safe_json_parse("not json")
        assert result == {}

    def test_parse_already_dict(self):
        """Should return dict unchanged."""
        d = {"key": "value"}
        result = _safe_json_parse(d)
        assert result == d

    def test_parse_nan(self):
        """Should return empty dict for NaN."""
        result = _safe_json_parse(float("nan"))
        assert result == {}

    def test_parse_none(self):
        """Should return empty dict for None."""
        result = _safe_json_parse(None)
        assert result == {}


class TestBuildSearchText:
    """Tests for _build_search_text function."""

    def test_build_with_attributes(self, sample_df):
        """Should include category, brand, name, and attributes."""
        row = sample_df.iloc[0].copy()
        row["attributes_dict"] = {"volume": "300л", "energy_class": "A++"}
        text = _build_search_text(row)
        assert "холодильники" in text
        assert "Bosch" in text
        assert "volume:300л" in text

    def test_build_empty_attributes(self, sample_df):
        """Should work with empty attributes."""
        row = sample_df.iloc[0].copy()
        row["attributes_dict"] = {}
        text = _build_search_text(row)
        assert "холодильники" in text
        assert "Bosch" in text


class TestPrepareSearchCorpus:
    """Tests for prepare_search_corpus function."""

    def test_adds_search_text_column(self, sample_df):
        """Should add search_text column."""
        result = prepare_search_corpus(sample_df)
        assert "search_text" in result.columns

    def test_no_empty_search_texts(self, sample_df):
        """All search_text values should be non-empty."""
        result = prepare_search_corpus(sample_df)
        assert all(len(text) > 0 for text in result["search_text"])


class TestLoadCpDatabase:
    """Tests for load_cp_database function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_cp_database("nonexistent_file.xlsx")

    def test_load_sample_file(self, tmp_path, sample_df):
        """Should load and process a valid Excel file."""
        filepath = tmp_path / "test.xlsx"
        sample_df.to_excel(filepath, index=False)
        result = load_cp_database(str(filepath))
        assert len(result) == len(sample_df)
        assert "attributes_dict" in result.columns
        assert pd.api.types.is_numeric_dtype(result["price"])