"""Tests for matcher module."""
import pytest
import pandas as pd
import numpy as np

from src.matcher import (
    find_top5_suppliers,
    _validate_database,
    _calculate_rank_score,
)


class TestValidateDatabase:
    """Tests for _validate_database function."""

    def test_valid_database(self, sample_df):
        """Should not raise for valid database."""
        _validate_database(sample_df)

    def test_missing_columns(self):
        """Should raise ValueError for missing columns."""
        df = pd.DataFrame({"only_one_column": [1, 2, 3]})
        with pytest.raises(ValueError):
            _validate_database(df)


class TestCalculateRankScore:
    """Tests for _calculate_rank_score function."""

    def test_higher_relevance_gives_higher_score(self):
        """Higher relevance should give higher rank score."""
        relevance = pd.Series([0.5, 0.9])
        price = pd.Series([100, 100])
        scores = _calculate_rank_score(relevance, price)
        assert scores.iloc[1] > scores.iloc[0]

    def test_lower_price_gives_higher_score(self):
        """Lower price should give higher rank score at same relevance."""
        relevance = pd.Series([0.8, 0.8])
        price = pd.Series([500, 100])
        scores = _calculate_rank_score(relevance, price)
        assert scores.iloc[1] > scores.iloc[0]

    def test_scores_in_valid_range(self):
        """Scores should be between 0 and 1."""
        relevance = pd.Series([0.0, 0.5, 1.0])
        price = pd.Series([10, 100, 1000])
        scores = _calculate_rank_score(relevance, price)
        assert all(0 <= s <= 1 for s in scores)


class TestFindTop5Suppliers:
    """Tests for find_top5_suppliers function."""

    @pytest.fixture
    def database(self, sample_df_with_search_text):
        """Prepared database with search_text."""
        return sample_df_with_search_text

    def test_returns_dataframe(self, database):
        """Should return a DataFrame."""
        result = find_top5_suppliers("холодильник Bosch", database)
        assert isinstance(result, pd.DataFrame)

    def test_max_five_results(self, database):
        """Should return at most 5 results."""
        result = find_top5_suppliers("чайник", database)
        assert len(result) <= 5

    def test_required_columns(self, database):
        """Should contain all required output columns."""
        result = find_top5_suppliers("холодильник", database)
        required = [
            "supplier_name",
            "product_name",
            "category",
            "brand",
            "price",
            "relevance_score",
            "rank_score",
        ]
        for col in required:
            assert col in result.columns

    def test_relevance_scores_between_zero_and_one(self, database):
        """Relevance scores should be between 0 and 1."""
        result = find_top5_suppliers("холодильник", database)
        assert all(0 <= s <= 1 for s in result["relevance_score"])

    def test_sorted_by_rank_score(self, database):
        """Results should be sorted by rank_score descending."""
        result = find_top5_suppliers("микроволновка", database)
        rank_scores = result["rank_score"].values
        for i in range(len(rank_scores) - 1):
            assert rank_scores[i] >= rank_scores[i + 1]

    def test_empty_query_handled(self, database):
        """Should handle empty query gracefully."""
        result = find_top5_suppliers("", database)
        assert isinstance(result, pd.DataFrame)

    def test_relevant_results_for_specific_query(self, database):
        """Should find relevant results for specific query."""
        result = find_top5_suppliers("Bosch", database)
        bosch_count = result["brand"].str.contains("Bosch").sum()
        assert bosch_count > 0