"""Tests for embeddings module."""
import pytest
import numpy as np

from src.embeddings import ProductMatcher


class TestProductMatcher:
    """Tests for ProductMatcher class."""

    @pytest.fixture
    def documents(self):
        return [
            "холодильник Bosch серебристый 300л",
            "стиральная машина LG белая 8кг",
            "микроволновка Samsung с грилем",
            "кондиционер Haier инверторный",
            "чайник Bosch стальной",
        ]

    @pytest.fixture
    def categories(self):
        return ["холодильники", "стиральные машины", "микроволновки", "кондиционеры", "чайники"]

    @pytest.fixture
    def brands(self):
        return ["Bosch", "LG", "Samsung", "Haier", "Bosch"]

    @pytest.fixture
    def attributes(self):
        return [
            {"volume": "300л", "energy_class": "A++", "color": "silver"},
            {"load": "8кг", "spin": "1200", "color": "white"},
            {"power": "900W", "volume": "23л", "grill": "true"},
            {"power": "12000BTU", "area": "35м²", "inverter": "true"},
            {"power": "2400W", "volume": "1.7л", "material": "steel"},
        ]

    @pytest.fixture
    def fitted_matcher(self, documents):
        matcher = ProductMatcher()
        matcher.fit(documents)
        return matcher

    def test_fit_creates_vocabulary(self, documents):
        matcher = ProductMatcher()
        matcher.fit(documents)
        assert matcher.is_fitted
        assert len(matcher.vectorizer.vocabulary_) > 0

    def test_find_similar_returns_correct_shape(self, fitted_matcher, categories, brands, attributes):
        indices, scores = fitted_matcher.find_similar(
            "холодильник Bosch", categories, brands, attributes, top_k=3
        )
        assert len(indices) == 3
        assert len(scores) == len(categories)

    def test_find_similar_before_fit(self, categories, brands, attributes):
        matcher = ProductMatcher()
        with pytest.raises(RuntimeError):
            matcher.find_similar("query", categories, brands, attributes)

    def test_brand_match_bonus(self, fitted_matcher, categories, brands, attributes):
        indices, scores = fitted_matcher.find_similar(
            "холодильник Bosch", categories, brands, attributes, top_k=5
        )
        first_brand = brands[indices[0]]
        assert first_brand == "Bosch"

    def test_attribute_matching(self, fitted_matcher, categories, brands, attributes):
        """Query with specific attributes should match relevant products."""
        indices, scores = fitted_matcher.find_similar(
            "300л серебристый", categories, brands, attributes, top_k=5
        )
        first_attrs = attributes[indices[0]]
        assert "300л" in str(first_attrs) or "silver" in str(first_attrs)

    def test_relevance_scores_range(self, fitted_matcher, categories, brands, attributes):
        """Scores should be between 0 and 1."""
        _, scores = fitted_matcher.find_similar("test", categories, brands, attributes)
        assert all(0 <= s <= 1 for s in scores)