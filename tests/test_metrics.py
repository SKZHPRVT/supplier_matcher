"""Tests for metrics module."""
import pytest
import json
import pandas as pd
from pathlib import Path

from src.metrics import (
    evaluate_extraction_quality,
    calculate_matching_metrics,
    generate_report,
    save_metrics_json,
)


class TestEvaluateExtractionQuality:
    """Tests for evaluate_extraction_quality function."""

    def test_returns_dict(self, sample_df):
        """Should return a dictionary."""
        result = evaluate_extraction_quality(sample_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_df):
        """Should contain dataset_overview, data_quality, price_statistics."""
        result = evaluate_extraction_quality(sample_df)
        assert "dataset_overview" in result
        assert "data_quality" in result
        assert "price_statistics" in result

    def test_correct_record_count(self, sample_df):
        """Should count records correctly."""
        result = evaluate_extraction_quality(sample_df)
        assert result["dataset_overview"]["total_records"] == len(sample_df)

    def test_no_missing_prices_in_sample(self, sample_df):
        """Sample data should have no missing prices."""
        result = evaluate_extraction_quality(sample_df)
        assert result["data_quality"]["missing_prices"] == 0


class TestCalculateMatchingMetrics:
    """Tests for calculate_matching_metrics function."""

    def test_returns_dict(self, sample_df):
        """Should return a dictionary."""
        results = sample_df.head(3).copy()
        results["relevance_score"] = [0.9, 0.7, 0.5]
        results["rank_score"] = [0.85, 0.65, 0.45]
        metrics = calculate_matching_metrics("test query", results, sample_df)
        assert isinstance(metrics, dict)

    def test_contains_query(self, sample_df):
        """Should include the query text."""
        results = sample_df.head(3).copy()
        results["relevance_score"] = [0.9, 0.7, 0.5]
        results["rank_score"] = [0.85, 0.65, 0.45]
        metrics = calculate_matching_metrics("холодильник", results, sample_df)
        assert metrics["query"] == "холодильник"

    def test_diversity_score_range(self, sample_df):
        """Diversity score should be between 0 and 1."""
        results = sample_df.head(3).copy()
        results["relevance_score"] = [0.9, 0.7, 0.5]
        results["rank_score"] = [0.85, 0.65, 0.45]
        metrics = calculate_matching_metrics("test", results, sample_df)
        diversity = metrics["metrics"]["diversity_score"]
        assert 0 <= diversity <= 1


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_creates_file(self, tmp_path, sample_df):
        """Should create a markdown report file."""
        extraction = evaluate_extraction_quality(sample_df)
        matching = [{
            "query": "test",
            "results_count": 3,
            "metrics": {
                "mean_relevance_score": 0.5,
                "max_relevance_score": 0.9,
                "unique_suppliers": 2,
                "diversity_score": 0.67,
                "mean_price": 15000.0,
                "min_price": 10000.0,
            }
        }]

        path = generate_report(extraction, matching, str(tmp_path))
        assert Path(path).exists()
        assert Path(path).suffix == ".md"

    def test_report_contains_info(self, tmp_path, sample_df):
        """Report should contain key information."""
        extraction = evaluate_extraction_quality(sample_df)
        matching = [{
            "query": "test",
            "results_count": 3,
            "metrics": {
                "mean_relevance_score": 0.5,
                "max_relevance_score": 0.9,
                "unique_suppliers": 2,
                "diversity_score": 0.67,
                "mean_price": 15000.0,
                "min_price": 10000.0,
            }
        }]

        path = generate_report(extraction, matching, str(tmp_path))
        content = Path(path).read_text(encoding="utf-8")
        assert "test" in content.lower()


class TestSaveMetricsJson:
    """Tests for save_metrics_json function."""

    def test_saves_file(self, tmp_path):
        """Should save a JSON file."""
        filepath = tmp_path / "metrics.json"
        metrics = {"key": "value", "number": 42}
        save_metrics_json(metrics, str(filepath))
        assert filepath.exists()

    def test_json_is_valid(self, tmp_path):
        """Saved file should be valid JSON."""
        filepath = tmp_path / "metrics.json"
        metrics = {"key": "value", "list": [1, 2, 3]}
        save_metrics_json(metrics, str(filepath))

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == metrics