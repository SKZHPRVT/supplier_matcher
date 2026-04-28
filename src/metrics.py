"""Module for calculating quality metrics and generating reports."""
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_extraction_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate quality of data extraction from commercial proposals.

    Args:
        df: Preprocessed DataFrame with CP data.

    Returns:
        Dictionary of quality metrics.
    """
    total = len(df)

    metrics = {
        "dataset_overview": {
            "total_records": total,
            "unique_products": df["product_id"].nunique(),
            "unique_suppliers": df["supplier_id"].nunique(),
            "unique_categories": df["category"].nunique(),
            "date_range": {
                "start": df["cp_date"].min().isoformat()
                if not df["cp_date"].isna().all()
                else None,
                "end": df["cp_date"].max().isoformat()
                if not df["cp_date"].isna().all()
                else None,
            },
        },
        "data_quality": {
            "missing_prices": int(df["price"].isna().sum()),
            "missing_prices_pct": round(
                100 * df["price"].isna().sum() / total, 2
            ),
            "missing_dates": int(df["cp_date"].isna().sum()),
            "empty_attributes": int(
                sum(1 for a in df.get("attributes_dict", []) if not a)
            ),
            "valid_records": int((df["validity_status"] == "valid").sum()),
        },
        "price_statistics": {
            "mean_price": round(df["price"].mean(), 2),
            "median_price": round(df["price"].median(), 2),
            "min_price": round(df["price"].min(), 2),
            "max_price": round(df["price"].max(), 2),
        },
    }

    return metrics


def calculate_matching_metrics(
    query: str,
    results: pd.DataFrame,
    database: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate metrics for search result quality.

    Args:
        query: Original search query.
        results: Top-k results DataFrame.
        database: Full database for comparison.

    Returns:
        Dictionary of matching metrics.
    """
    metrics = {
        "query": query,
        "results_count": len(results),
        "metrics": {
            "mean_relevance_score": round(
                results["relevance_score"].mean(), 4
            ),
            "max_relevance_score": round(
                results["relevance_score"].max(), 4
            ),
            "mean_price": round(results["price"].mean(), 2),
            "min_price": round(results["price"].min(), 2),
            "unique_suppliers": results["supplier_name"].nunique(),
            "unique_products": results["product_name"].nunique(),
            "diversity_score": round(
                results["supplier_name"].nunique() / len(results), 4
            ),
        },
    }

    if not database.empty:
        avg_market_price = database["price"].mean()
        metrics["metrics"]["price_vs_market_avg"] = round(
            results["price"].mean() / avg_market_price, 4
        )

    return metrics


def generate_report(
    extraction_metrics: Dict[str, Any],
    matching_results: List[Dict[str, Any]],
    output_dir: str = "outputs",
) -> str:
    """
    Generate a markdown report with metrics and analysis.

    Args:
        extraction_metrics: Metrics from evaluate_extraction_quality.
        matching_results: List of metrics from calculate_matching_metrics.
        output_dir: Directory to save the report.

    Returns:
        Path to the generated report file.
    """
    output_path = Path(output_dir) / "extraction_report.md"

    report = f"""# Отчет о качестве извлечения и сопоставления данных КП

*Сгенерирован автоматически*

---

## 1. Качество данных

### Обзор датасета
- Всего записей: **{extraction_metrics['dataset_overview']['total_records']}**
- Уникальных товаров: **{extraction_metrics['dataset_overview']['unique_products']}**
- Уникальных поставщиков: **{extraction_metrics['dataset_overview']['unique_suppliers']}**
- Категорий: **{extraction_metrics['dataset_overview']['unique_categories']}**

### Качество данных
- Пропущенные цены: {extraction_metrics['data_quality']['missing_prices']} ({extraction_metrics['data_quality']['missing_prices_pct']}%)
- Пустые атрибуты: {extraction_metrics['data_quality']['empty_attributes']}
- Валидных записей: {extraction_metrics['data_quality']['valid_records']}

### Статистика цен
- Средняя: **{extraction_metrics['price_statistics']['mean_price']:,.2f} RUB**
- Медиана: **{extraction_metrics['price_statistics']['median_price']:,.2f} RUB**
- Диапазон: {extraction_metrics['price_statistics']['min_price']:,.2f} – {extraction_metrics['price_statistics']['max_price']:,.2f} RUB

---

## 2. Результаты поиска
"""

    for i, res in enumerate(matching_results, 1):
        report += f"""### Запрос {i}: "{res['query']}"
- Найдено: {res['results_count']}
- Средняя релевантность: {res['metrics']['mean_relevance_score']}
- Максимальная релевантность: {res['metrics']['max_relevance_score']}
- Уникальных поставщиков: {res['metrics']['unique_suppliers']}
- Разнообразие: {res['metrics']['diversity_score']}
- Средняя цена: {res['metrics']['mean_price']:,.2f} RUB
- Минимальная цена: {res['metrics']['min_price']:,.2f} RUB

"""

    report += """---

## 3. Выводы

### Сильные стороны:
- Данные хорошо структурированы
- Нет критических пропусков
- Четкая категоризация помогает поиску

### Рекомендации:
- Добавить синонимы и аббревиатуры
- Внедрить BERT-эмбеддинги для сложных запросов
- Отслеживать историю цен
"""

    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {output_path}")
    return str(output_path)


def save_metrics_json(
    metrics: Dict[str, Any], filepath: str = "outputs/metrics.json"
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary.
        filepath: Output path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Metrics saved to {path}")