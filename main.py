#!/usr/bin/env python3
"""Main entry point for Supplier Matcher pipeline."""
import sys
import logging
from pathlib import Path
import pandas as pd
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.parser import load_cp_database, prepare_search_corpus
from src.matcher import find_top5_suppliers
from src.metrics import (
    evaluate_extraction_quality,
    calculate_matching_metrics,
    generate_report,
    save_metrics_json,
)
from src.visualization import (
    create_price_distribution,
    create_supplier_comparison,
    create_top5_visualization,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def process_single_query(df: pd.DataFrame, query: str, output_dir: str = "outputs"):
    """
    Process a single query and save results.

    Args:
        df: Preprocessed database.
        query: Search query.
        output_dir: Output directory.

    Returns:
        Metrics dictionary or None if no results.
    """
    logger.info(f"Query: '{query}'")
    results = find_top5_suppliers(query, df)

    if results.empty:
        logger.warning(f"No results found for: '{query}'")
        return None

    print(f"\nТоп-5 поставщиков для запроса: '{query}'")
    print(results.to_string(index=False))

    # Save results
    safe_query = query.replace(" ", "_")[:50]
    results.to_csv(f"{output_dir}/top5_{safe_query}.csv", index=False)
    logger.info(f"Saved to {output_dir}/top5_{safe_query}.csv")

    # Visualization
    create_top5_visualization(results, query, f"{output_dir}/plots/top5_{safe_query}.png")

    # Metrics
    metrics = calculate_matching_metrics(query, results, df)
    return metrics


def main():
    """Run the full pipeline: load, search, evaluate, visualize."""
    data_path = Path("data/cp_archive_sample.xlsx")

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Place cp_archive_sample.xlsx in data/ folder")
        sys.exit(1)

    # 1. Load data
    logger.info("=" * 50)
    logger.info("Step 1: Loading database...")
    df = load_cp_database(str(data_path))
    df = prepare_search_corpus(df)
    logger.info(f"Loaded {len(df)} records")

    # 2. Evaluate extraction quality
    logger.info("=" * 50)
    logger.info("Step 2: Evaluating extraction quality...")
    extraction_metrics = evaluate_extraction_quality(df)
    save_metrics_json(extraction_metrics, "outputs/metrics.json")
    logger.info("Metrics saved to outputs/metrics.json")

    # 3. Process queries
    test_queries = [
        "холодильник Bosch серебристый 300л A++",
        "кондиционер инверторный 35 квадратов",
        "стиральная машина 8кг белая",
        "пылесос беспроводной",
        "микроволновка с грилем",
    ]

    logger.info("=" * 50)
    logger.info("Step 3: Running test queries...")
    all_results = []
    all_suppliers = []

    for query in test_queries:
        metrics = process_single_query(df, query)
        if metrics:
            all_results.append(metrics)
            # Also collect for consolidated output
            results = find_top5_suppliers(query, df)
            if not results.empty:
                results["query"] = query
                all_suppliers.append(results)

    # Save consolidated top5_suppliers.csv
    if all_suppliers:
        consolidated = pd.concat(all_suppliers, ignore_index=True)
        consolidated.to_csv("outputs/top5_suppliers.csv", index=False)
        logger.info("Saved consolidated outputs/top5_suppliers.csv")

    # 4. Generate report
    logger.info("=" * 50)
    logger.info("Step 4: Generating report...")
    report_path = generate_report(extraction_metrics, all_results)
    logger.info(f"Report: {report_path}")

    # 5. Visualizations
    logger.info("=" * 50)
    logger.info("Step 5: Creating visualizations...")
    create_price_distribution(df)
    create_supplier_comparison(df)
    logger.info("Plots saved to outputs/plots/")

    # Save full metrics
    full_metrics = {
        "extraction_quality": extraction_metrics,
        "search_results": all_results,
    }
    save_metrics_json(full_metrics, "outputs/metrics.json")

    logger.info("=" * 50)
    logger.info("Pipeline completed successfully!")
    logger.info("Outputs:")
    logger.info("  - outputs/top5_suppliers.csv (consolidated)")
    logger.info("  - outputs/top5_*.csv (per query)")
    logger.info("  - outputs/metrics.json")
    logger.info("  - outputs/extraction_report.md")
    logger.info("  - outputs/plots/*.png")


if __name__ == "__main__":
    main()