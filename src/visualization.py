"""Module for data visualization and plotting."""
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def _save_plot(fig: plt.Figure, output_path: str) -> None:
    """Save plot to file, creating directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {path}")


def create_price_distribution(
    df: pd.DataFrame,
    output_path: str = "outputs/plots/price_distribution.png",
) -> str:
    """Create boxplot of price distribution by category."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(data=df, x="category", y="price", ax=ax)
    ax.set_title("Распределение цен по категориям", fontsize=14)
    ax.set_xlabel("Категория", fontsize=12)
    ax.set_ylabel("Цена (RUB)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save_plot(fig, output_path)
    plt.close(fig)
    return output_path


def create_supplier_comparison(
    df: pd.DataFrame,
    output_path: str = "outputs/plots/supplier_comparison.png",
) -> str:
    """Create bar plots comparing suppliers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    supplier_stats = (
        df.groupby("supplier_name")
        .agg(
            product_count=("product_id", "nunique"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )

    # Products count
    sorted_by_products = supplier_stats.sort_values("product_count", ascending=True)
    ax1.barh(sorted_by_products["supplier_name"], sorted_by_products["product_count"])
    ax1.set_title("Количество товаров у поставщиков")
    ax1.set_xlabel("Уникальных товаров")

    # Average price
    sorted_by_price = supplier_stats.sort_values("avg_price", ascending=True)
    ax2.barh(sorted_by_price["supplier_name"], sorted_by_price["avg_price"])
    ax2.set_title("Средняя цена у поставщиков")
    ax2.set_xlabel("Средняя цена (RUB)")

    plt.tight_layout()
    _save_plot(fig, output_path)
    plt.close(fig)
    return output_path


def create_top5_visualization(
    results: pd.DataFrame,
    query: str,
    output_path: str = "outputs/plots/top5_results.png",
) -> str:
    """Create visualization of top-5 search results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [
        f"{row['supplier_name']}\n{row['product_name'][:30]}..."
        for _, row in results.iterrows()
    ]

    x = range(len(results))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        results["price"],
        width,
        label="Цена (RUB)",
        color="steelblue",
    )
    ax.set_ylabel("Цена (RUB)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)

    ax2 = ax.twinx()
    ax2.bar(
        [i + width / 2 for i in x],
        results["relevance_score"],
        width,
        label="Релевантность",
        color="coral",
    )
    ax2.set_ylabel("Релевантность", color="coral")

    ax.set_title(f'Топ-5 поставщиков по запросу:\n"{query}"', fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    _save_plot(fig, output_path)
    plt.close(fig)
    return output_path