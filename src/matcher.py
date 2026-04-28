"""Module for supplier matching and ranking logic."""
import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from src.embeddings import ProductMatcher
from src.parser import prepare_search_corpus

logger = logging.getLogger(__name__)

# Минимальный порог релевантности для включения в выдачу
MIN_RELEVANCE_THRESHOLD = 0.05


def find_top5_suppliers(
    request_text: str,
    cp_database: pd.DataFrame,
    matcher: Optional[ProductMatcher] = None,
) -> pd.DataFrame:
    """
    Find top-5 suppliers for a given product request.

    Performs semantic matching of the request against the product database,
    then ranks suppliers by relevance and price.

    Args:
        request_text: Natural language product request.
        cp_database: Preprocessed DataFrame with CP data.
        matcher: Optional pre-fitted ProductMatcher instance.

    Returns:
        DataFrame with top-5 suppliers containing:
        - supplier_name, product_name, category, brand
        - price, relevance_score, rank_score, cp_date, currency
    """
    _validate_database(cp_database)

    # Prepare search corpus if needed
    if "search_text" not in cp_database.columns:
        cp_database = prepare_search_corpus(cp_database)

    # Fit matcher
    if matcher is None:
        matcher = ProductMatcher()
        matcher.fit(cp_database["search_text"].tolist())

    # Find similar products
    logger.info(f"Searching for: '{request_text}'")
    top_indices, scores = matcher.find_similar(
        request_text,
        cp_database["category"].values,
        cp_database["brand"].values,
        cp_database["attributes_dict"].tolist(),
        top_k=20,
    )

    # Build results
    results = cp_database.iloc[top_indices].copy()
    results["relevance_score"] = scores[top_indices]

    # Filter out results below minimum relevance
    results = results[results["relevance_score"] >= MIN_RELEVANCE_THRESHOLD]

    if results.empty:
        logger.warning(f"No results above relevance threshold for: '{request_text}'")
        return pd.DataFrame()

    # Group by supplier, take best price
    grouped = (
        results.groupby(
            ["supplier_id", "supplier_name", "product_id", "product_name"],
            as_index=False,
        )
        .agg(
            price=("price", "min"),
            relevance_score=("relevance_score", "max"),
            cp_date=("cp_date", "max"),
            category=("category", "first"),
            brand=("brand", "first"),
        )
    )

    # Calculate final rank score
    grouped["rank_score"] = _calculate_rank_score(
        grouped["relevance_score"], grouped["price"]
    )

    # Top-5
    top5 = grouped.nlargest(5, "rank_score")

    # Format output
    result_df = top5[
        [
            "supplier_name",
            "product_name",
            "category",
            "brand",
            "price",
            "relevance_score",
            "rank_score",
            "cp_date",
        ]
    ].copy()

    result_df["relevance_score"] = result_df["relevance_score"].round(4)
    result_df["rank_score"] = result_df["rank_score"].round(4)
    result_df["currency"] = "RUB"

    logger.info(f"Found {len(result_df)} suppliers")
    return result_df


def _validate_database(df: pd.DataFrame) -> None:
    """Validate required columns exist."""
    required = [
        "product_id",
        "product_name",
        "category",
        "brand",
        "price",
        "supplier_name",
        "supplier_id",
        "cp_date",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _calculate_rank_score(
    relevance: pd.Series,
    price: pd.Series,
    price_penalty_weight: float = 0.3,
) -> pd.Series:
    """
    Calculate ranking score: higher relevance + lower price = better.

    Args:
        relevance: Relevance scores.
        price: Product prices.
        price_penalty_weight: Weight for price penalty.

    Returns:
        Rank scores.
    """
    price_min = price.min()
    price_max = price.max()

    if price_max == price_min:
        return relevance

    price_norm = (price - price_min) / (price_max - price_min)
    return relevance * (1 - price_penalty_weight * price_norm)