"""Module for loading and preprocessing commercial proposal data."""
import json
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_cp_database(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess commercial proposal database from Excel file.

    Args:
        filepath: Path to the Excel file with CP data.

    Returns:
        Preprocessed DataFrame with parsed attributes and validated fields.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        ValueError: If required columns are missing.
    """
    logger.info(f"Loading CP database from {filepath}")

    required_columns = [
        "product_id", "product_name", "category", "brand",
        "attributes", "price", "currency", "supplier_name",
        "supplier_id", "cp_date", "cp_file_name", "validity_status",
    ]

    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["attributes_dict"] = df["attributes"].apply(_safe_json_parse)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["cp_date"] = pd.to_datetime(df["cp_date"], errors="coerce")

    missing_prices = df["price"].isna().sum()
    if missing_prices > 0:
        logger.warning(f"Found {missing_prices} rows with invalid prices")

    return df


def _safe_json_parse(json_str: Any) -> Dict[str, Any]:
    """
    Safely parse JSON string to dictionary.

    Args:
        json_str: JSON string or already parsed dict.

    Returns:
        Parsed dictionary or empty dict on failure.
    """
    if isinstance(json_str, dict):
        return json_str
    if pd.isna(json_str) or not isinstance(json_str, str):
        return {}
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {str(json_str)[:100]}... Error: {e}")
        return {}


def prepare_search_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare text corpus for semantic search by combining relevant fields.

    Args:
        df: DataFrame with loaded and preprocessed CP data.

    Returns:
        DataFrame with added 'search_text' column.
    """
    df = df.copy()

    df["search_text"] = df.apply(_build_search_text, axis=1)
    logger.info(f"Prepared search corpus with {len(df)} documents")
    return df


def _build_search_text(row: pd.Series) -> str:
    """
    Build searchable text from row fields.

    Args:
        row: DataFrame row with product information.

    Returns:
        Concatenated text string for semantic search.
    """
    parts = [
        str(row.get("category", "")),
        str(row.get("brand", "")),
        str(row.get("product_name", "")),
    ]

    attrs = row.get("attributes_dict", {})
    if attrs:
        attr_text = " ".join(f"{k}:{v}" for k, v in attrs.items())
        parts.append(attr_text)

    return " ".join(part for part in parts if part and part != "nan")