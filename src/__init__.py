"""
Supplier Matcher - AI-powered supplier matching system.
"""

__version__ = "0.1.0"

from src.matcher import find_top5_suppliers
from src.parser import load_cp_database, prepare_search_corpus
from src.metrics import evaluate_extraction_quality, calculate_matching_metrics