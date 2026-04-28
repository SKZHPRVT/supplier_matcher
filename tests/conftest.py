"""Shared fixtures for all tests."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking the real data structure."""
    import json
    data = {
        "product_id": ["P001", "P002", "P003", "P004", "P005"],
        "product_name": [
            "Холодильник Bosch KGN39XI32R",
            "Стиральная машина LG F2V5GS0W",
            "Микроволновка Samsung ME88SUG",
            "Чайник Bosch TWK7601",
            "Кондиционер Haier AS12NS5ERA-W",
        ],
        "category": [
            "холодильники",
            "стиральные машины",
            "микроволновки",
            "чайники",
            "кондиционеры",
        ],
        "brand": ["Bosch", "LG", "Samsung", "Bosch", "Haier"],
        "attributes": [
            '{"volume": "300л", "energy_class": "A++", "color": "silver"}',
            '{"load": "8кг", "spin": "1200", "color": "white"}',
            '{"power": "900W", "volume": "23л", "grill": true}',
            '{"power": "2400W", "volume": "1.7л", "material": "steel"}',
            '{"power": "12000BTU", "area": "35м²", "inverter": true}',
        ],
        "price": [45990, 32500, 18400, 12900, 54200],
        "currency": ["RUB"] * 5,
        "supplier_name": [
            "ТехноОпт",
            "СнабжениеПро",
            "Электронные Системы",
            "Электронные Системы",
            "БТ-Трейд",
        ],
        "supplier_id": ["S001", "S002", "S003", "S003", "S004"],
        "cp_date": pd.to_datetime(
            ["2026-03-15", "2026-03-18", "2026-03-20", "2026-03-20", "2026-03-22"]
        ),
        "cp_file_name": [
            "kp_tehnika_Q1.zip",
            "offer_march.pdf",
            "cp_electronics.eml",
            "cp_electronics.eml",
            "offer_Q1_hometech.zip",
        ],
        "validity_status": ["valid"] * 5,
    }
    df = pd.DataFrame(data)
    # Parse attributes_dict explicitly
    df["attributes_dict"] = df["attributes"].apply(json.loads)
    return df


@pytest.fixture
def sample_df_with_search_text(sample_df):
    """Sample DataFrame with search_text column added."""
    from src.parser import prepare_search_corpus
    return prepare_search_corpus(sample_df)