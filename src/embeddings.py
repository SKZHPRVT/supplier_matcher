"""Module for semantic text matching using TF-IDF and cosine similarity."""
import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Russian stop words for better text processing
RUSSIAN_STOP_WORDS = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы",
    "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может",
    "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",
    "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том",
    "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между",
]


class ProductMatcher:
    """
    Semantic product matcher using TF-IDF vectorization.

    Finds similar products based on text queries combining:
    - TF-IDF text similarity with Russian stop words
    - Category match bonus
    - Brand match bonus
    - Attribute keyword matching
    """

    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
        text_weight: float = 0.4,
        category_weight: float = 0.3,
        brand_weight: float = 0.2,
        attribute_weight: float = 0.1,
    ):
        """
        Initialize ProductMatcher with configurable parameters.

        Args:
            max_features: Maximum number of TF-IDF features.
            ngram_range: Range of n-grams to consider.
            text_weight: Weight for text similarity score.
            category_weight: Weight for category match bonus.
            brand_weight: Weight for brand match bonus.
            attribute_weight: Weight for attribute keyword match bonus.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.text_weight = text_weight
        self.category_weight = category_weight
        self.brand_weight = brand_weight
        self.attribute_weight = attribute_weight

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=RUSSIAN_STOP_WORDS,
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )

        self.is_fitted = False
        self.text_matrix: Optional[np.ndarray] = None

    def fit(self, documents: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on document corpus.

        Args:
            documents: List of text documents for training.
        """
        logger.info(f"Fitting TF-IDF on {len(documents)} documents")
        self.text_matrix = self.vectorizer.fit_transform(documents)
        self.is_fitted = True
        logger.info(f"Created vocabulary of size {len(self.vectorizer.vocabulary_)}")

    def find_similar(
        self,
        query: str,
        categories: np.ndarray,
        brands: np.ndarray,
        attributes: List[dict],
        top_k: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find top-k most similar products to the query.

        Args:
            query: Search query text.
            categories: Array of categories for each document.
            brands: Array of brands for each document.
            attributes: List of attribute dicts for each document.
            top_k: Number of top results to return.

        Returns:
            Tuple of (indices, scores) sorted by relevance descending.

        Raises:
            RuntimeError: If called before fitting the model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling find_similar")

        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        text_similarities = cosine_similarity(query_vec, self.text_matrix)[0]
        query_lower = query.lower()

        # Extract query keywords (words with digits and units)
        query_keywords = self._extract_keywords(query_lower)

        # Calculate bonuses
        category_bonus = np.array(
            [1.0 if cat.lower() in query_lower else 0.0 for cat in categories]
        )

        brand_bonus = np.array(
            [1.0 if brand.lower() in query_lower else 0.0 for brand in brands]
        )

        # Attribute matching: check if query keywords appear in attributes
        attribute_bonus = np.zeros(len(categories))
        for i, attr_dict in enumerate(attributes):
            if attr_dict:
                attr_text = " ".join(
                    f"{k} {v}" for k, v in attr_dict.items()
                ).lower()
                matches = sum(
                    1 for kw in query_keywords if kw.lower() in attr_text
                )
                attribute_bonus[i] = min(matches / max(len(query_keywords), 1), 1.0)

        # Combine scores
        final_scores = (
            self.text_weight * text_similarities
            + self.category_weight * category_bonus
            + self.brand_weight * brand_bonus
            + self.attribute_weight * attribute_bonus
        )

        # Get top-k indices
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        return top_indices, final_scores

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from query (numbers with units, key terms).

        Args:
            text: Lowercase query text.

        Returns:
            List of extracted keywords.
        """
        import re

        keywords = []

        # Find patterns like "300л", "8кг", "12000BTU", "A++", "35м²"
        unit_patterns = re.findall(r"\d+\s*(?:л|кг|btu|w|м²|см|мин|ч|a\+\+?)", text)
        keywords.extend(unit_patterns)

        # Find dimensions like "60см"
        dim_patterns = re.findall(r"\d+\s*(?:см|мм|м)", text)
        keywords.extend(dim_patterns)

        # Find energy classes
        energy_patterns = re.findall(r"a\+{1,2}", text)
        keywords.extend(energy_patterns)

        return keywords