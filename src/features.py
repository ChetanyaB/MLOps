"""
src/features.py
Extracts lightweight TF-IDF + handcrafted features from headlines.
No heavy dependencies — works on AWS free tier.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import logging

logger = logging.getLogger(__name__)

# ── Clickbait signal words ──────────────────────────────────────────────────
CLICKBAIT_WORDS = {
    "shocking", "unbelievable", "you", "your", "won't", "will", "believe",
    "this", "secret", "trick", "weird", "amazing", "mindblowing", "viral",
    "genius", "doctors", "hate", "truth", "revealed", "exposed", "never",
    "always", "everyone", "nobody", "every", "must", "need", "what",
    "why", "how", "top", "best", "worst", "only", "just", "simple",
    "instantly", "overnight", "guaranteed", "free", "exclusive"
}

NUMBER_PATTERN = re.compile(r"\b\d+\b")
QUESTION_PATTERN = re.compile(r"\?")
EXCLAIM_PATTERN = re.compile(r"!")


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Numeric feature vector derived from linguistic patterns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._featurize(text) for text in X])

    def _featurize(self, text: str) -> list:
        lower = text.lower()
        words = lower.split()
        total_words = max(len(words), 1)

        clickbait_count = sum(1 for w in words if w in CLICKBAIT_WORDS)
        has_number = int(bool(NUMBER_PATTERN.search(text)))
        has_question = int(bool(QUESTION_PATTERN.search(text)))
        has_exclaim = int(bool(EXCLAIM_PATTERN.search(text)))
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        word_count = total_words
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        starts_with_number = int(bool(words and words[0].isdigit()))
        second_person = int("you" in words or "your" in words)
        clickbait_ratio = clickbait_count / total_words

        return [
            clickbait_count,
            has_number,
            has_question,
            has_exclaim,
            caps_ratio,
            word_count,
            avg_word_len,
            starts_with_number,
            second_person,
            clickbait_ratio,
        ]


def build_feature_pipeline(max_features: int = 5000, ngram_range: tuple = (1, 2)):
    """Combine TF-IDF and handcrafted features."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        analyzer="word",
    )

    union = FeatureUnion([
        ("tfidf", tfidf),
        ("handcrafted", HandcraftedFeatures()),
    ])
    return union


def save_features(pipeline, path: str = "models/feature_pipeline.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Saved feature pipeline to {path}")


def load_features(path: str = "models/feature_pipeline.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed_data.csv")
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df["clean_headline"])
    print(f"Feature matrix shape: {X.shape}")
    save_features(pipeline)
