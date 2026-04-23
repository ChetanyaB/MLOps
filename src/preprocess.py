"""
src/preprocess.py
Cleans and prepares clickbait dataset for feature extraction.
"""

import re
import string
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load CSV and clean headlines."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    required = {"headline", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    df = df.dropna(subset=["headline", "label"])
    df["label"] = df["label"].astype(int)
    df["clean_headline"] = df["headline"].apply(clean_text)

    logger.info(f"Loaded {len(df)} rows | Clickbait: {df['label'].sum()} | Not: {(df['label'] == 0).sum()}")
    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    df = load_and_clean("data/clickbait_data.csv")
    save_processed(df, "data/processed_data.csv")
