"""
src/train.py
Trains a Logistic Regression clickbait classifier.
Tracks experiments with MLflow. No GPU / heavy models needed.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from preprocess import load_and_clean, save_processed
from features import build_feature_pipeline, save_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH       = "data/clickbait_data.csv"
PROCESSED_PATH  = "data/processed_data.csv"
MODEL_PATH      = "models/classifier.pkl"
PIPELINE_PATH   = "models/feature_pipeline.pkl"
METRICS_PATH    = "models/metrics.json"

PARAMS = {
    "test_size":       0.2,
    "random_state":    42,
    "C":               1.0,
    "max_iter":        1000,
    "solver":          "lbfgs",
    "max_features":    2500,
    "ngram_range":     (1, 2),
}


def train(params: dict = PARAMS) -> dict:
    mlflow.set_experiment("clickbait-detection")

    with mlflow.start_run():
        # ── 1. Load & preprocess ────────────────────────────────────────────
        df = load_and_clean(DATA_PATH)
        save_processed(df, PROCESSED_PATH)

        X_text = df["clean_headline"].values
        y      = df["label"].values

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y,
            test_size=params["test_size"],
            random_state=params["random_state"],
            stratify=y,
        )

        # ── 2. Features ─────────────────────────────────────────────────────
        feature_pipeline = build_feature_pipeline(
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
        )
        X_train = feature_pipeline.fit_transform(X_train_text)
        X_test  = feature_pipeline.transform(X_test_text)
        save_features(feature_pipeline, PIPELINE_PATH)

        # ── 3. Train ────────────────────────────────────────────────────────
        clf = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            random_state=params["random_state"],
        )
        clf.fit(X_train, y_train)

        # ── 4. Evaluate ─────────────────────────────────────────────────────
        y_pred = clf.predict(X_test)
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        }

        # Cross-val for robustness estimate
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        metrics["cv_mean_accuracy"] = round(cv_scores.mean(), 4)
        metrics["cv_std_accuracy"]  = round(cv_scores.std(), 4)

        logger.info("\n" + classification_report(y_test, y_pred, target_names=["Not Clickbait", "Clickbait"]))
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        # ── 5. Log to MLflow ────────────────────────────────────────────────
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, "model")

        # ── 6. Save artefacts ───────────────────────────────────────────────
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics: {metrics}")
        logger.info(f"Model saved to {MODEL_PATH}")
        return metrics


if __name__ == "__main__":
    metrics = train()
    print("\n=== Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
