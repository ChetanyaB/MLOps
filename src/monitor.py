"""
src/monitor.py
Simple prediction logging + optional data-drift placeholder.
"""

import os
import json
import logging
import datetime
from collections import deque

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)
PREDICTION_LOG = "logs/predictions.jsonl"

# Rolling window for drift detection placeholder
_recent_predictions = deque(maxlen=1000)


def log_prediction(headline: str, result: dict) -> None:
    """Append a prediction record to the JSONL log file."""
    record = {
        "timestamp":    datetime.datetime.utcnow().isoformat(),
        "headline":     headline,
        "label":        result.get("label"),
        "confidence":   result.get("confidence"),
        "is_clickbait": result.get("is_clickbait"),
    }
    _recent_predictions.append(record)
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_prediction_stats() -> dict:
    """Return basic stats from the rolling window."""
    if not _recent_predictions:
        return {"total": 0, "clickbait_rate": None, "avg_confidence": None}

    total = len(_recent_predictions)
    clickbait_count = sum(1 for r in _recent_predictions if r["label"] == 1)
    avg_conf = sum(r["confidence"] for r in _recent_predictions) / total

    return {
        "total":          total,
        "clickbait_rate": round(clickbait_count / total, 4),
        "avg_confidence": round(avg_conf, 4),
    }


# ── Drift Detection Placeholder ─────────────────────────────────────────────
def check_drift() -> dict:
    """
    Placeholder for production drift detection.

    In production you would:
    1. Compare incoming headline length / vocabulary distribution
       against the training distribution using PSI or KS-test.
    2. Alert if clickbait_rate deviates more than a threshold
       from the training base rate (~0.5 here).

    Returns a dict with drift_detected flag and a message.
    """
    stats = get_prediction_stats()
    if stats["total"] < 50:
        return {"drift_detected": False, "message": "Not enough data to detect drift."}

    base_rate = 0.5  # Expected clickbait rate from training data
    drift = abs(stats["clickbait_rate"] - base_rate) > 0.2

    return {
        "drift_detected": drift,
        "current_rate":   stats["clickbait_rate"],
        "base_rate":      base_rate,
        "message":        "Possible drift detected — review recent predictions." if drift else "No drift detected.",
    }


if __name__ == "__main__":
    print("Monitoring module loaded. Check logs/predictions.jsonl for prediction history.")
    print(get_prediction_stats())
