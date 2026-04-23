"""
app.py
FastAPI inference server for the Clickbait Detector.

Endpoints:
  GET  /health   — liveness / readiness probe
  POST /predict  — single headline prediction
  POST /predict/batch — multiple headlines at once
  GET  /stats    — monitoring stats
  GET  /drift    — drift detection status
"""

import os
import sys
import logging
import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import ClickbaitPredictor
from monitor import log_prediction, get_prediction_stats, check_drift

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clickbait Detector API",
    description="Lightweight NLP classifier — detects clickbait headlines.",
    version="1.0.0",
)

# Load model once at startup
predictor: ClickbaitPredictor = None


@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = ClickbaitPredictor(
            model_path="models/classifier.pkl",
            pipeline_path="models/feature_pipeline.pkl",
        )
        logger.info("Model loaded successfully at startup.")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}. Run 'python src/train.py' first.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    headline: str = Field(..., example="You Won't Believe What Happened Next!", min_length=3)

class PredictResponse(BaseModel):
    headline:     str
    is_clickbait: bool
    label:        int
    label_text:   str
    confidence:   float

class BatchPredictRequest(BaseModel):
    headlines: List[str] = Field(..., min_items=1, max_items=100)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Liveness + readiness check."""
    model_ready = predictor is not None
    return {
        "status":      "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "version":     "1.0.0",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Predict whether a single headline is clickbait."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    result = predictor.predict(request.headline)
    log_prediction(request.headline, result)
    logger.info(f"Prediction: '{request.headline[:60]}' → {result['label_text']} ({result['confidence']:.0%})")
    return result


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    """Predict multiple headlines in one call."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = predictor.predict_batch(request.headlines)
    for h, r in zip(request.headlines, results):
        log_prediction(h, r)
    return {"predictions": results, "count": len(results)}


@app.get("/stats", tags=["Monitoring"])
def stats():
    """Aggregated stats from recent predictions."""
    return get_prediction_stats()


@app.get("/drift", tags=["Monitoring"])
def drift():
    """Data drift detection check."""
    return check_drift()


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
