"""
src/predict.py
Load trained model + pipeline and return predictions.
"""

import joblib
import logging
import re
import string

logger = logging.getLogger(__name__)

MODEL_PATH    = "models/classifier.pkl"
PIPELINE_PATH = "models/feature_pipeline.pkl"


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ClickbaitPredictor:
    def __init__(self, model_path: str = MODEL_PATH, pipeline_path: str = PIPELINE_PATH):
        self.model    = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
        logger.info("Model and feature pipeline loaded successfully.")

    def predict(self, headline: str) -> dict:
        cleaned   = clean_text(headline)
        features  = self.pipeline.transform([cleaned])
        label     = int(self.model.predict(features)[0])
        proba     = self.model.predict_proba(features)[0]
        confidence = round(float(max(proba)), 4)

        return {
            "headline":   headline,
            "is_clickbait": bool(label),
            "label":      label,
            "confidence": confidence,
            "label_text": "Clickbait" if label == 1 else "Not Clickbait",
        }

    def predict_batch(self, headlines: list) -> list:
        return [self.predict(h) for h in headlines]


if __name__ == "__main__":
    predictor = ClickbaitPredictor()
    tests = [
        "You Won't Believe What Happened Next",
        "Parliament Passes New Environmental Protection Bill",
        "This One Weird Trick Will Make You Rich",
        "Scientists Discover Water on Mars",
    ]
    for headline in tests:
        result = predictor.predict(headline)
        print(f"[{result['label_text']:15s}] ({result['confidence']:.0%}) — {headline}")
