"""
tests/test_pipeline.py
Unit + integration tests for the clickbait detection pipeline.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import clean_text, load_and_clean
from features import HandcraftedFeatures, build_feature_pipeline
from monitor import log_prediction, get_prediction_stats, check_drift


# ── Preprocessing Tests ───────────────────────────────────────────────────────
class TestPreprocess:
    def test_clean_text_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_clean_text_removes_punctuation(self):
        assert clean_text("Hello, World!") == "hello world"

    def test_clean_text_strips_extra_spaces(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_clean_text_combined(self):
        result = clean_text("You WON'T Believe This!!!")
        assert "!" not in result
        assert result == result.lower()

    def test_load_and_clean_returns_dataframe(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("headline,label\nTest headline,1\nAnother one,0\n")
        df = load_and_clean(str(csv))
        assert isinstance(df, pd.DataFrame)
        assert "clean_headline" in df.columns
        assert len(df) == 2

    def test_load_and_clean_drops_nulls(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("headline,label\nGood headline,1\n,0\nAnother,1\n")
        df = load_and_clean(str(csv))
        assert len(df) == 2  # null row dropped

    def test_load_and_clean_raises_on_missing_column(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("text,label\nSome text,1\n")
        with pytest.raises(ValueError, match="CSV must contain columns"):
            load_and_clean(str(csv))


# ── Feature Tests ─────────────────────────────────────────────────────────────
class TestFeatures:
    def test_handcrafted_features_shape(self):
        hf = HandcraftedFeatures()
        features = hf.transform(["you won't believe this", "new bill passed"])
        assert features.shape == (2, 10)

    def test_handcrafted_clickbait_signals(self):
        hf = HandcraftedFeatures()
        clickbait = hf.transform(["you won't believe this shocking secret"])[0]
        normal    = hf.transform(["parliament passes infrastructure bill"])[0]
        # clickbait_count (index 0) should be higher for clickbait headline
        assert clickbait[0] > normal[0]

    def test_handcrafted_number_detection(self):
        hf = HandcraftedFeatures()
        with_num    = hf.transform(["10 ways to lose weight"])[0]
        without_num = hf.transform(["ways to lose weight"])[0]
        assert with_num[1] == 1   # has_number
        assert without_num[1] == 0

    def test_pipeline_output_is_sparse_or_array(self):
        pipeline = build_feature_pipeline(max_features=100)
        texts    = ["You won't believe this", "Scientists discover new species"]
        X        = pipeline.fit_transform(texts)
        assert X.shape[0] == 2
        assert X.shape[1] > 10  # tfidf + handcrafted combined

    def test_pipeline_transform_consistency(self):
        pipeline = build_feature_pipeline(max_features=100)
        texts    = ["clickbait headline here", "normal news headline"]
        pipeline.fit_transform(texts)
        # Transform single should not raise
        X = pipeline.transform(["new headline"])
        assert X.shape[0] == 1


# ── Monitoring Tests ──────────────────────────────────────────────────────────
class TestMonitoring:
    def test_log_prediction_writes_to_jsonl(self, tmp_path, monkeypatch):
        import monitor
        monkeypatch.setattr(monitor, "PREDICTION_LOG", str(tmp_path / "test.jsonl"))
        monkeypatch.setattr(monitor, "_recent_predictions", __import__("collections").deque(maxlen=1000))

        result = {"label": 1, "confidence": 0.9, "is_clickbait": True}
        log_prediction("Test headline", result)

        with open(str(tmp_path / "test.jsonl")) as f:
            lines = f.readlines()
        assert len(lines) == 1
        import json
        record = json.loads(lines[0])
        assert record["headline"] == "Test headline"
        assert record["label"] == 1

    def test_get_prediction_stats_empty(self, monkeypatch):
        import monitor, collections
        monkeypatch.setattr(monitor, "_recent_predictions", collections.deque(maxlen=1000))
        stats = get_prediction_stats()
        assert stats["total"] == 0

    def test_check_drift_insufficient_data(self, monkeypatch):
        import monitor, collections
        monkeypatch.setattr(monitor, "_recent_predictions", collections.deque(maxlen=1000))
        result = check_drift()
        assert result["drift_detected"] is False
        assert "Not enough" in result["message"]


# ── End-to-End Smoke Test ─────────────────────────────────────────────────────
class TestEndToEnd:
    """Requires trained model artefacts in models/."""

    @pytest.mark.skipif(
        not os.path.exists("models/classifier.pkl"),
        reason="Trained model not found — run src/train.py first",
    )
    def test_predict_returns_expected_keys(self):
        from predict import ClickbaitPredictor
        p = ClickbaitPredictor()
        result = p.predict("You Won't Believe What Happened")
        assert "is_clickbait" in result
        assert "confidence" in result
        assert "label_text" in result
        assert isinstance(result["is_clickbait"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.skipif(
        not os.path.exists("models/classifier.pkl"),
        reason="Trained model not found — run src/train.py first",
    )
    def test_obvious_clickbait_detected(self):
        from predict import ClickbaitPredictor
        p = ClickbaitPredictor()
        result = p.predict("You Won't Believe This Shocking Secret Doctors Hide")
        assert result["is_clickbait"] is True

    @pytest.mark.skipif(
        not os.path.exists("models/classifier.pkl"),
        reason="Trained model not found — run src/train.py first",
    )
    def test_obvious_non_clickbait_detected(self):
        from predict import ClickbaitPredictor
        p = ClickbaitPredictor()
        result = p.predict("Parliament Passes Annual Infrastructure Budget")
        assert result["is_clickbait"] is False
