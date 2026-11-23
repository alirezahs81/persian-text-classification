"""Load the trained pipeline and run inference."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from src.config import LABEL_EN, PIPELINE_FILE


class TextClassifier:
    def __init__(self, model_path=None):
        path = model_path or PIPELINE_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python -m src.train"
            )
        self.pipeline = joblib.load(path)

    def _decision_confidence(self, texts: list[str]) -> list[float]:
        clf = self.pipeline.named_steps["clf"]
        features = self.pipeline[:-1].transform(texts)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(features)
            return [float(np.max(row)) for row in proba]
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(features)
            if scores.ndim == 1:
                return [float(1 / (1 + np.exp(-abs(s)))) for s in scores]
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
            return [float(np.max(row)) for row in proba]
        return [0.0] * len(texts)

    def predict_one(self, text: str) -> dict[str, Any]:
        label = self.pipeline.predict([text])[0]
        confidence = self._decision_confidence([text])[0]
        return {
            "label": label,
            "label_name": LABEL_EN.get(label, label),
            "confidence": round(confidence, 4),
            "text": text,
        }

    def predict_many(self, texts: list[str]) -> list[dict[str, Any]]:
        labels = self.pipeline.predict(texts)
        confidences = self._decision_confidence(texts)
        return [
            {
                "label": lab,
                "label_name": LABEL_EN.get(lab, lab),
                "confidence": round(conf, 4),
                "text": txt,
            }
            for lab, conf, txt in zip(labels, confidences, texts)
        ]
