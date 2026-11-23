"""FastAPI inference service for Persian text classification."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import TextClassifier

app = FastAPI(
    title="Persian Text Classification API",
    description="Intent / complaint classification for Persian support texts.",
    version="1.0.0",
)

_classifier: TextClassifier | None = None


def get_classifier() -> TextClassifier:
    global _classifier
    if _classifier is None:
        try:
            _classifier = TextClassifier()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    return _classifier


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["اینترنت از صبح قطع است"])


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class PredictionOut(BaseModel):
    label: str
    label_fa: str
    confidence: float
    text: str


@app.get("/health")
def health():
    ready = True
    try:
        get_classifier()
    except HTTPException:
        ready = False
    return {"status": "ok" if ready else "model_missing", "model_ready": ready}


@app.post("/predict", response_model=PredictionOut)
def predict(body: TextIn):
    clf = get_classifier()
    return clf.predict_one(body.text)


@app.post("/predict/batch")
def predict_batch(body: BatchIn):
    clf = get_classifier()
    return {"predictions": clf.predict_many(body.texts)}
