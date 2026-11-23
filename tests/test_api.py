from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config import PIPELINE_FILE
from src.train import train


@pytest.fixture(scope="module")
def client():
    if not PIPELINE_FILE.exists():
        train(persist=True, cv_folds=3)
    from src.api import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_ready"] is True


def test_predict(client):
    r = client.post("/predict", json={"text": "سرعت اینترنت خیلی کم شده"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {
        "billing",
        "technical",
        "sales",
        "cancellation",
        "other",
    }
    assert "label_fa" in body


def test_predict_batch(client):
    r = client.post(
        "/predict/batch",
        json={"texts": ["پرداخت انجام شد اما بدهی مانده", "می‌خواهم سرویس را لغو کنم"]},
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_empty_rejected(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422
