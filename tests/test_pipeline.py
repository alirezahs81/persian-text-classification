from __future__ import annotations

import pytest

from src.config import LABEL_COLUMN, LABELS, PIPELINE_FILE, TEXT_COLUMN
from src.data import build_dataset
from src.predict import TextClassifier
from src.train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    result = train(persist=True, cv_folds=3)
    return result


def test_dataset_schema_and_balance():
    df = build_dataset()
    assert TEXT_COLUMN in df.columns
    assert LABEL_COLUMN in df.columns
    assert not df.empty
    assert set(df[LABEL_COLUMN].unique()) == set(LABELS)
    counts = df[LABEL_COLUMN].value_counts()
    assert counts.min() >= 10


def test_train_metrics(trained):
    m = trained["metrics"]
    assert m["best_model"] in {"logreg", "linearsvc"}
    assert m["test_macro_f1"] >= 0.85
    assert m["test_accuracy"] >= 0.85
    assert PIPELINE_FILE.exists()


def test_predict_labels(trained):
    clf = TextClassifier()
    samples = {
        "billing": "مبلغ فاکتور اشتباه است و دوبار پول کم شده",
        "technical": "اینترنت قطع شده و مودم وصل نمی‌شود",
        "sales": "میخوام پلن بالاتر بخرم",
        "cancellation": "درخواست لغو اشتراک دارم",
        "other": "ساعت کار پشتیبانی را بگویید",
    }
    for expected, text in samples.items():
        out = clf.predict_one(text)
        assert out["label"] == expected
        assert 0.0 <= out["confidence"] <= 1.0
        assert out["label_name"]


def test_predict_many(trained):
    clf = TextClassifier()
    outs = clf.predict_many(["قطعی اینترنت", "لغو سرویس"])
    assert len(outs) == 2
