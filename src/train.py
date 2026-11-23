"""Train and select a Persian text classification pipeline.

Vectorizer and classifier are fitted on the training split only.
Model selection uses stratified CV on the train split; test is held out.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    CV_FOLDS,
    LABEL_COLUMN,
    LABELS,
    LABELS_FILE,
    METRICS_FILE,
    PIPELINE_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
)
from src.data import build_dataset
from src.preprocess import PersianTextNormalizer


def build_candidates() -> dict[str, Pipeline]:
    shared_tfidf = dict(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
    )
    return {
        "logreg": Pipeline(
            steps=[
                ("norm", PersianTextNormalizer()),
                ("tfidf", TfidfVectorizer(**shared_tfidf)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linearsvc": Pipeline(
            steps=[
                ("norm", PersianTextNormalizer()),
                ("tfidf", TfidfVectorizer(**shared_tfidf)),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train(
    *,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
    persist: bool = True,
) -> dict[str, Any]:
    df = build_dataset()
    X = df[TEXT_COLUMN].tolist()
    y = df[LABEL_COLUMN].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()

    best_name = None
    best_pipe = None
    best_cv = -np.inf
    cv_table: dict[str, float] = {}

    for name, pipe in candidates.items():
        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        mean_score = float(scores.mean())
        cv_table[name] = mean_score
        print(f"{name}: CV macro-F1 = {mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_cv:
            best_cv = mean_score
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None and best_name is not None
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    metrics = {
        "best_model": best_name,
        "cv_macro_f1": round(best_cv, 4),
        "cv_scores": {k: round(v, 4) for k, v in cv_table.items()},
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "test_weighted_f1": round(
            float(f1_score(y_test, y_pred, average="weighted")), 4
        ),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "labels": LABELS,
    }

    print("\nHeld-out test report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Metrics:", metrics)

    if persist:
        PIPELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipe, PIPELINE_FILE)
        METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        LABELS_FILE.write_text(json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved pipeline -> {PIPELINE_FILE}")

    return {"pipeline": best_pipe, "metrics": metrics, "y_test": y_test, "y_pred": y_pred}


if __name__ == "__main__":
    train()
