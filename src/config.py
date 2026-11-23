from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_FILE = RAW_DATA_DIR / "persian_complaints.csv"
PIPELINE_FILE = MODELS_DIR / "text_pipeline.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
LABELS_FILE = MODELS_DIR / "labels.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Intent / complaint categories for support tickets
LABELS = [
    "billing",
    "technical",
    "sales",
    "cancellation",
    "other",
]

LABEL_EN = {
    "billing": "Billing and payments",
    "technical": "Technical issues",
    "sales": "Sales and plans",
    "cancellation": "Service cancellation",
    "other": "Other",
}

# Backward-compatible alias used by predict/API
LABEL_FA = LABEL_EN

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
