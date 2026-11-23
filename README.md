# Persian Text Classification

[![CI](https://github.com/alirezahs81/persian-text-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/persian-text-classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Production-style Persian (Farsi) intent / complaint classification** for support-desk text.

Leakage-free `sklearn` pipeline · character n-gram TF-IDF · stratified CV model selection · FastAPI · Docker · pytest + GitHub Actions.

> Sample support-ticket dataset (billing, technical, sales, cancellation, other) is generated in-repo — no third-party copyrighted text required.

---

## Features

| Area | Detail |
|------|--------|
| **Normalization** | Arabic→Persian letters, digits, diacritics, ZWJ, URL/email/phone scrubbing |
| **Features** | `char_wb` TF-IDF (2–5), strong for morphology-rich Persian without a stemmer |
| **Models** | Logistic Regression & LinearSVC, `class_weight=balanced` |
| **Selection** | Stratified 5-fold **macro-F1** on train only; test held out |
| **Serving** | FastAPI `/predict`, `/predict/batch`, `/health` |
| **Ops** | Docker image trains at build time; CI on 3.11 & 3.12 |

### Labels

| ID | Meaning |
|----|---------|
| `billing` | Billing and payments |
| `technical` | Technical issues |
| `sales` | Sales and plans |
| `cancellation` | Service cancellation |
| `other` | Other |

---

## Project layout

```text
persian-text-classification/
├── src/
│   ├── config.py          # paths, labels, train defaults
│   ├── preprocess.py      # Persian normalizer (no heavy NLP deps)
│   ├── data.py            # curated dataset builder
│   ├── train.py           # CV selection + persist pipeline
│   ├── predict.py         # inference + confidence
│   └── api.py             # FastAPI app
├── tests/
├── main.py                # train + demo predictions
├── Dockerfile
└── requirements.txt
```

---

## Quick start

```bash
git clone https://github.com/alirezahs81/persian-text-classification.git
cd persian-text-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                 # build data, train, demo
pytest tests/ -v
uvicorn src.api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

```bash
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Internet has been down since morning and the modem will not connect"}'
```

### Docker

```bash
docker build -t persian-text-api .
docker run --rm -p 8000:8000 persian-text-api
```

---

## Methodology

1. **Normalize** text with a deterministic pure-Python normalizer (no hazm / GPU required for CI).
2. **Split** stratified train/test before any vectorizer fitting.
3. **Select** between LogReg and LinearSVC via stratified CV macro-F1 on the **train** split only.
4. **Refit** the winner on full train; report accuracy / macro-F1 / weighted-F1 on the held-out test set.
5. **Persist** a single `joblib` pipeline (`norm → tfidf → clf`) for identical train/serve preprocessing.

On the bundled synthetic ticket set, held-out macro-F1 is typically **≥ 0.90** (exact number is written to `models/metrics.json` after training).

---

## Design notes

- **Why char n-grams?** Persian morphology and mixed Arabic/Persian orthography are noisy; character n-grams are robust without a large pretrained model.
- **Why not transformers in CI?** Keeps installs small and deterministic. The pipeline boundary is clean if you later swap the vectorizer for a frozen encoder.
- **Data** is template-expanded support language for demos and regression tests — replace `data/raw/persian_complaints.csv` with your own labeled tickets for production.

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
