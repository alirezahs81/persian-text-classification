"""Lightweight Persian text normalization without external NLP packages.

Designed for CI/production: pure Python + regex, deterministic, fast.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic presentation forms / variant letters → Persian standard
_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ء": "",
        "ٔ": "",
        "ٰ": "",
        "ٓ": "",
        "ً": "",
        "ٌ": "",
        "ٍ": "",
        "َ": "",
        "ُ": "",
        "ِ": "",
        "ّ": "",
        "ْ": "",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)

_ZW_RE = re.compile(r"[\u200c\u200d\u200e\u200f\ufeff]")
_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:\+98|0)?9\d{9}|\d{3,4}-?\d{3,4}-?\d{4}")
_PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06FF]+", re.UNICODE)


def normalize_persian(text: str) -> str:
    """Normalize a single Persian (or mixed) string for classification."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_MAP)
    text = _ZW_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip().lower()
    return text


class PersianTextNormalizer:
    """sklearn-compatible transformer for use inside Pipelines."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_persian(t) for t in X]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
