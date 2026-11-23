#!/usr/bin/env python3
"""End-to-end demo: build data, train, print sample predictions."""

from __future__ import annotations

from src.predict import TextClassifier
from src.train import train


def main() -> None:
    print("=== Persian Text Classification Pipeline ===\n")
    result = train(persist=True)
    metrics = result["metrics"]
    print("\nSummary metrics:")
    for k, v in metrics.items():
        if k != "labels":
            print(f"  {k}: {v}")

    clf = TextClassifier()
    demos = [
        "صورتحساب این ماه خیلی زیاد آمده لطفا بررسی کنید",
        "مودم روشن است ولی اینترنت نداریم",
        "میخوام پلن را ارتقا بدم",
        "لطفا سرویس را لغو کنید",
        "ساعت کاری پشتیبانی چقدر است؟",
    ]
    print("\nSample predictions:")
    for text in demos:
        out = clf.predict_one(text)
        print(f"  [{out['label']}] ({out['confidence']:.2f}) {text}")

    print("\nDone. API: uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
