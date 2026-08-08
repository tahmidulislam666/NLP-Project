"""Train a multilingual TF-IDF model from the supplied Jigsaw and Bangla datasets.

Run: python train_model.py --data-dir data/raw
The live Flask app deliberately keeps a transparent rule layer; use this script as
a reproducible dataset-backed extension or for an API deployment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="models/toxicity_pipeline.joblib")
    parser.add_argument("--max-rows", type=int, default=0, help="Use a smaller sample for quick experiments.")
    args = parser.parse_args()
    data = Path(args.data_dir)

    english = pd.read_csv(data / "train.csv")
    # Turn the six Jigsaw labels into severity classes. This is the only source
    # of severity decisions at runtime; app.py contains no keyword rules.
    harmful_count = english[["toxic", "obscene", "insult", "identity_hate"]].sum(axis=1)
    english["label"] = "safe"
    english.loc[harmful_count > 0, "label"] = "mild"
    english.loc[harmful_count >= 2, "label"] = "moderate"
    # Identity-based hate is never treated as only mild; threat/severe labels
    # below still take precedence and elevate it to severe.
    english.loc[english["identity_hate"] == 1, "label"] = "moderate"
    english.loc[(english["threat"] == 1) | (english["severe_toxic"] == 1), "label"] = "severe"
    en = english[["comment_text", "label"]].rename(columns={"comment_text": "text"})

    # The CSV is UTF-8; errors='replace' keeps one malformed row from stopping training.
    bangla = pd.read_csv(data / "Bengali hate speech .csv", encoding="utf-8", encoding_errors="replace")
    bn = bangla[["sentence", "hate"]].rename(columns={"sentence": "text", "hate": "label"})
    # The Bangla source is binary. Per project policy, every hateful Bangla row
    # receives the highest severity so it triggers the immediate cooldown.
    bn["label"] = bn["label"].map({0: "safe", 1: "severe"})
    dataset = pd.concat([en, bn], ignore_index=True).dropna()
    if args.max_rows:
        per_class = max(1, args.max_rows // dataset["label"].nunique())
        dataset = dataset.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), per_class), random_state=42)
        )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=180_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)),
    ])
    model.fit(dataset["text"], dataset["label"])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    print(f"Saved model trained on {len(dataset):,} messages to {destination}")


if __name__ == "__main__":
    main()
