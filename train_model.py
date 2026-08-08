"""Train and evaluate a multilingual TF-IDF model from the supplied datasets.

Run: python train_model.py --data-dir data/raw
It saves both a deployment model and held-out evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


TOXIC_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def add_english_severity_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Map Jigsaw's multi-label annotations to this project's four severities."""
    frame = frame.copy()
    harmful_count = frame[["toxic", "obscene", "insult", "identity_hate"]].sum(axis=1)
    frame["label"] = "safe"
    frame.loc[harmful_count > 0, "label"] = "mild"
    frame.loc[harmful_count >= 2, "label"] = "moderate"
    # Identity-based hate is never treated as only mild; threat/severe labels
    # below still take precedence and elevate it to severe.
    frame.loc[frame["identity_hate"] == 1, "label"] = "moderate"
    frame.loc[(frame["threat"] == 1) | (frame["severe_toxic"] == 1), "label"] = "severe"
    return frame


def load_official_jigsaw_test(data: Path) -> pd.DataFrame | None:
    """Return Jigsaw's independently labeled test rows, excluding -1 unknown labels."""
    test_path = data / "test.csv"
    labels_path = data / "test_labels.csv"
    if not test_path.exists() or not labels_path.exists():
        return None
    test = pd.read_csv(test_path)
    labels = pd.read_csv(labels_path)
    merged = test.merge(labels, on="id", how="inner")
    valid = merged[(merged[TOXIC_COLUMNS] != -1).all(axis=1)].copy()
    valid = add_english_severity_labels(valid)
    return valid[["comment_text", "label"]].rename(columns={"comment_text": "text"}).dropna()


def predict_two_stage(bundle: dict, texts: pd.Series):
    """Predict safe/unsafe first, then severity for unsafe messages only."""
    import numpy as np
    safety_model = bundle["safety_model"]
    severity_model = bundle["severity_model"]
    safety_probabilities = safety_model.predict_proba(texts)
    safety_classes = list(safety_model.named_steps["classifier"].classes_)
    unsafe_probability = safety_probabilities[:, safety_classes.index("unsafe")]
    safe_probability = safety_probabilities[:, safety_classes.index("safe")]
    severity_probabilities = severity_model.predict_proba(texts)
    severity_classes = list(severity_model.named_steps["classifier"].classes_)
    classes = ["safe", "mild", "moderate", "severe"]
    probabilities = np.zeros((len(texts), len(classes)))
    probabilities[:, 0] = safe_probability
    for index, severity in enumerate(severity_classes):
        probabilities[:, classes.index(str(severity))] = unsafe_probability * severity_probabilities[:, index]
    severity_predictions = severity_model.predict(texts)
    predictions = ["safe" if unsafe < 0.5 else str(severity) for unsafe, severity in zip(unsafe_probability, severity_predictions)]
    return predictions, probabilities, classes


def evaluate(bundle: dict, texts: pd.Series, labels_true: pd.Series) -> dict:
    """Calculate serializable metrics for one unseen evaluation dataset."""
    predictions, probabilities, classes = predict_two_stage(bundle, texts)
    observed = [label for label in classes if label in set(labels_true)]
    result = {
        "rows": int(len(texts)),
        "accuracy": accuracy_score(labels_true, predictions),
        "precision_weighted": precision_score(labels_true, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels_true, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels_true, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(labels_true, predictions, labels=classes, output_dict=True, zero_division=0),
    }
    if len(observed) == 2:
        positive_label = "severe" if "severe" in observed else observed[-1]
        positive_index = classes.index(positive_label)
        result["roc_auc"] = roc_auc_score(labels_true == positive_label, probabilities[:, positive_index])
        result["roc_auc_type"] = f"binary ({positive_label} vs. other)"
    elif len(observed) == len(classes):
        result["roc_auc"] = roc_auc_score(labels_true, probabilities, labels=classes, multi_class="ovr", average="weighted")
        result["roc_auc_type"] = "multiclass one-vs-rest weighted"
    else:
        result["roc_auc"] = None
        result["roc_auc_type"] = "not calculated: evaluation set does not contain every model class"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="models/toxicity_pipeline.joblib")
    parser.add_argument("--max-rows", type=int, default=0, help="Use a smaller sample for quick experiments.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of Bangla rows held out for evaluation (default: 0.2).")
    args = parser.parse_args()
    data = Path(args.data_dir)

    english = add_english_severity_labels(pd.read_csv(data / "train.csv"))
    en = english[["comment_text", "label"]].rename(columns={"comment_text": "text"})

    # The CSV is UTF-8; errors='replace' keeps one malformed row from stopping training.
    bangla = pd.read_csv(data / "Bengali hate speech .csv", encoding="utf-8", encoding_errors="replace")
    bn = bangla[["sentence", "hate"]].rename(columns={"sentence": "text", "hate": "label"})
    # The Bangla source is binary. Per project policy, every hateful Bangla row
    # receives the highest severity so it triggers the immediate cooldown.
    bn["label"] = bn["label"].map({0: "safe", 1: "severe"})
    bn = bn.dropna()
    # Reserve unseen Bangla examples so multilingual performance can be reported.
    bn_train, bn_test = train_test_split(bn, test_size=args.test_size, random_state=42, stratify=bn["label"])
    dataset = pd.concat([en, bn_train], ignore_index=True).dropna()
    if args.max_rows:
        per_class = max(1, args.max_rows // dataset["label"].nunique())
        dataset = pd.concat(
            [group.sample(min(len(group), per_class), random_state=42) for _, group in dataset.groupby("label")],
            ignore_index=True,
        )
    official_test = load_official_jigsaw_test(data)
    if official_test is not None:
        # Official competition test labels are independent of training data, so
        # train on all available English and Bangla training messages.
        X_train, y_train = dataset["text"], dataset["label"]
        evaluations = {
            "english_jigsaw_official_test": (official_test["text"], official_test["label"]),
            "bangla_holdout": (bn_test["text"], bn_test["label"]),
        }
        evaluation_source = "Jigsaw official labeled test set plus a stratified Bangla hold-out"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["text"], dataset["label"], test_size=args.test_size, random_state=42, stratify=dataset["label"]
        )
        evaluations = {
            "combined_stratified_holdout": (X_test, y_test),
            "bangla_holdout": (bn_test["text"], bn_test["label"]),
        }
        evaluation_source = "Stratified combined hold-out plus a stratified Bangla hold-out"
    safety_model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=180_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    safety_model.fit(X_train, y_train.where(y_train == "safe", "unsafe"))
    unsafe_rows = y_train != "safe"
    severity_model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=180_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    severity_model.fit(X_train[unsafe_rows], y_train[unsafe_rows])
    model_bundle = {
        "format_version": 2,
        "safety_model": safety_model,
        "severity_model": severity_model,
        "description": "Two-stage safe/unsafe followed by mild/moderate/severe",
    }

    metrics = {
        "training_rows": int(len(X_train)),
        "evaluation_source": evaluation_source,
        "architecture": "Two-stage: safe/unsafe, then mild/moderate/severe for unsafe messages",
        "evaluations": {name: evaluate(model_bundle, texts, labels_true) for name, (texts, labels_true) in evaluations.items()},
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, destination)
    report_path = destination.with_name("evaluation_metrics.json")
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model trained on {len(X_train):,} messages to {destination}")
    print(f"Evaluation source: {evaluation_source}")
    for name, result in metrics["evaluations"].items():
        print(f"{name} ({result['rows']:,} messages):")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Precision (weighted): {result['precision_weighted']:.3f}")
        print(f"  Recall (weighted): {result['recall_weighted']:.3f}")
        print(f"  F1 score (weighted): {result['f1_weighted']:.3f}")
        print(f"  ROC-AUC ({result['roc_auc_type']}): {result['roc_auc']:.3f}" if result["roc_auc"] is not None else f"  ROC-AUC: {result['roc_auc_type']}")
    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
