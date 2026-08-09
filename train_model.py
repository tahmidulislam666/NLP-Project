"""Train and evaluate a multilingual TF-IDF model from the supplied datasets.

Run: python train_model.py --data-dir data/raw
It saves a deployment model plus separate Stage 1 and Stage 2 held-out reports.
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


def is_bangla(text: object) -> bool:
    """Return whether text contains Bengali-script characters for model routing."""
    return any("\u0980" <= character <= "\u09ff" for character in str(text))


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
    """Predict safety first; only English unsafe text receives severity classification."""
    import numpy as np
    safety_model = bundle["safety_model"]
    severity_model = bundle["severity_model"]
    texts_list = [str(text) for text in texts]
    safety_probabilities = safety_model.predict_proba(texts_list)
    safety_classes = list(safety_model.named_steps["classifier"].classes_)
    unsafe_probability = safety_probabilities[:, safety_classes.index("unsafe")]
    safe_probability = safety_probabilities[:, safety_classes.index("safe")]
    severity_classes = list(severity_model.named_steps["classifier"].classes_)
    classes = ["safe", "mild", "moderate", "severe"]
    probabilities = np.zeros((len(texts_list), len(classes)))
    probabilities[:, 0] = safe_probability
    bangla_messages = [is_bangla(text) for text in texts_list]
    english_indices = [index for index, is_bangla_message in enumerate(bangla_messages) if not is_bangla_message]
    if english_indices:
        english_texts = [texts_list[index] for index in english_indices]
        severity_probabilities = severity_model.predict_proba(english_texts)
        severity_predictions = severity_model.predict(english_texts)
        for probability_index, severity in enumerate(severity_classes):
            probabilities[english_indices, classes.index(str(severity))] = unsafe_probability[english_indices] * severity_probabilities[:, probability_index]
    else:
        severity_predictions = []
    for index, is_bangla_message in enumerate(bangla_messages):
        if is_bangla_message:
            probabilities[index, classes.index("severe")] = unsafe_probability[index]
    predictions = []
    english_prediction_index = 0
    for is_bangla_message, unsafe in zip(bangla_messages, unsafe_probability):
        if unsafe < 0.5:
            predictions.append("safe")
        elif is_bangla_message:
            predictions.append("severe")
        else:
            predictions.append(str(severity_predictions[english_prediction_index]))
        if not is_bangla_message:
            english_prediction_index += 1
    return predictions, probabilities, classes


def evaluation_result(labels_true: pd.Series, predictions, probabilities, classes: list[str]) -> dict:
    """Calculate serializable classification metrics for one unseen dataset."""
    observed = [label for label in classes if label in set(labels_true)]
    result = {
        "rows": int(len(labels_true)),
        "accuracy": accuracy_score(labels_true, predictions),
        "precision_weighted": precision_score(labels_true, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels_true, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels_true, predictions, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels_true, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(labels_true, predictions, labels=classes, output_dict=True, zero_division=0),
    }
    if len(observed) == 2:
        positive_label = "severe" if "severe" in observed else observed[-1]
        positive_index = classes.index(positive_label)
        result["roc_auc"] = roc_auc_score(labels_true == positive_label, probabilities[:, positive_index])
        result["roc_auc_type"] = f"binary ({positive_label} vs. other)"
    elif len(observed) == len(classes):
        # sklearn requires the ROC-AUC labels and probability columns to use
        # alphabetical order; keep the app's severity order elsewhere.
        roc_labels = sorted(classes)
        roc_columns = [classes.index(label) for label in roc_labels]
        result["roc_auc"] = roc_auc_score(
            labels_true,
            probabilities[:, roc_columns],
            labels=roc_labels,
            multi_class="ovr",
            average="weighted",
        )
        result["roc_auc_type"] = "multiclass one-vs-rest weighted"
    else:
        result["roc_auc"] = None
        result["roc_auc_type"] = "not calculated: evaluation set does not contain every model class"
    return result


def evaluate_stage1(safety_model: Pipeline, texts: pd.Series, severity_labels: pd.Series) -> dict:
    """Evaluate the shared first stage as binary safe/unsafe classification."""
    import numpy as np
    labels_true = severity_labels.where(severity_labels == "safe", "unsafe")
    classes = ["safe", "unsafe"]
    probabilities = safety_model.predict_proba(texts)
    model_classes = list(safety_model.named_steps["classifier"].classes_)
    unsafe_probabilities = probabilities[:, model_classes.index("unsafe")]
    safe_probabilities = probabilities[:, model_classes.index("safe")]
    combined_probabilities = np.column_stack((safe_probabilities, unsafe_probabilities))
    predictions = safety_model.predict(texts)
    result = evaluation_result(labels_true, predictions, combined_probabilities, classes)
    result["roc_auc"] = roc_auc_score(labels_true == "unsafe", unsafe_probabilities)
    result["roc_auc_type"] = "binary (unsafe vs. safe)"
    return result


def evaluate_stage2(severity_model: Pipeline, texts: pd.Series, labels_true: pd.Series) -> dict:
    """Evaluate English severity classification on known-unsafe messages only."""
    classes = ["mild", "moderate", "severe"]
    probabilities = severity_model.predict_proba(texts)
    model_classes = list(severity_model.named_steps["classifier"].classes_)
    ordered_probabilities = probabilities[:, [model_classes.index(label) for label in classes]]
    predictions = severity_model.predict(texts)
    return evaluation_result(labels_true, predictions, ordered_probabilities, classes)


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
        stage1_evaluations = {
            "english_jigsaw_official_test": (official_test["text"], official_test["label"]),
            "bangla_holdout": (bn_test["text"], bn_test["label"]),
        }
        stage2_texts = official_test.loc[official_test["label"] != "safe", "text"]
        stage2_labels = official_test.loc[official_test["label"] != "safe", "label"]
        stage2_evaluation_name = "english_jigsaw_official_unsafe_test"
        evaluation_source = "Jigsaw official labeled test set plus a stratified Bangla hold-out"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["text"], dataset["label"], test_size=args.test_size, random_state=42, stratify=dataset["label"]
        )
        english_test_rows = ~X_test.map(is_bangla)
        stage1_evaluations = {
            "english_stratified_holdout": (X_test[english_test_rows], y_test[english_test_rows]),
            "bangla_holdout": (bn_test["text"], bn_test["label"]),
        }
        stage2_texts = X_test[english_test_rows & (y_test != "safe")]
        stage2_labels = y_test[english_test_rows & (y_test != "safe")]
        stage2_evaluation_name = "english_stratified_unsafe_holdout"
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
    # Bangla source labels are binary: each unsafe row is necessarily severe.
    # Do not teach Stage 2 that Bengali script itself means "severe"; it only
    # learns English mild/moderate/severe distinctions.
    english_unsafe_rows = (y_train != "safe") & ~X_train.map(is_bangla)
    severity_model.fit(X_train[english_unsafe_rows], y_train[english_unsafe_rows])
    model_bundle = {
        "format_version": 3,
        "safety_model": safety_model,
        "severity_model": severity_model,
        "description": "Safety stage for all text; English severity stage; Bangla unsafe maps to severe",
    }

    stage1_metrics = {
        "training_rows": int(len(X_train)),
        "evaluation_source": evaluation_source,
        "stage": "Stage 1: safe vs. unsafe for English and Bangla",
        "evaluations": {
            name: evaluate_stage1(safety_model, texts, labels_true)
            for name, (texts, labels_true) in stage1_evaluations.items()
        },
    }
    stage2_metrics = {
        "training_rows": int(english_unsafe_rows.sum()),
        "evaluation_source": evaluation_source,
        "stage": "Stage 2: English mild/moderate/severe on known-unsafe English messages only",
        "evaluations": {
            stage2_evaluation_name: evaluate_stage2(severity_model, stage2_texts, stage2_labels),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, destination)
    stage1_report_path = destination.with_name("stage1_evaluation_metrics.json")
    stage2_report_path = destination.with_name("stage2_evaluation_metrics.json")
    stage1_report_path.write_text(json.dumps(stage1_metrics, indent=2), encoding="utf-8")
    stage2_report_path.write_text(json.dumps(stage2_metrics, indent=2), encoding="utf-8")
    print(f"Saved model trained on {len(X_train):,} messages to {destination}")
    print(f"Evaluation source: {evaluation_source}")
    for stage_name, metrics in (("Stage 1", stage1_metrics), ("Stage 2", stage2_metrics)):
        print(f"{stage_name} evaluation:")
        for name, result in metrics["evaluations"].items():
            print(f"  {name} ({result['rows']:,} messages):")
            print(f"    Accuracy: {result['accuracy']:.3f}")
            print(f"    F1 score (weighted): {result['f1_weighted']:.3f}")
            print(f"    F1 score (macro): {result['f1_macro']:.3f}")
            print(f"    ROC-AUC ({result['roc_auc_type']}): {result['roc_auc']:.3f}" if result["roc_auc"] is not None else f"    ROC-AUC: {result['roc_auc_type']}")
    print(f"Stage 1 report saved to {stage1_report_path}")
    print(f"Stage 2 report saved to {stage2_report_path}")


if __name__ == "__main__":
    main()
