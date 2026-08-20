"""Train and evaluate Spanish TF-IDF baseline classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "training"
MODEL_DIR = ROOT / "backend" / "models"

# Kept in source to make training reproducible without NLTK downloads.
SPANISH_STOPWORDS = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "dos", "el",
    "ella", "ellas", "ellos", "en", "entre", "era", "eran", "es", "esa", "esas",
    "ese", "eso", "esos", "esta", "estas", "este", "esto", "estos", "fue", "han",
    "hasta", "hay", "la", "las", "le", "les", "lo", "los", "más", "me", "mi",
    "mis", "muy", "no", "nos", "nosotros", "o", "para", "pero", "por", "que",
    "se", "sea", "ser", "si", "sin", "sobre", "son", "su", "sus", "también",
    "te", "tener", "ti", "tiene", "todo", "todos", "tu", "tus", "un", "una",
    "unas", "uno", "unos", "y", "ya", "yo",
]


def load_partition(name: str) -> pd.DataFrame:
    """Load and validate one prepared dataset partition."""
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{name}.csv must contain text and label columns")
    if frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"{name}.csv contains missing values")
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()) <= {0, 1}:
        raise ValueError(f"{name}.csv labels must be 0 (RF) or 1 (RNF)")
    return frame


def build_pipeline(classifier: Any) -> Pipeline:
    """Create a raw-text pipeline with its own fitted vectorizer."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    stop_words=SPANISH_STOPWORDS,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def evaluate(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict[str, Any]:
    """Return binary and macro classification metrics."""
    predictions = model.predict(texts)
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "precision_binary": round(float(precision_score(labels, predictions, average="binary", zero_division=0)), 6),
        "recall_binary": round(float(recall_score(labels, predictions, average="binary", zero_division=0)), 6),
        "f1_binary": round(float(f1_score(labels, predictions, average="binary", zero_division=0)), 6),
        "precision_macro": round(float(precision_score(labels, predictions, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(labels, predictions, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def main() -> None:
    """Train, evaluate, persist, and print both baseline models."""
    train = load_partition("train")
    validation = load_partition("val")
    test = load_partition("test")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "naive_bayes": build_pipeline(MultinomialNB(alpha=0.1)),
        "random_forest": build_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        ),
    }
    metrics: dict[str, Any] = {
        "dataset": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "models": {},
    }

    for name, model in models.items():
        model.fit(train["text"], train["label"])
        test_metrics = evaluate(model, test["text"], test["label"])
        metrics["models"][name] = {"test": test_metrics}
        output_path = MODEL_DIR / f"{name}_pipeline.joblib"
        dump(model, output_path)
        print(f"\n{name.replace('_', ' ').title()}")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value}")
        print(f"  saved: {output_path}")

    metrics_path = DATA_DIR / "baselines_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
