"""Run version 2 labeled RF/RNF evaluation against the local benchmark API."""

from __future__ import annotations

import csv
import json
import math
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
API_URL = "http://127.0.0.1:8000/api/predict/benchmark/batch"
BATCH_SIZE = 256
FILES = {
    "RF": EVALUATION_DIR / "requisitos_funcionales.txt",
    "RNF": EVALUATION_DIR / "requisitos_no_funcionales.txt",
}
MODEL_NAMES = ("BETO + BiLSTM", "BETO", "Random Forest", "Naive Bayes")
LABEL_IDS = {"RF": 0, "RNF": 1}
VERSION_SUFFIX = "_v2"


def read_requirements(path: Path) -> list[str]:
    """Read one non-empty requirement per line."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    requirements = [line for line in lines if line]
    if not requirements:
        raise ValueError(f"No requirements found in {path}")
    return requirements


def predict(texts: list[str]) -> list[dict[str, Any]]:
    """Call API in chunks below schema limit."""
    output: list[dict[str, Any]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        payload = json.dumps({"texts": texts[start : start + BATCH_SIZE]}).encode("utf-8")
        request = urllib.request.Request(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
        output.extend(body["predictions"])
    return output


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """Return 95% Wilson interval for a proportion."""
    if not total:
        return 0.0, 0.0
    z = 1.96
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def metrics(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    """Calculate binary classification metrics from labeled predictions."""
    actual = [expected] * len(rows)
    predicted = [int(row["label_id"]) for row in rows]
    matrix = [[0, 0], [0, 0]]
    for truth, prediction in zip(actual, predicted):
        matrix[truth][prediction] += 1

    per_class: dict[str, dict[str, float]] = {}
    for label, class_id in LABEL_IDS.items():
        tp = matrix[class_id][class_id]
        fn = sum(matrix[class_id]) - tp
        fp = sum(matrix[row][class_id] for row in LABEL_IDS.values() if row != class_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    correct = sum(truth == prediction for truth, prediction in zip(actual, predicted))
    accuracy = correct / len(rows) if rows else 0.0
    balanced_accuracy = sum(item["recall"] for item in per_class.values()) / 2
    low, high = wilson_interval(correct, len(rows))
    confidences = [float(row["confidence"]) for row in rows]
    errors = [row for row in rows if int(row["label_id"]) != expected]
    high_confidence_errors = [row for row in errors if float(row["confidence"]) >= 0.80]
    return {
        "count": len(rows),
        "correct": correct,
        "errors": len(errors),
        "accuracy": accuracy,
        "accuracy_wilson_95": [low, high],
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": sum(item["f1"] for item in per_class.values()) / 2,
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": matrix,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "mean_confidence_errors": (
            sum(float(row["confidence"]) for row in errors) / len(errors) if errors else None
        ),
        "high_confidence_errors_at_80": len(high_confidence_errors),
    }


def metrics_labeled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate metrics when each row carries its own expected label."""
    actual = [int(row["expected_label_id"]) for row in rows]
    predicted = [int(row["label_id"]) for row in rows]
    matrix = [[0, 0], [0, 0]]
    for truth, prediction in zip(actual, predicted):
        matrix[truth][prediction] += 1
    per_class: dict[str, dict[str, float]] = {}
    for label, class_id in LABEL_IDS.items():
        tp = matrix[class_id][class_id]
        fn = sum(matrix[class_id]) - tp
        fp = sum(matrix[row][class_id] for row in LABEL_IDS.values() if row != class_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    correct = sum(truth == prediction for truth, prediction in zip(actual, predicted))
    errors = [row for row, truth, prediction in zip(rows, actual, predicted) if truth != prediction]
    confidences = [float(row["confidence"]) for row in rows]
    return {
        "count": len(rows),
        "correct": correct,
        "errors": len(errors),
        "accuracy": correct / len(rows) if rows else 0.0,
        "accuracy_wilson_95": list(wilson_interval(correct, len(rows))),
        "balanced_accuracy": sum(item["recall"] for item in per_class.values()) / 2,
        "f1_macro": sum(item["f1"] for item in per_class.values()) / 2,
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": matrix,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "mean_confidence_errors": (
            sum(float(row["confidence"]) for row in errors) / len(errors) if errors else None
        ),
        "high_confidence_errors_at_80": sum(float(row["confidence"]) >= 0.80 for row in errors),
    }


def main() -> None:
    """Evaluate both labeled files and write reproducible artifacts."""
    labeled: dict[str, list[dict[str, Any]]] = {}
    for expected_label, path in FILES.items():
        texts = read_requirements(path)
        print(f"{expected_label}: {len(texts)} requirements")
        predictions = predict(texts)
        if len(predictions) != len(texts):
            raise RuntimeError(f"API returned {len(predictions)} rows for {len(texts)} inputs")
        labeled[expected_label] = [
            {"text": text, "expected_label": expected_label, "expected_label_id": LABEL_IDS[expected_label], **result}
            for text, result in zip(texts, predictions)
        ]

    combined = labeled["RF"] + labeled["RNF"]
    all_rows = labeled["RF"] + labeled["RNF"]
    report: dict[str, Any] = {
        "api_url": API_URL,
        "batch_size": BATCH_SIZE,
        "counts": {name: len(rows) for name, rows in labeled.items()} | {"combined": len(combined)},
        "models": {},
    }

    for model_name in MODEL_NAMES:
        model_rows = {
            subset: [
                {**row, **next(pred for pred in row["predictions"] if pred["model_name"] == model_name)}
                for row in rows
            ]
            for subset, rows in {**labeled, "combined": combined}.items()
        }
        report["models"][model_name] = {
            subset: metrics(rows, LABEL_IDS[subset]) if subset != "combined" else {
                **metrics_labeled(rows),
            }
            for subset, rows in model_rows.items()
        }

        for subset, rows in model_rows.items():
            for row in rows:
                row.pop("predictions", None)
                row.pop("model_name", None)
            output_path = EVALUATION_DIR / f"results_{model_name.lower().replace(' ', '_')}_{subset.lower()}{VERSION_SUFFIX}.csv"
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["text", "expected_label", "expected_label_id", "label", "label_id", "confidence", "latency_ms", "rank"])
                writer.writeheader()
                writer.writerows(rows)

    (EVALUATION_DIR / f"evaluation_results{VERSION_SUFFIX}.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (EVALUATION_DIR / f"evaluation_report{VERSION_SUFFIX}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# External RF/RNF Evaluation v2",
        "",
        f"API: `{API_URL}`",
        "",
        "Inputs: 200 RF and 200 RNF requirements. Ground truth comes from source file.",
        "",
        "| Model | Set | Accuracy | F1 macro | RF recall | RNF recall | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, subsets in report["models"].items():
        item = subsets["combined"]
        lines.append(
            f"| {model_name} | combined | {item['accuracy']:.3f} | {item['f1_macro']:.3f} | "
            f"{item['per_class']['RF']['recall']:.3f} | {item['per_class']['RNF']['recall']:.3f} | {item['errors']} |"
        )
    lines.extend(
        [
            "",
            "## BETO Findings",
            "",
            "- Strong RNF recall, weak RF recall on this external set.",
            "- Review source labels before retraining: functional file contains authentication, time, and quality language that may be ambiguous or non-functional.",
            "- Do not use confidence alone as correctness evidence.",
            "",
            "Artifacts: `evaluation_report_v2.json`, `evaluation_results_v2.json`, and versioned per-model CSV files.",
        ]
    )
    (EVALUATION_DIR / f"REPORT{VERSION_SUFFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise
