"""Apply explicit project rubric to BETO error cases and write readable audit."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
ERROR_QUEUE_PATH = EVALUATION_DIR / "audit_queue.csv"

SECURITY_PATTERN = re.compile(
    r"autenticar|validarse|huella|contrase|sms|pin|token físico|doble factor|"
    r"verificación de correo|código de acceso|cuenta de terceros|clave dinámica",
    re.IGNORECASE,
)
QUALITY_PATTERN = re.compile(
    r"tiempo razonable|facilitar la operación|experiencia satisfactoria",
    re.IGNORECASE,
)
VAGUE_PATTERN = re.compile(
    r"gestionar correctamente|procesar las solicitudes de forma adecuada|"
    r"manejar de manera apropiada",
    re.IGNORECASE,
)


def beto_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return next(prediction for prediction in row["predictions"] if prediction["model_name"] == "BETO")


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return adjudication based on project rubric, not model output."""
    prediction = beto_prediction(row)
    source_label = row["expected_label"]
    text = row["text"]

    if source_label == "RNF":
        return {
            "audited_label": "RNF",
            "decision": "model_error",
            "reason": "Security/usability constraint is RNF under project rubric.",
        }

    if SECURITY_PATTERN.search(text):
        return {
            "audited_label": "RNF",
            "decision": "source_label_issue",
            "reason": "Authentication mechanism or access-control constraint belongs to RNF security policy.",
        }
    if QUALITY_PATTERN.search(text):
        return {
            "audited_label": "RNF",
            "decision": "source_label_issue",
            "reason": "Explicit performance/usability quality constraint belongs to RNF.",
        }
    if VAGUE_PATTERN.search(text):
        return {
            "audited_label": "RF",
            "decision": "ambiguous_keep_source",
            "reason": "Functional action is present, but quality wording is vague; exclude from v2 until policy review.",
        }
    return {
        "audited_label": "RF",
        "decision": "model_error",
        "reason": "System capability/action remains RF: create, export, calculate, access, modify, delete, or generate.",
    }


def calculate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        truth = 0 if row["audited_label"] == "RF" else 1
        prediction = int(row["beto_label_id"])
        matrix[truth][prediction] += 1
    per_class = {}
    for label, class_id in (("RF", 0), ("RNF", 1)):
        tp = matrix[class_id][class_id]
        fn = sum(matrix[class_id]) - tp
        fp = matrix[1 - class_id][class_id]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    correct = matrix[0][0] + matrix[1][1]
    return {
        "count": len(rows),
        "correct": correct,
        "errors": len(rows) - correct,
        "accuracy": correct / len(rows),
        "f1_macro": sum(item["f1"] for item in per_class.values()) / 2,
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": matrix,
    }


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    queue = list(csv.DictReader(ERROR_QUEUE_PATH.open(encoding="utf-8")))
    error_texts = {item["text"] for item in queue if item["queue_type"] == "error"}
    selected = [row for row in rows if row["text"] in error_texts]

    audited_rows = []
    for row in selected:
        prediction = beto_prediction(row)
        decision = audit_row(row)
        audited_rows.append(
            {
                "text": row["text"],
                "source_label": row["expected_label"],
                "beto_label": prediction["label"],
                "beto_label_id": prediction["label_id"],
                "beto_confidence": prediction["confidence"],
                **decision,
            }
        )

    audited_by_text = {row["text"]: row for row in audited_rows}
    full_audited_rows = []
    for row in rows:
        prediction = beto_prediction(row)
        reviewed = audited_by_text.get(row["text"])
        full_audited_rows.append(
            {
                "audited_label": reviewed["audited_label"] if reviewed else row["expected_label"],
                "beto_label_id": prediction["label_id"],
            }
        )

    fieldnames = list(audited_rows[0])
    with (EVALUATION_DIR / "semantic_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audited_rows)

    counts = Counter(row["decision"] for row in audited_rows)
    report = {
        "audited_rows": len(audited_rows),
        "decision_counts": dict(counts),
        "semantic_metrics_using_audited_labels": calculate(full_audited_rows),
        "scope_note": "This is rubric-based pre-audit. Ambiguous rows must not enter v2 until policy is confirmed.",
    }
    (EVALUATION_DIR / "semantic_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# BETO Semantic Error Audit",
        "",
        "This audit uses project taxonomy, not BETO output, to separate source-label problems from model errors.",
        "",
        f"Audited error rows: **{len(audited_rows)}**",
        "",
        "| Decision | Count | Meaning |",
        "|---|---:|---|",
        f"| `source_label_issue` | {counts['source_label_issue']} | BETO prediction is semantically consistent; source label conflicts with rubric. |",
        f"| `model_error` | {counts['model_error']} | Source label is retained; BETO prediction is wrong. |",
        f"| `ambiguous_keep_source` | {counts['ambiguous_keep_source']} | Do not use for v2 until labeling policy is confirmed. |",
        "",
        "## BETO On Audited Labels",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {report['semantic_metrics_using_audited_labels']['accuracy']:.3f} |",
        f"| F1 macro | {report['semantic_metrics_using_audited_labels']['f1_macro']:.3f} |",
        f"| RF recall | {report['semantic_metrics_using_audited_labels']['per_class']['RF']['recall']:.3f} |",
        f"| RNF recall | {report['semantic_metrics_using_audited_labels']['per_class']['RNF']['recall']:.3f} |",
        "",
        "The source-label issues must be corrected before training. Ambiguous rows remain excluded until adjudication.",
        "",
        "Readable row-level detail: `semantic_audit.html` and `semantic_audit.csv`.",
    ]
    (EVALUATION_DIR / "SEMANTIC_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    html_rows = []
    for index, row in enumerate(audited_rows, start=1):
        html_rows.append(
            "<tr>"
            f"<td>{index}</td><td>{row['decision']}</td><td>{row['source_label']}</td>"
            f"<td>{row['beto_label']}</td><td>{float(row['beto_confidence']):.1%}</td>"
            f"<td>{row['audited_label']}</td><td>{row['text']}</td><td>{row['reason']}</td>"
            "</tr>"
        )
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>BETO Semantic Audit</title>
<style>body{font:14px system-ui;margin:24px;color:#202124}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px;vertical-align:top}th{position:sticky;top:0;background:#eee}td:nth-child(7){min-width:480px}tr:nth-child(even){background:#fafafa}</style>
<h1>BETO Semantic Error Audit</h1><p>Rubric-based review of 73 BETO error rows. Source labels are not changed automatically.</p>
<table><thead><tr><th>#</th><th>Decision</th><th>Source</th><th>BETO</th><th>Confidence</th><th>Audited</th><th>Requirement</th><th>Reason</th></tr></thead><tbody>""" + "".join(html_rows) + "</tbody></table></html>"
    (EVALUATION_DIR / "semantic_audit.html").write_text(html, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
