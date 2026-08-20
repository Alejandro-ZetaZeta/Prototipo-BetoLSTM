"""Prepare a human audit queue without changing source labels."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
OUTPUT_PATH = EVALUATION_DIR / "audit_queue.csv"
SEED = 42
CONTROL_COUNT_PER_CLASS = 20


def signal_category(text: str) -> str:
    """Suggest review priority; never use this as a replacement label."""
    categories = {
        "security": r"autent|validar|huella|contrase|sms|pin|cifrar|segur|acceso no autorizado|doble factor",
        "performance": r"segundo|tiempo|rápido|carga|disponib|concurrent|rendimiento|latencia",
        "usability_quality": r"accesib|wcag|experiencia|satisfact|comprensible|amigable|flexible|razonable|apropiad",
        "functional_action": r"permit|registr|crear|elimin|generar|export|calcular|determinar|descargar|autenticar",
    }
    matched = [name for name, pattern in categories.items() if re.search(pattern, text, re.IGNORECASE)]
    if len(matched) > 1:
        return "mixed:" + "+".join(matched)
    return matched[0] if matched else "other"


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    errors = [
        row
        for row in rows
        if any(int(prediction["label_id"]) != int(row["expected_label_id"]) for prediction in row["predictions"] if prediction["model_name"] == "BETO")
    ]
    correct_by_class = {
        label: [
            row
            for row in rows
            if row["expected_label"] == label
            and next(prediction for prediction in row["predictions"] if prediction["model_name"] == "BETO")["label_id"] == row["expected_label_id"]
        ]
        for label in ("RF", "RNF")
    }
    randomizer = random.Random(SEED)
    controls = []
    for label, candidates in correct_by_class.items():
        controls.extend(randomizer.sample(candidates, min(CONTROL_COUNT_PER_CLASS, len(candidates))))

    selected = [(row, "error") for row in errors] + [(row, "control") for row in controls]
    selected.sort(key=lambda item: (item[1] != "error", item[0]["expected_label"], item[0]["text"]))
    output_rows = []
    for row, queue_type in selected:
        beto = next(prediction for prediction in row["predictions"] if prediction["model_name"] == "BETO")
        output_rows.append(
            {
                "queue_type": queue_type,
                "text": row["text"],
                "source_label": row["expected_label"],
                "source_label_id": row["expected_label_id"],
                "beto_label": beto["label"],
                "beto_label_id": beto["label_id"],
                "beto_confidence": beto["confidence"],
                "signal_category": signal_category(row["text"]),
                "audit_label": "",
                "audit_decision": "",
                "audit_reason": "",
                "reviewer": "",
            }
        )

    fieldnames = list(output_rows[0])
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "seed": SEED,
        "error_rows": len(errors),
        "control_rows": len(controls),
        "queue_rows": len(output_rows),
        "controls_per_class": CONTROL_COUNT_PER_CLASS,
        "source_labels_unchanged": True,
        "signal_category_is_only_a_review_hint": True,
    }
    (EVALUATION_DIR / "audit_sampling.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
