"""Generate report plots from the persisted BETO model and local test set.

The repository does not contain Trainer epoch logs, so these are final-model
test diagnostics rather than training-history curves. This distinction is
written into each figure and the final report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "backend" / "models" / "beto_rf_rnf"
TEST_PATH = ROOT / "training" / "test.csv"
OUTPUT_DIR = ROOT / "evaluation"


def predict_test_set() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, predictions, and per-example cross-entropy losses."""
    frame = pd.read_csv(TEST_PATH, usecols=["text", "label"])
    labels = frame["label"].astype(int).to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    predictions: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(frame), 16):
            texts = frame["text"].iloc[start : start + 16].astype(str).tolist()
            batch_labels = torch.tensor(labels[start : start + 16], dtype=torch.long)
            batch = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            logits = model(**batch).logits
            batch_loss = torch.nn.functional.cross_entropy(
                logits, batch_labels, reduction="none"
            )
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            losses.extend(batch_loss.cpu().numpy().tolist())
    return labels, np.asarray(predictions), np.asarray(losses)


def style_axes(ax: plt.Axes) -> None:
    """Apply consistent report styling."""
    ax.set_facecolor("#f7f8fb")
    ax.grid(axis="y", color="#d9dee8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8c0ce")
    ax.spines["bottom"].set_color("#b8c0ce")


def save_performance_curve(labels: np.ndarray, predictions: np.ndarray) -> None:
    """Save cumulative accuracy and per-example correctness diagnostic."""
    correct = labels == predictions
    cumulative_accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    examples = np.arange(1, len(correct) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.scatter(
        examples,
        correct.astype(int),
        c=np.where(correct, "#1f9d72", "#d95d62"),
        s=28,
        alpha=0.9,
        label="Correct / incorrect prediction",
    )
    ax.plot(
        examples,
        cumulative_accuracy,
        color="#4057d6",
        linewidth=2.6,
        label="Cumulative accuracy",
    )
    ax.axhline(
        float(correct.mean()),
        color="#4057d6",
        linestyle="--",
        linewidth=1,
        alpha=0.65,
        label=f"Final accuracy: {correct.mean():.3f}",
    )
    ax.set_title("BETO performance diagnostic on local test set", loc="left", weight="bold")
    ax.text(
        0,
        1.02,
        "Final model, ordered test examples; not an epoch-level training curve",
        transform=ax.transAxes,
        color="#5d6675",
        fontsize=9,
    )
    ax.set_xlabel("Test example")
    ax.set_ylabel("Accuracy / correctness")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "beto_performance_curve.png", bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(losses: np.ndarray) -> None:
    """Save per-example cross-entropy and rolling loss diagnostics."""
    examples = np.arange(1, len(losses) + 1)
    window = min(10, len(losses))
    rolling = pd.Series(losses).rolling(window=window, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.plot(examples, losses, color="#d95d62", alpha=0.35, linewidth=1.2, label="Per-example loss")
    ax.plot(examples, rolling, color="#4057d6", linewidth=2.6, label=f"Rolling mean (n={window})")
    ax.axhline(
        float(losses.mean()),
        color="#1f9d72",
        linestyle="--",
        linewidth=1,
        label=f"Mean cross-entropy: {losses.mean():.3f}",
    )
    ax.set_title("BETO loss diagnostic on local test set", loc="left", weight="bold")
    ax.text(
        0,
        1.02,
        "Final-model cross-entropy by ordered test example; not training loss by epoch",
        transform=ax.transAxes,
        color="#5d6675",
        fontsize=9,
    )
    ax.set_xlabel("Test example")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "beto_loss_curve.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate both PNG report assets."""
    labels, predictions, losses = predict_test_set()
    save_performance_curve(labels, predictions)
    save_loss_curve(losses)
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
