"""Train BETO plus BiLSTM on prepared RF/RNF CSV partitions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.beto_lstm import BETOBiLSTM  # noqa: E402


class RequirementDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized requirement partition."""

    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.texts = frame["text"].astype(str).tolist()
        self.labels = torch.tensor(frame["label"].astype(int).tolist(), dtype=torch.long)
        encoded = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.encoded = {key: value for key, value in encoded.items()}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def set_seed(seed: int) -> None:
    """Make CPU/GPU training as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_partition(data_dir: Path, prefix: str, name: str) -> pd.DataFrame:
    """Load one ETL partition and validate its contract."""
    frame = pd.read_csv(data_dir / f"{prefix}_{name}.csv")
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"Partition must contain text and label columns: {frame}")
    if set(frame["label"].astype(int).unique()) != {0, 1}:
        raise ValueError("Partitions must contain labels 0 (RF) and 1 (RNF).")
    return frame


def evaluate(model: BETOBiLSTM, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device) -> dict[str, Any]:
    """Evaluate model and return metrics plus confusion matrix."""
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.item()))
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
    return {
        "loss": round(float(np.mean(losses)), 6),
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "precision_binary_rnf": round(float(precision_score(labels, predictions, zero_division=0)), 6),
        "recall_binary_rnf": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "f1_binary_rnf": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "f1_macro": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "training")
    parser.add_argument("--data-prefix", default="combined")
    parser.add_argument("--init-model", type=Path, default=ROOT / "backend" / "models" / "beto_rf_rnf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "backend" / "models" / "beto_lstm_rf_rnf")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    """Train, select by validation macro F1, and save model artifacts."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.init_model, local_files_only=True)
    encoder = BertModel.from_pretrained(args.init_model, local_files_only=True)
    model = BETOBiLSTM(encoder, hidden_size=args.lstm_hidden_size, dropout=args.dropout).to(device)

    # Keep same controlled fine-tuning policy as current BETO, while training new head.
    for parameter in model.encoder.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.encoder.layer[:9]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    frames = {name: load_partition(args.data_dir, args.data_prefix, name) for name in ("train", "val", "test")}
    datasets = {name: RequirementDataset(frame, tokenizer, args.max_length) for name, frame in frames.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size),
    }
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            assert output.loss is not None
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(output.loss.item()))
        validation = evaluate(model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": round(float(np.mean(train_losses)), 6), "validation": validation}
        history.append(record)
        print(json.dumps(record))
        if validation["f1_macro"] > best_f1:
            best_f1 = validation["f1_macro"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(args.output_dir / "base_encoder")
    tokenizer.save_pretrained(args.output_dir)
    torch.save(
        {key.removeprefix("encoder."): value for key, value in model.state_dict().items() if key.startswith("lstm.") or key.startswith("classifier.") or key.startswith("dropout.")},
        args.output_dir / "head.pt",
    )
    metadata = {
        "architecture": "BETO+BiLSTM",
        "base_encoder": "BETO",
        "labels": {"0": "RF", "1": "RNF"},
        "max_length": args.max_length,
        "lstm_hidden_size": args.lstm_hidden_size,
        "dropout": args.dropout,
        "data_prefix": args.data_prefix,
        "seed": args.seed,
        "device": str(device),
        "history": history,
        "test": test_metrics,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
