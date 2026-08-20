"""Prepare requirement datasets for Spanish RF/RNF classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("etl_prepare")
TEXT_NAMES = ("requirement text", "requirement", "text", "description", "sentence")
LABEL_NAMES = ("class", "label", "category", "type", "requirement type", "is quality", "isquality")
FUNCTIONAL_LABELS = {"fr", "rf", "functional", "functional requirement", "0"}
NON_FUNCTIONAL_LABELS = {
    "nfr",
    "rnf",
    "non-functional",
    "nonfunctional",
    "non functional",
    "non-functional requirement",
    "1",
}


def _normalise_column_name(value: object) -> str:
    """Return comparable column-name text."""
    return re.sub(r"[_\-]+", " ", str(value).strip().lower())


def _find_column(columns: list[object], candidates: tuple[str, ...], kind: str) -> str:
    """Find column by known name or a conservative partial match."""
    normalised = {_normalise_column_name(column): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    matches = [
        original
        for normalised_name, original in normalised.items()
        if any(candidate in normalised_name for candidate in candidates)
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not identify {kind} column. Available columns: {list(normalised.values())}")


def clean_text(value: object) -> str:
    """Normalize Unicode, whitespace, and surrounding punctuation noise."""
    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_input(path: Path) -> pd.DataFrame:
    """Read CSV or Excel input using the file extension."""
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError("Install openpyxl to read Excel datasets.") from exc
    return pd.read_csv(path)


def _load_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    """Read and concatenate input files while preserving source metadata."""
    frames = []
    for path in paths:
        frame = _read_input(path).copy()
        text_column = _find_column(list(frame.columns), TEXT_NAMES, "text")
        label_column = _find_column(list(frame.columns), LABEL_NAMES, "label")
        frames.append(
            pd.DataFrame(
                {
                    "text": frame[text_column].map(clean_text),
                    "label": map_labels(frame[label_column]),
                    "source": path.name,
                }
            )
        )
    if not frames:
        raise ValueError("At least one input dataset is required.")
    return pd.concat(frames, ignore_index=True, sort=False)


def map_labels(series: pd.Series) -> pd.Series:
    """Map binary or multi-class requirement labels to RF=0 and RNF=1."""
    values = series.map(lambda value: str(value).strip().lower())
    unique = set(values.dropna())

    if unique and unique <= FUNCTIONAL_LABELS | NON_FUNCTIONAL_LABELS:
        return values.map(lambda value: 1 if value in NON_FUNCTIONAL_LABELS else 0).astype("int64")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()) <= {0, 1}:
        return numeric.astype("int64")

    if len(unique) <= 1:
        raise ValueError("Label column must contain both RF and RNF classes.")

    # PROMISE multi-class data uses quality-category labels for NFRs. Any
    # recognized functional category remains RF; every other category is RNF.
    functional = values.isin(FUNCTIONAL_LABELS)
    if functional.all() or (~functional).all():
        raise ValueError(f"Unsupported label values: {sorted(unique)}")
    return (~functional).astype("int64")


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def translate_texts(texts: list[str], cache_path: Path, source: str = "en", target: str = "es") -> list[str]:
    """Translate texts with a persistent cache and retry handling."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise RuntimeError("Install deep-translator or pass --skip-translate.") from exc

    cache = _load_cache(cache_path)
    translator = GoogleTranslator(source=source, target=target)
    translated: list[str] = []
    for index, text in enumerate(texts, start=1):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in cache:
            for attempt in range(3):
                try:
                    cache[key] = clean_text(translator.translate(text))
                    break
                except Exception as exc:  # network providers expose varied errors
                    if attempt == 2:
                        # Keep pipeline usable when provider rejects an isolated sentence.
                        # Cache original so future runs do not repeatedly hit the provider.
                        LOGGER.warning("Translation unavailable at row %d; keeping original: %s", index, text)
                        cache[key] = text
                        break
                    time.sleep(2**attempt)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        translated.append(cache[key])
        if index % 25 == 0:
            LOGGER.info("Translated %d/%d rows", index, len(texts))
    return translated


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Run cleaning, translation, label mapping, and stratified partitioning."""
    input_paths = [args.input, *args.additional_input]
    frame = _load_inputs(input_paths)
    original_rows = len(frame)

    output = frame
    output = output[output["text"].str.len() >= 3]
    output = output.drop_duplicates(subset="text").reset_index(drop=True)
    translation_fallback_rows = 0
    if not args.skip_translate:
        original_texts = output["text"].tolist()
        translated_texts = translate_texts(original_texts, args.cache)
        translation_fallback_rows = sum(original == translated for original, translated in zip(original_texts, translated_texts))
        output["text"] = translated_texts

    if output["label"].nunique() != 2:
        raise ValueError("Prepared data must contain both labels 0 (RF) and 1 (RNF).")
    train, remainder = train_test_split(output, test_size=0.30, random_state=args.seed, stratify=output["label"])
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=args.seed, stratify=remainder["label"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "" if args.dataset_name == "dataset" else f"{args.dataset_name}_"
    output.to_csv(args.output_dir / f"{file_prefix}dataset_clean.csv", index=False)
    for name, partition in (("train", train), ("val", validation), ("test", test)):
        partition.to_csv(args.output_dir / f"{file_prefix}{name}.csv", index=False)

    report: dict[str, Any] = {
        "input_rows": original_rows,
        "clean_rows": len(output),
        "dropped_rows": original_rows - len(output),
        "inputs": [str(path) for path in input_paths],
        "sources": output["source"].value_counts().to_dict(),
        "label_counts": {"rf": int((output.label == 0).sum()), "rnf": int((output.label == 1).sum())},
        "duplicate_texts_removed": int(original_rows - len(output["text"].unique())),
        "dataset_name": args.dataset_name,
        "translated": not args.skip_translate,
        "translation_fallback_rows": translation_fallback_rows,
        "seed": args.seed,
        "splits": {
            name: {"rows": len(partition), "rf": int((partition.label == 0).sum()), "rnf": int((partition.label == 1).sum())}
            for name, partition in (("train", train), ("val", validation), ("test", test))
        },
    }
    (args.output_dir / "etl_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "dataset_raw.csv")
    parser.add_argument("--additional-input", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--cache", type=Path, default=root / ".translate_cache.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-translate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_args()
    LOGGER.info("ETL report: %s", prepare(arguments))
