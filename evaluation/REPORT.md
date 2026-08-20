# RF/RNF Classification Report

## Scope

This report combines two evaluation scopes already present in the project:

- **Local held-out test set:** 94 records from `training/test.csv` (36 RF, 58 RNF). The split was created with seed 42 from 622 cleaned records: 435 train, 93 validation, 94 test.
- **External benchmark:** 400 labeled requirements from `requisitos_funcionales.txt` and `requisitos_no_funcionales.txt` (200 RF, 200 RNF). Results were produced through `evaluation/run_evaluation.py` against the local API.

Labels use `0 = RF` (Functional Requirement) and `1 = RNF` (Non-Functional Requirement). Confusion matrices use rows = true label and columns = predicted label, ordered `[RF, RNF]`.

## 1. Local Held-Out Test Results

Metrics requested for the final test set are shown below. `Precision`, `Recall`, and `F1` are reported both for RNF as the positive binary class and as macro averages. Macro values are the preferred summary because both classes matter.

| Model | Accuracy | Precision (RNF) | Recall (RNF) | F1 (RNF) | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BETO | 0.925532 | 0.918033 | 0.965517 | 0.941176 | 0.928713 | 0.913314 | 0.919864 |
| Random Forest | 0.861702 | 0.868852 | 0.913793 | 0.890756 | 0.858669 | 0.845785 | 0.851175 |
| Naive Bayes | 0.872340 | 0.896552 | 0.896552 | 0.896552 | 0.864943 | 0.864943 | 0.864943 |

### Confusion Counts

| Model | RF correct | RNF correct | RF -> RNF | RNF -> RF | Total errors |
|---|---:|---:|---:|---:|---:|
| BETO | 31 | 56 | 5 | 2 | 7 |
| Random Forest | 28 | 53 | 8 | 5 | 13 |
| Naive Bayes | 30 | 52 | 6 | 6 | 12 |

Equivalent matrices:

```text
BETO            [[31, 5], [2, 56]]
Random Forest   [[28, 8], [5, 53]]
Naive Bayes     [[30, 6], [6, 52]]
```

Interpretation: for BETO, 31 functional requirements were classified correctly and 56 non-functional requirements were classified correctly. Five functional requirements were classified as non-functional, and two non-functional requirements were classified as functional.

## 2. Model Configuration

### BETO

| Hyperparameter | Value |
|---|---|
| Base model | `dccuchile/bert-base-spanish-wwm-cased` |
| Maximum sequence length | 128 tokens |
| Epochs | 6 |
| Training batch size | 16 per device |
| Evaluation batch size | 16 per device |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Optimizer | Trainer default AdamW setup |
| Seed | 42 |
| Evaluation schedule | Every epoch |
| Best-model metric | RNF binary F1, greater is better |
| Layer freezing | Yes: embeddings and first 9 of 12 encoder layers frozen |
| Trainable layers | Last 3 encoder layers and classification head |
| Padding/tokenization | Dynamic padding, truncation to 128 tokens |

The settings come from `training/colab_train_beto.ipynb`. The persisted artifact is `backend/models/beto_rf_rnf`.

### Random Forest

- TF-IDF features with word n-grams `(1, 2)`.
- Maximum 5,000 TF-IDF features.
- Project-defined Spanish stopword list.
- `n_estimators=100`, `random_state=42`, `n_jobs=-1`.

Random Forest has no epoch, learning-rate, or weight-decay hyperparameters.

### Naive Bayes

- Same TF-IDF preprocessing as Random Forest.
- Multinomial Naive Bayes with `alpha=0.1`.

Naive Bayes has no epoch, learning-rate, or weight-decay hyperparameters.

Source implementation: `training/train_baselines.py`.

## 3. External Benchmark Results

This benchmark is separate from the local held-out test set. It contains 200 RF and 200 RNF requirements sourced from the evaluation text files.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Errors |
|---|---:|---:|---:|---:|---:|
| BETO | 0.817500 | 0.860396 | 0.817500 | 0.811903 | 73 |
| Random Forest | 0.527500 | 0.757069 | 0.527500 | 0.391691 | 189 |
| Naive Bayes | 0.685000 | 0.746667 | 0.685000 | 0.664000 | 126 |

### External Confusion Counts

| Model | RF correct | RNF correct | RF -> RNF | RNF -> RF | Total errors |
|---|---:|---:|---:|---:|---:|
| BETO | 129 | 198 | 71 | 2 | 73 |
| Random Forest | 11 | 200 | 189 | 0 | 189 |
| Naive Bayes | 87 | 187 | 113 | 13 | 126 |

Equivalent matrices:

```text
BETO            [[129, 71], [2, 198]]
Random Forest   [[11, 189], [0, 200]]
Naive Bayes     [[87, 113], [13, 187]]
```

External benchmark finding: BETO strongly favors RNF recall (`0.990`) but has weaker RF recall (`0.645`). The source functional file contains authentication, timing, and quality language that can be semantically non-functional, so benchmark accuracy should be interpreted alongside the audit below.

## 4. Semantic Audit Context

The existing audit reviewed all 73 BETO errors from the external benchmark against the project taxonomy:

| Audit decision | Count |
|---|---:|
| Source-label issue | 40 |
| Model error | 30 |
| Ambiguous, keep source temporarily | 3 |

Using audited labels, BETO scored accuracy `0.917500` and macro F1 `0.910882`, with matrix `[[129, 31], [2, 238]]`. This is a rubric-based diagnostic, not a replacement for the original benchmark result. Details: `semantic_audit.html`, `semantic_audit.csv`, and `semantic_audit_report.json`.

## 5. BETO PNG Curves

Files:

- [`beto_performance_curve.png`](beto_performance_curve.png)
- [`beto_loss_curve.png`](beto_loss_curve.png)

Reproduce them with:

```text
python evaluation/generate_beto_curves.py
```

Important limitation: the repository contains the final BETO weights and `test_metrics.json`, but no `trainer_state.json`, epoch logs, or saved checkpoints with validation history. Therefore, true training curves such as `epoch vs validation accuracy` and `epoch vs training/validation loss` cannot be reconstructed from current artifacts without retraining.

The supplied PNGs are clearly labeled **final-model test diagnostics**:

- Performance plot: cumulative accuracy and correct/incorrect predictions over ordered local test examples.
- Loss plot: per-example cross-entropy and rolling mean over ordered local test examples.

They must not be described as epoch-level training curves.

## 6. Source Artifacts

- Local BETO metrics: `beto_local_test_metrics.json` and `backend/models/beto_rf_rnf/test_metrics.json`.
- Local baseline metrics: `training/baselines_metrics.json`.
- External metrics: `evaluation_report.json`.
- External row-level predictions: `results_*_combined.csv`.
- Training notebook and BETO settings: `training/colab_train_beto.ipynb`.
- Baseline training code and settings: `training/train_baselines.py`.
