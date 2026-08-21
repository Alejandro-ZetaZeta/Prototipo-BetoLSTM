# RF/RNF Classification Report v2

## Executive Summary

This report evaluates the three models established in version 1 and the new
main model, **BETO+BiLSTM**, for binary classification of Spanish software
requirements:

- `RF` (`0`): Functional Requirement.
- `RNF` (`1`): Non-Functional Requirement.

The decisive comparison uses the same external benchmark for every model:
400 requirements, balanced between 200 RF and 200 RNF. On this benchmark,
BETO+BiLSTM is the strongest model:

- Accuracy: **0.905**.
- Macro F1: **0.905**.
- RF recall: **0.890**.
- RNF recall: **0.920**.
- Errors: **38 of 400**.

Compared with plain BETO, the final model improves accuracy by **8.75
percentage points**, macro F1 by **9.31 points**, and RF recall by **24.5
points**. It gives up 7 points of RNF recall, but produces a substantially
more balanced classifier instead of strongly favoring RNF.

## 1. Evaluation Scope and Data

### 1.1 Version 1 local test

The v1 local evaluation used the original cleaned dataset and stratified
partitions:

| Partition | Records |
|---|---:|
| Train | 435 |
| Validation | 93 |
| Test | 94 |
| Total | 622 |

The test set contains 36 RF and 58 RNF records. These results are useful for
reproducing v1, but they are not directly comparable with the v2 BiLSTM local
test because v2 was trained and tested with the larger `combined` dataset.

### 1.2 Combined dataset used for BETO+BiLSTM

The combined ETL process used `Dataset6000Req.xlsx` and `dataset_raw.csv`:

| Item | Value |
|---|---:|
| Input records | 6,708 |
| Clean records | 6,587 |
| Removed records | 121 |
| RF records | 4,167 |
| RNF records | 2,420 |
| Train | 4,610 |
| Validation | 988 |
| Test | 989 |

The split used seed `42` and stratification. Requirements from the XLSX were
translated to Spanish before training. Duplicate texts were removed during
ETL.

### 1.3 External benchmark

The primary v2 comparison was executed through:

`http://127.0.0.1:8000/api/predict/benchmark/batch`

The benchmark contains 400 requirements sourced from the evaluation text
files:

- 200 RF requirements.
- 200 RNF requirements.

All four models were evaluated on this same set. Confusion matrices use rows
as true labels and columns as predicted labels, ordered `[RF, RNF]`.

## 2. The Three Initial Models

### 2.1 BETO

BETO is the Spanish Transformer model
`dccuchile/bert-base-spanish-wwm-cased`. It creates contextual token
representations, so word meaning changes according to surrounding words.
This matters for requirements because terms such as `debe`, `permitir`,
`tiempo`, `seguro`, `disponible`, and `usuarios` can signal different
requirement types depending on their context.

### 2.2 Random Forest

Random Forest is a classical lexical baseline. The pipeline converts each
requirement into TF-IDF features using word unigrams and bigrams, then trains
100 decision trees. It provides a useful non-neural reference and is less
computationally demanding than BETO.

### 2.3 Multinomial Naive Bayes

Naive Bayes uses the same TF-IDF representation as Random Forest and applies
`MultinomialNB(alpha=0.1)`. Its conditional-independence assumption makes it
simple and fast, but it does not represent word order or contextual meaning.

## 3. Version 1 Performance

### 3.1 Local held-out test

| Model | Accuracy | Precision RNF | Recall RNF | F1 RNF | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BETO | 0.925532 | 0.918033 | 0.965517 | 0.941176 | 0.928713 | 0.913314 | **0.919864** |
| Naive Bayes | 0.872340 | 0.896552 | 0.896552 | 0.896552 | 0.864943 | 0.864943 | 0.864943 |
| Random Forest | 0.861702 | 0.868852 | 0.913793 | 0.890756 | 0.858669 | 0.845785 | 0.851175 |

Confusion matrices:

```text
BETO            [[31, 5], [2, 56]]
Naive Bayes     [[30, 6], [6, 52]]
Random Forest   [[28, 8], [5, 53]]
```

On this local test, BETO had the best accuracy, macro F1, and total error
count. It made 7 errors, compared with 12 for Naive Bayes and 13 for Random
Forest.

### 3.2 External benchmark baseline results

| Model | Accuracy | Macro F1 | RF Recall | RNF Recall | Errors |
|---|---:|---:|---:|---:|---:|
| BETO | 0.817500 | 0.811903 | 0.645 | 0.990 | 73 |
| Naive Bayes | 0.685000 | 0.664000 | 0.435 | 0.935 | 126 |
| Random Forest | 0.527500 | 0.391691 | 0.055 | 1.000 | 189 |

The external benchmark reveals an important weakness hidden by aggregate
accuracy: plain BETO strongly favored RNF. It correctly identified 198 of
200 RNF requirements, but classified 71 of 200 RF requirements as RNF.
Random Forest showed an even stronger one-class bias, correctly identifying
only 11 RF requirements.

## 4. Parameters Preserved from Version 1

The BiLSTM experiment deliberately preserved v1's main training controls so
that the architectural change, rather than a completely different training
recipe, could be evaluated.

| Parameter | v1 BETO | BETO+BiLSTM |
|---|---|---|
| Base model | `dccuchile/bert-base-spanish-wwm-cased` | Same BETO encoder, initialized from v1 artifact |
| Labels | `0=RF`, `1=RNF` | Same |
| Maximum length | 128 tokens | 128 tokens |
| Batch size | 16 | 16 |
| Maximum epochs | 6 | 6 |
| Learning rate | `2e-5` | `2e-5` |
| Weight decay | `0.01` | `0.01` |
| Optimizer | AdamW | AdamW |
| Seed | 42 | 42 |
| Frozen layers | Embeddings and first 9 of 12 BETO layers | Same |
| Trainable BETO layers | Last 3 layers and classifier head | Last 3 layers plus new BiLSTM/classifier head |
| Evaluation | Every epoch | Every epoch |

The BiLSTM training additionally used validation macro F1 for checkpoint
selection, patience of 2 epochs, and gradient clipping at `1.0`.

## 5. Why BETO Was Selected

BETO was selected for three reasons:

1. **Spanish specialization.** The model was pretrained on Spanish text,
   matching the language of the target requirements better than a generic
   English or language-agnostic baseline.
2. **Contextual representations.** BETO can distinguish meaning from word
   context instead of treating a requirement as an unordered collection of
   terms.
3. **Best v1 evidence.** BETO achieved the best v1 local result: 0.925532
   accuracy and 0.919864 macro F1. It also outperformed both classical
   baselines on the external benchmark.

BETO was not selected because it was perfect. The external benchmark showed
that its RNF recall of `0.990` came with RF recall of only `0.645`. This
imbalance motivated the next architectural step.

## 6. Why BETO Was Combined with BiLSTM

BETO and BiLSTM address different parts of the sequence-classification
problem:

- BETO supplies deep, pretrained, Spanish contextual representations for
  each token.
- BiLSTM reads the resulting sequence in both directions and learns how
  information across the requirement contributes to its final classification.
- The classifier receives the concatenated final forward and backward LSTM
  states, applies dropout, and predicts RF or RNF.

The combination was chosen because the task is not only vocabulary matching.
Requirement category can depend on relations across the sentence, such as
an action, its object, and a constraint on time, security, availability, or
quality. BETO provides semantic features; the BiLSTM adds an explicit
sequence-modeling stage over those features.

The architecture uses:

| Component | Configuration |
|---|---|
| BETO hidden size | 768 |
| BiLSTM hidden size | 256 per direction |
| Directions | 2, forward and backward |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Classifier input | 512 features, from both directions |
| Output classes | 2 |

## 7. What Was Salvaged from BETO

The combination did not discard the successful part of v1. It retained:

- The pretrained BETO vocabulary and tokenizer.
- BETO's Spanish language knowledge.
- The v1-trained encoder as initialization.
- The v1 maximum sequence length, batch size, optimizer family, learning
  rate, weight decay, random seed, and controlled fine-tuning policy.
- The last three BETO encoder layers as the adaptable Transformer component.

Embeddings and the first nine encoder layers remained frozen. This preserved
general linguistic knowledge, reduced the number of parameters being changed,
and allowed training to focus on task-specific representations and the new
sequence head.

## 8. Version 2 BETO+BiLSTM Performance

### 8.1 Primary external benchmark

| Model | Accuracy | Macro F1 | RF Precision | RF Recall | RNF Precision | RNF Recall | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BETO+BiLSTM** | **0.905** | **0.904979** | 0.917526 | **0.890** | 0.893204 | 0.920 | **38** |
| BETO | 0.817500 | 0.811903 | 0.984733 | 0.645 | 0.736059 | 0.990 | 73 |
| Naive Bayes | 0.685000 | 0.664000 | 0.870000 | 0.435 | 0.623333 | 0.935 | 126 |
| Random Forest | 0.527500 | 0.391691 | 1.000000 | 0.055 | 0.514139 | 1.000 | 189 |

BETO+BiLSTM confusion matrix:

```text
[[178, 22],
 [ 16, 184]]
```

Interpretation:

- 178 RF requirements were correctly classified.
- 184 RNF requirements were correctly classified.
- 22 RF requirements were classified as RNF.
- 16 RNF requirements were classified as RF.

### 8.2 Improvement over plain BETO

| Metric | BETO | BETO+BiLSTM | Change |
|---|---:|---:|---:|
| Accuracy | 0.817500 | 0.905000 | **+0.087500** |
| Macro F1 | 0.811903 | 0.904979 | **+0.093076** |
| RF recall | 0.645 | 0.890 | **+0.245** |
| RNF recall | 0.990 | 0.920 | -0.070 |
| Errors | 73 | 38 | **-35** |

The main gain is correction of BETO's RF under-recognition. RF errors fell
from 71 to 22, while RNF errors rose only from 2 to 16. The result is a
better-balanced model, which is preferable for a classifier where both
classes matter.

### 8.3 Local combined test

The saved BiLSTM artifact was also evaluated on the combined local test set
of 989 records:

| Metric | BETO+BiLSTM |
|---|---:|
| Accuracy | 0.864804 |
| Precision RNF | 0.813559 |
| Recall RNF | 0.784314 |
| F1 RNF | 0.798669 |
| F1 macro | 0.848451 |
| Loss | 0.334253 |

Confusion matrix:

```text
[[534, 55],
 [ 66, 240]]
```

This result must not be compared directly with the v1 94-row local test.
The datasets, sample counts, class distributions, and training data differ.
The external benchmark is the valid apples-to-apples comparison among all
models.

## 9. What Improved and What Was Gained

### Improved

- RF recall increased from `0.645` to `0.890` on the external benchmark.
- Macro F1 increased from `0.811903` to `0.904979`.
- Total benchmark errors decreased by 35, from 73 to 38.
- Predictions became more balanced across RF and RNF.
- The classifier stopped behaving as if most ambiguous requirements were RNF.

### Gained

- BETO's Spanish contextual knowledge remained available.
- Sequence information was modeled explicitly after Transformer encoding.
- A regularized classification head was added through dropout `0.3`.
- Larger combined training data was used for the final artifact.
- The model retained local CPU inference support and is the active API model.

### Trade-off

RNF recall decreased from `0.990` to `0.920`. This is not a regression in
overall usefulness: it reflects a deliberate reduction in one-class bias.
The final model sacrifices 14 RNF detections to recover 49 RF detections,
reducing total errors from 73 to 38.

## 10. Interpretation and Limitations

The external benchmark contains source-label noise. The existing semantic
audit classified the 73 plain BETO errors as:

| Audit decision | Count |
|---|---:|
| Source-label issue | 40 |
| Model error | 30 |
| Ambiguous, retained temporarily | 3 |

Therefore, benchmark scores should not be treated as perfect ground truth.
The audit also raised BETO's diagnostic accuracy to `0.9175` and macro F1 to
`0.910882` under corrected rubric-based labels. A comparable semantic audit
of all BETO+BiLSTM errors would strengthen the conclusion about the final
model.

Confidence is not correctness. The v2 evaluation recorded high-confidence
errors, including 19 BETO+BiLSTM errors at confidence of at least `0.80`.
The API confidence value should therefore be used as an uncertainty signal,
not as proof of a correct label.

## 11. Final Decision

BETO remains the correct foundation because it is Spanish-specific, context
aware, and was the strongest v1 model. BETO+BiLSTM is selected as the main
model because it preserves those strengths while adding bidirectional
sequence modeling and correcting BETO's most consequential weakness: poor RF
recall on external requirements.

The final external result, **0.905 accuracy and 0.905 macro F1**, demonstrates
that the combination was not cosmetic. It produced 35 fewer errors and a
much more balanced RF/RNF decision boundary. The primary next improvement is
not a larger head; it is label-quality review and a matched error audit for
BETO+BiLSTM.

## 12. Reproducibility Artifacts

- v1 report: `evaluation/REPORT.md`.
- v1 BETO notebook: `training/colab_train_beto.ipynb`.
- Baseline training: `training/train_baselines.py`.
- BiLSTM training: `training/train_beto_lstm.py`.
- BiLSTM architecture: `backend/app/beto_lstm.py`.
- BiLSTM artifact metadata: `backend/models/beto_lstm_rf_rnf/metadata.json`.
- v1 external metrics: `evaluation/evaluation_report.json`.
- v2 external metrics: `evaluation/evaluation_report_v2.json`.
- Semantic audit: `evaluation/SEMANTIC_AUDIT.md`.
