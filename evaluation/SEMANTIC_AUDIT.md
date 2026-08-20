# BETO Semantic Error Audit

This audit uses project taxonomy, not BETO output, to separate source-label problems from model errors.

Audited error rows: **73**

| Decision | Count | Meaning |
|---|---:|---|
| `source_label_issue` | 40 | BETO prediction is semantically consistent; source label conflicts with rubric. |
| `model_error` | 30 | Source label is retained; BETO prediction is wrong. |
| `ambiguous_keep_source` | 3 | Do not use for v2 until labeling policy is confirmed. |

## BETO On Audited Labels

| Metric | Value |
|---|---:|
| Accuracy | 0.917 |
| F1 macro | 0.911 |
| RF recall | 0.806 |
| RNF recall | 0.992 |

The source-label issues must be corrected before training. Ambiguous rows remain excluded until adjudication.

Readable row-level detail: `semantic_audit.html` and `semantic_audit.csv`.
