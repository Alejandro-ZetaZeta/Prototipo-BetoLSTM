# External RF/RNF Evaluation v2

API: `http://127.0.0.1:8000/api/predict/benchmark/batch`

Inputs: 200 RF and 200 RNF requirements. Ground truth comes from source file.

| Model | Set | Accuracy | F1 macro | RF recall | RNF recall | Errors |
|---|---:|---:|---:|---:|---:|---:|
| BETO + BiLSTM | combined | 0.905 | 0.905 | 0.890 | 0.920 | 38 |
| BETO | combined | 0.818 | 0.812 | 0.645 | 0.990 | 73 |
| Random Forest | combined | 0.527 | 0.392 | 0.055 | 1.000 | 189 |
| Naive Bayes | combined | 0.685 | 0.664 | 0.435 | 0.935 | 126 |

## BETO Findings

- Strong RNF recall, weak RF recall on this external set.
- Review source labels before retraining: functional file contains authentication, time, and quality language that may be ambiguous or non-functional.
- Do not use confidence alone as correctness evidence.

Artifacts: `evaluation_report_v2.json`, `evaluation_results_v2.json`, and versioned per-model CSV files.
