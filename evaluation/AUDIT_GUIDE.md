# RF/RNF Label Audit

`audit_queue.csv` is a review queue. It does not alter source labels and does not assign automatic replacement labels.

## Review Order

1. Review every row with `queue_type=error`.
2. Review control rows with `queue_type=control` to detect overfitting in the audit process.
3. If a label is changed, enter `RF` or `RNF` in `audit_label` and explain decision in `audit_reason`.
4. Use `ambiguous` in `audit_decision` when requirement mixes capability and quality/security constraint.

## Label Rubric

- `RF`: system capability, operation, calculation, registration, deletion, generation, export, or authentication action.
- `RNF`: measurable quality or constraint: performance, availability, security protection, encryption, usability, accessibility, reliability, or capacity.
- `ambiguous`: both categories are present or project policy cannot determine one label.

Security wording needs explicit policy. “The system authenticates the user” can be RF. “Authentication must use SMS, fingerprint, PIN, or two-factor verification” can be RNF when mechanism is treated as a security constraint.

## Do Not Do

- Do not change labels because BETO disagreed.
- Do not train on reviewed rows before preserving an untouched external test set.
- Do not use confidence as ground truth.

After review, create an adjudicated dataset from rows with non-empty `audit_label`. Keep original label, audited label, reason, and reviewer for traceability.
