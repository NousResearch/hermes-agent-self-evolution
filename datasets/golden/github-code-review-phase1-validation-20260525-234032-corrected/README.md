# github-code-review Phase 1 validation dataset

Created: 2026-05-25 23:40 KST
Purpose: golden/session-derived holdout for HSE Phase 1 validation packet.

Splits:
- train: 2
- val: 2
- holdout: 8

Notes:
- `rubric_checks` are intentionally extra JSON fields. HSE's EvalExample loader ignores unknown fields; the validation packet script uses them for deterministic document-level paired scoring.
- `session-derived` rows are distilled from SessionDB search patterns around PR/code-review/tool-access requests; no raw secrets or credentials are included.
- This dataset is for validation and gate evidence, not automatic active skill application.

Correction note: cleanup rubric now accepts safe deletion of assistant-created review branches instead of requiring force deletion.
