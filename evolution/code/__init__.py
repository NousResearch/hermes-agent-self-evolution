"""Phase 4 — Darwinian code evolution.

Tool source files are wrapped as organisms living in throwaway git worktrees.
Candidates are produced by an LLM mutator, scored by a composite fitness function
(pytest + ruff + bug-repro + signature/registry/error-handling freeze), and gated
through ``evaluate_phase4_gate`` before any patch is surfaced for human review.

Auto-merge is never permitted at this tier; evolved code is emitted as a patch
plus a reproducibility manifest only.
"""
