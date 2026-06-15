"""Propose-only self-hosting triage sentinel for the code-evolution loop.

Scans a target repo's recent git stream for repair candidates (the validated
repair loop's supply), classifies and ranks them (dependency-regressions first),
and emits a triage queue with a ready-to-run attempt command. Never auto-evolves
or opens PRs — a human reviews the queue and triggers any deploy. The
continuous-improvement front-end (Phase 5 / roadmap item 11), now with the
validated repair loop as its consumer.
"""
