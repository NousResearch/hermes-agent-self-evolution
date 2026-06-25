# Reports

Validation reports and research findings for Agent Self-Evolution. Each phase
validation report is a PDF rendered from a committed `*_prose.yaml`; every number in a
findings report is checked against its source JSON at render time (the renderer refuses
to emit a PDF if any number drifts).

## Phase validation reports

The five-phase series, one PDF per phase (rendered in the same house style):

| Phase | Report | What it validates |
|-------|--------|-------------------|
| 1 | [phase1_validation_report.pdf](phase1_validation_report.pdf) | Skill files (`SKILL.md`) via DSPy + GEPA |
| 2 | [phase2_validation_report.pdf](phase2_validation_report.pdf) | Tool descriptions + the dual-signal deploy gate |
| 3 | [phase3_validation_report.pdf](phase3_validation_report.pdf) | System-prompt sections via splice-and-restore |
| 4 | [phase4_validation_report.pdf](phase4_validation_report.pdf) | Tool implementation code via iterative test-feedback repair |
| 5 | [phase5_validation_report.pdf](phase5_validation_report.pdf) | Continuous-improvement loop via a propose-only triage sentinel |

Phases 1–3 render via `generate_report.py` (numbers pulled from a run dir); Phases 4–5
render via `generate_findings_report.py` (a cross-campaign synthesis with a
provenance-checked `phase{4,5}_prose.yaml`).

## Findings (deep dives)

The evidence behind the phase reports and the campaign's headline result — that artifact
text moves a capable agent only where it carries non-inferable signal, while code repair
behind an executable oracle gets real traction:

- [asymmetry_findings.md](asymmetry_findings.md) ([PDF](asymmetry_report.pdf)) — the oracle asymmetry; the full Phase 4 deep dive.
- [asymmetry_headroom_experiments_findings.md](asymmetry_headroom_experiments_findings.md) — the GREEN / NULL / FRONTIER regime map.
- [darwinian_evolver_evaluation.md](darwinian_evolver_evaluation.md) — population search vs. best-of-N (why the value is the signal + gate, not the search).
- [asymmetry_gate_gaming_findings.md](asymmetry_gate_gaming_findings.md) — adversarial audit: does the oracle gate resist input-hardcoding?
- [asymmetry_supply_generalization_findings.md](asymmetry_supply_generalization_findings.md) — does the bug supply port to a second repo? (the Phase 5 supply caveat).
- [frontier_quality_findings.md](frontier_quality_findings.md) — why open-ended generative quality resists measurement.
- [mcp_description_findings.md](mcp_description_findings.md) — tool-description coupling on novel, non-inferable tools.
- [skill_body_coupling_findings.md](skill_body_coupling_findings.md) — where a thick skill body does drive behavior (output conventions).
- [reasoning_quality_saturation_findings.md](reasoning_quality_saturation_findings.md) — reasoning methodology is internalized on capable agents (no headroom).
- [saturation_calibration_findings.md](saturation_calibration_findings.md) · [calibration_findings.md](calibration_findings.md) — noise-floor calibration behind the pre-flight gate.

## Regenerating

```bash
# Phase 1–3 (run-dir renderer)
python generate_report.py --run output/<artifact>/<ts>/ \
    --prose reports/phase1_prose.yaml --out reports/phase1_validation_report.pdf

# Phase 4–5 and findings (prose-driven, provenance-checked)
python generate_findings_report.py \
    --prose reports/phase4_prose.yaml --out reports/phase4_validation_report.pdf
```

Editorial content lives in the `*_prose.yaml` files; measured numbers live in the run
dirs and the committed result JSONs. `reports/_*` files are local audit scratch (gitignored).
