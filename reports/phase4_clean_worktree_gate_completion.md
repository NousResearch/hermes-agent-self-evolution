# Phase 4 completion — clean-worktree target contract gate

- Created: 2026-06-06 11:52:34 CEST
- Authorization: Sunwoo approved autonomous Phase3+Phase4 completion
- HSE branch: `hse/phase5-continuous-loop-prep`
- HSE pre-fix HEAD: `0e7d748cd86aaba97ac4452a0c63947c0f8cfb93`

## Conclusion

Phase 4 now has a local, verified bug-fix completion packet: a concrete safety bug was reproduced with a RED test, fixed, and verified with targeted, Phase4, and full HSE test gates.

## Known bug fixed

`load_target_spec()` accepted Phase 4 target specs where `hermes_base.require_clean_worktree` was `false` or omitted.

This weakened Phase 4 evidence provenance: code-evolution scaffolds could be built against a dirty Hermes worktree even though Phase 4 mutation/reproduction/benchmark evidence should be anchored to a clean base.

## RED test

Added:

```text
tests/code/test_phase4_target_contract.py::test_target_contract_requires_clean_worktree_gate
```

Before the fix:

```text
FAILED: DID NOT RAISE TargetContractError
```

Command:

```bash
/Users/snw/.hermes/evolution/venvs/self-evolution/bin/python -m pytest tests/code/test_phase4_target_contract.py::test_target_contract_requires_clean_worktree_gate -q
```

## Fix

Changed:

```text
evolution/code/target_contract.py
tests/code/test_phase4_target_contract.py
```

Behavior now enforced:

- `hermes_base.require_clean_worktree` must be exactly `true`.
- missing or `false` values fail closed with `TargetContractError`.
- the report-safe target summary includes `require_clean_worktree` for reviewer visibility.

## Verification

```text
Targeted GREEN: 1 passed in 0.03s
Phase4 code suite: 72 passed in 0.44s
Full HSE tests: 351 passed, 11 warnings in 4.96s
compileall: passed for changed Python files
git diff --check: passed for changed files
DSPy: 3.2.1
```

## Benchmark-hold assessment

No Phase4 runtime benchmark harness exists in this repo. For this bugfix, benchmark-hold is assessed as PASS because:

1. the full HSE suite passed;
2. benchmark-related tests are included in the full suite;
3. no benchmark adapter, scoring, optimizer, or candidate-generation code was changed;
4. the patch only tightens Phase4 target-spec safety metadata.

## Safety boundaries

- Darwinian CLI invoked: no
- external network calls: no
- HSE mutation of Hermes source: no
- active runtime apply by HSE: no
- remote push: no

## Gate result

- Bugs fixed: PASS
- Tests pass: PASS
- Benchmarks hold: PASS
- Phase4 local completion: PASS
