"""Optionally attempt the top triage candidates with the validated repair loop.

Reuses the campaign's per-organism flow (worktree → repair → oracle gate) verbatim
and annotates each queue row with the verdict. Strictly propose-only: it records
whether the loop *could* repair the bug; it never opens a PR. A human reads the
annotated queue and decides what to deploy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from evolution.code.campaign import Skip, run_organism
from evolution.code.campaign_report import OrganismResult
from evolution.code.harvest import Candidate
from evolution.monitor.sentinel import RepairCandidate


def _build_runner(*, proposer_model, base_python, seeds, max_rounds, max_cost_usd):
    """The real organism runner: sets up the proposer LM + cost tracking exactly
    as the campaign does, and returns a callable (repo, Candidate) -> result."""
    import dspy

    from evolution.code.campaign import PROPOSER_MAX_TOKENS
    from evolution.code.repair import RepairEngine, build_dspy_proposer
    from evolution.core.hermes_provider import resolve_default_lm
    from evolution.core.lm_timing_callback import (
        COST_LEDGER,
        LMTimingCallback,
        register_litellm_cost_callback,
        register_litellm_failure_callback,
    )

    rlm = resolve_default_lm(role="optimizer", explicit_model=proposer_model)
    lm = dspy.LM(rlm.model, **rlm.lm_kwargs, temperature=0.7, max_tokens=PROPOSER_MAX_TOKENS)
    dspy.configure(callbacks=[LMTimingCallback()])
    register_litellm_cost_callback()
    register_litellm_failure_callback()
    COST_LEDGER.reset()
    if max_cost_usd is not None:
        COST_LEDGER.set_ceiling(max_cost_usd)
    engine = RepairEngine(build_dspy_proposer(lm), max_rounds=max_rounds)

    def _run(repo: Path, c: Candidate):
        return run_organism(repo, c, engine, seeds=seeds, base_python=base_python)

    return _run, COST_LEDGER.summary


def attempt_candidates(
    repo: Path,
    candidates: list[RepairCandidate],
    payload: dict,
    *,
    max_cost_usd: Optional[float] = None,
    proposer_model: Optional[str] = None,
    base_python: Optional[str] = None,
    seeds: int = 3,
    max_rounds: int = 5,
    console=None,
    organism_runner=None,
    cost_summary=dict,
) -> None:
    """Attempt each candidate; annotate ``payload['candidates']`` rows in place
    with the verdict. ``organism_runner`` is an injection seam for testing."""
    from evolution.core.lm_timing_callback import CostCeilingExceeded

    if organism_runner is None:
        from evolution.code.worktree import prune_orphan_worktrees  # noqa: PLC0415
        prune_orphan_worktrees(repo)  # self-heal leaks from a prior hard-killed run
        organism_runner, cost_summary = _build_runner(
            proposer_model=proposer_model, base_python=base_python,
            seeds=seeds, max_rounds=max_rounds, max_cost_usd=max_cost_usd)

    by_key = {(r["tool"], r["fix_sha"]): r for r in payload.get("candidates", [])}
    for rc in candidates:
        c = Candidate(rc.tool_path, rc.test_path, rc.fix_sha, rc.parent_sha)
        row = by_key.get((rc.tool_path, rc.fix_sha))
        try:
            res = organism_runner(repo, c)
        except CostCeilingExceeded:
            if row is not None:
                row["attempt"] = {"status": "cost_ceiling"}
            if console is not None:
                console.print("[yellow]cost ceiling reached during attempts[/yellow]")
            break
        if isinstance(res, OrganismResult):
            attempt = {"status": "attempted", "correct_seeds": res.n_correct,
                       "seeds": len(res.seeds), "deploy_reachable": res.deploy_reachable}
        else:
            attempt = {"status": res.reason if isinstance(res, Skip) else "not_valid"}
        if row is not None:
            row["attempt"] = attempt
        if console is not None:
            console.print(f"  attempted {rc.tool_path.split('/')[-1]}@{rc.fix_sha[:8]}: "
                          f"{attempt}")
    payload["cost_summary"] = cost_summary()
