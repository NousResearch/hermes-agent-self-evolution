"""SWE-bench Lite external-validity campaign entry. Loads single-file Lite
organisms, validity-gates each, and drives the reused run_campaign with a
SWE-bench organism_runner: repair in a warm SWEbenchEnv across `seeds`, verify via
run_code_oracle_gate (bug_tests=F2P, pass_to_pass=P2P). Drops + per-instance
characterization (LOC/hunks/reason) are written for report.py."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from evolution.code.campaign import PROPOSER_MAX_TOKENS, Skip, run_campaign
from evolution.code.harvest import stratify
from evolution.code.campaign_report import OrganismResult
from evolution.code.gate import run_code_oracle_gate
from evolution.code.repair import RepairEngine, build_dspy_proposer
from evolution.code.swebench.env import SWEbenchEnv
from evolution.code.swebench.loader import load_single_file_lite
from evolution.code.swebench.validity import Drop, Organism, validate_instance
from evolution.core.hermes_provider import resolve_default_lm

console = Console()
log = logging.getLogger(__name__)


@dataclass
class _Cand:
    fix_sha: str        # = instance_id; lets the reused run_campaign loop key on it unchanged
    tool_path: str      # = repo; reused as the cluster key
    instance: object


def instances_to_candidates(instances) -> list[_Cand]:
    return [_Cand(fix_sha=i.instance_id, tool_path=i.repo, instance=i) for i in instances]


def _arm_cost_ledger(max_cost_usd: Optional[float]) -> None:
    """Wire the cost ledger + litellm callbacks and set the ceiling. run_campaign only
    does this inside its (skipped) organism_runner-is-None branch, so on the injected
    path --max-cost-usd is dead unless armed here. The BaseLM guard then raises
    CostCeilingExceeded, which run_campaign catches."""
    import dspy  # noqa: PLC0415
    from evolution.core.lm_timing_callback import (  # noqa: PLC0415
        COST_LEDGER, LMTimingCallback,
        register_litellm_cost_callback, register_litellm_failure_callback)
    dspy.configure(callbacks=[LMTimingCallback()])
    register_litellm_cost_callback()
    register_litellm_failure_callback()
    COST_LEDGER.reset()
    if max_cost_usd is not None:
        COST_LEDGER.set_ceiling(max_cost_usd)


def build_swebench_runner(*, seeds: int, max_rounds: int, proposer_model: Optional[str],
                          characterization: list, max_cost_usd: Optional[float]):
    import dspy  # noqa: PLC0415
    _arm_cost_ledger(max_cost_usd)
    rlm = resolve_default_lm(role="optimizer", explicit_model=proposer_model)
    lm = dspy.LM(rlm.model, **rlm.lm_kwargs, temperature=0.7, max_tokens=PROPOSER_MAX_TOKENS)
    engine = RepairEngine(build_dspy_proposer(lm), max_rounds=max_rounds)

    def runner(c: _Cand):
        inst = c.instance
        try:
            env = SWEbenchEnv.create(inst)
        except Exception as e:  # noqa: BLE001 — build/pull failure → drop
            characterization.append({"instance_id": inst.instance_id, "reason": "build_failed", "err": str(e)[:300]})
            return Skip("build_failed")
        try:
            env.assert_authoritative(inst.repo.split("/")[-1])
            v = validate_instance(inst, env)
            if isinstance(v, Drop):
                characterization.append({"instance_id": inst.instance_id, "reason": v.reason,
                                         "gold_loc": v.gold_loc, "gold_hunks": v.gold_hunks})
                return Skip(v.reason)
            org: Organism = v
            characterization.append({"instance_id": inst.instance_id, "reason": "kept",
                                     "gold_loc": org.gold_loc, "gold_hunks": org.gold_hunks,
                                     "emulated": env.emulated})
            seed_results = []
            for _ in range(seeds):
                env.reset_file(inst.gold_file)
                repair = engine.repair(env, inst.gold_file, org.bug_tests)
                gate = run_code_oracle_gate(
                    env, tool_relpath=inst.gold_file, test_relpath="(swebench)",
                    bug_tests=org.bug_tests, oracle_failures=org.oracle_failures,
                    base_src=org.base_src, repair_result=repair, pass_to_pass=inst.pass_to_pass)
                seed_results.append(bool(gate.deploy))
            return OrganismResult(tool=inst.repo, fix_sha=inst.instance_id, seeds=seed_results)
        finally:
            env.destroy()
    return runner


@click.command()
@click.option("--max-organisms", default=10, type=click.IntRange(min=1))
@click.option("--stages", default="10,30,50")
@click.option("--seeds", default=3, type=click.IntRange(min=1))
@click.option("--repair-rounds", default=5, type=click.IntRange(min=1))
@click.option("--max-cost-usd", required=True, type=click.FloatRange(min=0.0),
              help="REQUIRED — abort when cumulative LM cost exceeds this.")
@click.option("--limit-instances", default=None, type=int)
@click.option("--exclude-repos", default="",
              help="Comma-separated repos to skip (e.g. arch-unreachable: pydata/xarray,scikit-learn/scikit-learn).")
@click.option("--proposer-model", default=None)
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, path_type=Path))
def main(max_organisms, stages, seeds, repair_rounds, max_cost_usd, limit_instances,
         exclude_repos, proposer_model, output_dir):
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(output_dir or Path("output") / "swebench_campaign" /
                      datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = load_single_file_lite(limit=limit_instances)
    excluded = {r.strip() for r in exclude_repos.split(",") if r.strip()}
    if excluded:
        instances = [i for i in instances if i.repo not in excluded]
    # Round-robin across repos so the first N valid organisms span many repos
    # (Lite's dataset order is repo-grouped) — cluster-honest, like the Hermes campaign.
    candidates = stratify(instances_to_candidates(instances), max_per_tool=None)
    console.print(f"[bold]swebench external-validity[/bold] — {len(candidates)} single-file Lite "
                  f"organisms across {len({c.tool_path for c in candidates})} repos (stratified)")
    characterization: list = []
    runner = build_swebench_runner(seeds=seeds, max_rounds=repair_rounds,
                                   proposer_model=proposer_model, characterization=characterization,
                                   max_cost_usd=max_cost_usd)
    run_campaign(output_dir, output_dir=output_dir, max_organisms=max_organisms,
                 stages=tuple(int(s) for s in stages.split(",")), seeds=seeds, max_rounds=repair_rounds,
                 max_cost_usd=max_cost_usd, candidates=candidates,
                 organism_runner=runner)
    from evolution.core.lm_timing_callback import COST_LEDGER  # noqa: PLC0415
    (output_dir / "cost.json").write_text(json.dumps(COST_LEDGER.summary(), indent=2))
    (output_dir / "characterization.json").write_text(json.dumps(characterization, indent=2))
    console.print(f"  characterization → {output_dir/'characterization.json'} "
                  f"({sum(1 for r in characterization if r['reason']=='kept')} kept, "
                  f"{sum(1 for r in characterization if r['reason']!='kept')} dropped)")
    from evolution.code.swebench import report as _report  # noqa: PLC0415
    rep = _report.build(output_dir)
    console.print(f"  [bold]deploy-reachable[/bold] {rep['deploy_reachable']} | "
                  f"kept_difficulty {rep['kept_difficulty']} vs hermes {rep['hermes_difficulty']}")
    console.print(f"  [yellow]{rep['interpretation_guard']}[/yellow]")


if __name__ == "__main__":
    main()
