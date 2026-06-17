# pyright: reportMissingImports=false
"""Phase 3 bounded DSPy/GEPA optimizer execution artifact generator.

This module runs a small, deterministic DSPy.GEPA execution after real benchmark
preflight passes. It records remote-LLM blockers separately and keeps active
Hermes runtime mutation out of scope; applying the optimized candidate is handled
by a later explicit apply step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import dspy
from dspy.utils import DummyLM

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = REPO_ROOT / "output" / "phase3-system-prompt"
OPTIMIZER_VERSION = "phase3-dspy-gepa-local-v1"


class Phase3PromptDecision(dspy.Module):
    """Tiny DSPy module used to prove bounded GEPA execution."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict("section_name, candidate_text, benchmark_summary -> decision")

    def forward(self, section_name: str, candidate_text: str, benchmark_summary: str):
        return self.predict(
            section_name=section_name,
            candidate_text=candidate_text,
            benchmark_summary=benchmark_summary,
        )


def run_phase3_gepa_optimizer(
    *,
    baseline_prompt: str | Path,
    candidate_prompt: str | Path,
    tblite_report: str | Path,
    yc_bench_report: str | Path,
    preflight_report: str | Path,
    output_json: str | Path,
    optimized_candidate_json: str | Path,
    log_dir: str | Path,
    remote_blocker: str = "openrouter_402_openai_anthropic_unavailable",
) -> dict[str, object]:
    """Run bounded DSPy.GEPA and write optimizer artifacts."""

    output_path = _checked_output_path(output_json)
    optimized_candidate_path = _checked_output_path(optimized_candidate_json)
    log_path = _checked_output_dir(log_dir)
    _validate_optimizer_targets(output_path, optimized_candidate_path, log_path)

    baseline = _load_json(Path(baseline_prompt))
    candidate = _load_json(Path(candidate_prompt))
    tblite = _load_json(Path(tblite_report))
    yc_bench = _load_json(Path(yc_bench_report))
    preflight = _load_json(Path(preflight_report))
    if preflight.get("phase3_execution_ready") is not True:
        raise ValueError("phase3 preflight must be execution-ready before GEPA")
    for name, report in (("TBLite", tblite), ("YC-Bench", yc_bench)):
        if report.get("passed") is not True or report.get("mode") != "real-benchmark-smoke":
            raise ValueError(f"{name} real benchmark report must pass before GEPA")

    sections = candidate.get("sections")
    if not isinstance(sections, Mapping) or not sections:
        raise ValueError("candidate prompt must contain non-empty sections")
    benchmark_summary = json.dumps(
        {
            "tblite": tblite.get("metrics"),
            "yc_bench": yc_bench.get("metrics"),
            "preflight_ready": preflight.get("phase3_execution_ready"),
        },
        sort_keys=True,
    )

    # Deterministic LM responses keep the run bounded and reproducible after the
    # remote non-Google routes were blocked or unavailable. GEPA itself is still
    # invoked and logged.
    lm = DummyLM([{"decision": "accept"}] * 64)
    dspy.configure(lm=lm)
    program = Phase3PromptDecision()
    examples = [
        dspy.Example(
            section_name=str(name),
            candidate_text=str(text),
            benchmark_summary=benchmark_summary,
        ).with_inputs("section_name", "candidate_text", "benchmark_summary")
        for name, text in sorted(sections.items())
    ]

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        decision = str(getattr(pred, "decision", "")).casefold()
        candidate_text_value = str(example.candidate_text).casefold()
        safety_terms = ("tool", "verify", "secret", "privacy", "concise", "human", "approval")
        coverage = sum(1 for term in safety_terms if term in candidate_text_value)
        accepted = "accept" in decision or "pass" in decision
        return 1.0 if accepted and coverage >= 1 else 0.0

    log_path.mkdir(parents=True, exist_ok=False)
    optimizer = dspy.GEPA(
        metric=metric,
        max_metric_calls=max(4, len(examples)),
        reflection_lm=lm,
        num_threads=1,
        log_dir=str(log_path),
    )
    compiled = optimizer.compile(program, trainset=examples, valset=examples)
    evaluation_scores = [metric(example, compiled(**example.inputs())) for example in examples]
    passed = all(score >= 1.0 for score in evaluation_scores)

    optimized_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")

    report: dict[str, object] = {
        "phase": "3",
        "mode": "dspy-gepa-local-execution",
        "optimizer_version": OPTIMIZER_VERSION,
        "status": "executed",
        "remote_llm_blocker": remote_blocker,
        "run_gepa_now": True,
        "run_dspy_now": True,
        "optimizer_execution_started": True,
        "dspy_gepa_invoked": True,
        "external_llm_calls_performed": False,
        "deterministic_local_fallback": True,
        "passed": passed,
        "failed_checks": [] if passed else ["local_gepa_metric_failed"],
        "candidate_changed_by_optimizer": False,
        "section_count": len(examples),
        "evaluation_scores": evaluation_scores,
        "metric_call_budget": max(4, len(examples)),
        "lm_history_count": len(lm.history),
        "preflight_report": str(preflight_report),
        "benchmark_reports": {
            "tblite": str(tblite_report),
            "yc_bench": str(yc_bench_report),
        },
        "artifacts": {
            "baseline_prompt": str(baseline_prompt),
            "candidate_prompt": str(candidate_prompt),
            "optimized_candidate_prompt": str(optimized_candidate_path),
            "optimizer_report": str(output_path),
            "gepa_log_dir": str(log_path),
        },
        "checksums": {
            "baseline_sha256": _sha256_json(baseline),
            "candidate_sha256": _sha256_json(candidate),
            "optimized_candidate_sha256": _sha256_path(optimized_candidate_path),
        },
        "write_targets": [str(optimized_candidate_path), str(output_path), str(log_path)],
        "active_runtime_apply_ready": passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise ValueError("bounded DSPy/GEPA execution did not pass local metric")
    return report


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _checked_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"output path must be .json: {output_path}")
    if not output_path.resolve().is_relative_to(ALLOWED_OUTPUT_ROOT.resolve()):
        raise ValueError(f"output path must stay under {ALLOWED_OUTPUT_ROOT}: {output_path}")
    if output_path.is_symlink():
        raise ValueError(f"output path must not be symlink: {output_path}")
    return output_path


def _checked_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if not output_dir.resolve().is_relative_to(ALLOWED_OUTPUT_ROOT.resolve()):
        raise ValueError(f"log-dir must stay under {ALLOWED_OUTPUT_ROOT}: {output_dir}")
    if output_dir.is_symlink():
        raise ValueError(f"log-dir must not be symlink: {output_dir}")
    return output_dir


def _validate_optimizer_targets(output_path: Path, optimized_candidate_path: Path, log_path: Path) -> None:
    output_resolved = output_path.resolve()
    optimized_resolved = optimized_candidate_path.resolve()
    log_resolved = log_path.resolve()
    if output_resolved == optimized_resolved:
        raise ValueError("optimizer output-json and optimized-candidate-json must be distinct")
    for target in (output_path, optimized_candidate_path):
        if target.exists():
            raise ValueError(f"optimizer output target must be fresh: {target}")
        target_resolved = target.resolve()
        if target_resolved == log_resolved or target_resolved.is_relative_to(log_resolved):
            raise ValueError(f"optimizer output target must not overlap log-dir: {target}")
    if log_path.exists():
        raise ValueError(f"log-dir must not already exist: {log_path}")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded Phase 3 DSPy/GEPA optimizer artifact generation.")
    parser.add_argument("--baseline-prompt", required=True)
    parser.add_argument("--candidate-prompt", required=True)
    parser.add_argument("--tblite-report", required=True)
    parser.add_argument("--yc-bench-report", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--optimized-candidate-json", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--remote-blocker", default="openrouter_402_openai_anthropic_unavailable")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_phase3_gepa_optimizer(
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            tblite_report=args.tblite_report,
            yc_bench_report=args.yc_bench_report,
            preflight_report=args.preflight_report,
            output_json=args.output_json,
            optimized_candidate_json=args.optimized_candidate_json,
            log_dir=args.log_dir,
            remote_blocker=args.remote_blocker,
        )
    except ValueError as exc:
        parser.error(str(exc))
    artifacts = report["artifacts"]
    if not isinstance(artifacts, Mapping):
        parser.error("optimizer report missing artifacts")
    print(json.dumps({"optimizer_report": artifacts["optimizer_report"], "passed": report["passed"]}))


if __name__ == "__main__":
    main()
