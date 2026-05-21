#!/usr/bin/env python3
"""Nightly profile-aware controller for Hermes Agent self-evolution.

This script is intentionally cron-safe:
- Python entrypoint, so Hermes cron can run it directly.
- No bash `source` of profile .env files.
- Silent stdout on ordinary skip nights.
- Never applies evolved skills; it only writes artifacts and reports candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = DEFAULT_REPO_ROOT / "evolution-policy.yaml"
DEFAULT_PYTHON = Path("/usr/bin/python")


@dataclass(frozen=True)
class Target:
    profile: str
    profile_root: Path
    skill: str
    domain: str
    title: str

    @property
    def key(self) -> str:
        return f"{self.profile}:{self.skill}"


def now_local() -> datetime:
    return datetime.now().astimezone()


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Policy must be a YAML mapping: {path}")
    return data


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_runs": {}, "run_history": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + f".bad-{int(time.time())}")
        shutil.copy2(path, backup)
        return {"last_runs": {}, "run_history": [], "state_backup": str(backup)}
    data.setdefault("last_runs", {})
    data.setdefault("run_history", [])
    return data


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parse_env_file(path: Path, keys: set[str]) -> dict[str, str]:
    """Parse only selected KEY=value entries without shell evaluation."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in keys:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            values[key] = value
    return values


def build_env(policy: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env_cfg = policy.get("environment", {}) or {}
    keys = set(env_cfg.get("keys", []))
    loaded: dict[str, str] = {}
    for item in env_cfg.get("env_files", []) or []:
        loaded.update(parse_env_file(Path(item).expanduser(), keys))
    # Inherited environment wins; parsed files fill gaps only.
    for key, value in loaded.items():
        env.setdefault(key, value)

    for target_key, source_key in (env_cfg.get("alias_keys", {}) or {}).items():
        if not env.get(target_key) and env.get(source_key):
            env[target_key] = env[source_key]

    # LiteLLM backoff: set before importing DSPy/LiteLLM in the child process.
    env.setdefault("INITIAL_RETRY_DELAY", "5")
    env.setdefault("MAX_RETRY_DELAY", "60")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def required_key_for_model(model: str) -> str | None:
    if model.startswith("openai/"):
        return "OPENAI_API_KEY"
    if model.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    if model.startswith("anthropic/"):
        return "ANTHROPIC_API_KEY"
    return None


def validate_model_env(policy: dict[str, Any], env: dict[str, str]) -> list[str]:
    models = policy.get("models", {}) or {}
    missing: list[str] = []
    for model in [models.get("eval", ""), models.get("optimizer", "")]:
        key = required_key_for_model(str(model))
        if key and not env.get(key) and key not in missing:
            missing.append(key)
    return missing


def skill_path(profile_root: Path, skill: str) -> Path | None:
    skills_dir = profile_root / "skills"
    if not skills_dir.exists():
        return None
    for candidate in skills_dir.rglob("SKILL.md"):
        if candidate.parent.name == skill:
            return candidate
    needle_a = f"name: {skill}"
    needle_b = f'name: "{skill}"'
    for candidate in skills_dir.rglob("SKILL.md"):
        try:
            head = candidate.read_text(errors="replace")[:600]
        except OSError:
            continue
        if needle_a in head or needle_b in head:
            return candidate
    return None


def build_targets(policy: dict[str, Any], include_disabled: bool = False) -> list[Target]:
    profiles = policy.get("profiles", {}) or {}
    skills_by_domain = policy.get("skills", {}) or {}
    targets: list[Target] = []
    for profile_name, profile_cfg in profiles.items():
        if not include_disabled and not profile_cfg.get("enabled", False):
            continue
        profile_root = Path(profile_cfg["root"]).expanduser()
        title = str(profile_cfg.get("title", profile_name))
        for domain in profile_cfg.get("domains", []) or []:
            for skill in skills_by_domain.get(domain, []) or []:
                if skill_path(profile_root, skill):
                    targets.append(Target(profile_name, profile_root, skill, domain, title))
    return targets


def find_requested_target(policy: dict[str, Any], profile: str, skill: str) -> Target:
    profiles = policy.get("profiles", {}) or {}
    profile_cfg = profiles.get(profile)
    if not profile_cfg:
        raise ValueError(f"Unknown profile: {profile}")
    profile_root = Path(profile_cfg["root"]).expanduser()
    found = skill_path(profile_root, skill)
    if not found:
        raise ValueError(f"Skill '{skill}' not found in profile '{profile}' at {profile_root / 'skills'}")
    domain = str(profile_cfg.get("domain", "manual"))
    return Target(profile, profile_root, skill, domain, str(profile_cfg.get("title", profile)))


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def should_run_today(policy: dict[str, Any], state: dict[str, Any], now: datetime) -> tuple[bool, str]:
    cadence = policy.get("cadence", {}) or {}
    today = DAY_NAMES[now.weekday()]
    allowed_days = cadence.get("run_days", []) or []
    if allowed_days and today not in allowed_days:
        return False, f"skip: {today} not in run_days"

    max_runs = int(cadence.get("max_runs_per_week", 0) or 0)
    if max_runs > 0:
        week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        count = 0
        for item in state.get("run_history", []) or []:
            ts = parse_iso(item.get("timestamp")) if isinstance(item, dict) else None
            if ts and ts >= week_start and item.get("attempted", False):
                count += 1
        if count >= max_runs:
            return False, f"skip: max_runs_per_week reached ({count}/{max_runs})"
    return True, "run"


def select_target(policy: dict[str, Any], state: dict[str, Any], now: datetime) -> Target | None:
    targets = build_targets(policy, include_disabled=False)
    if not targets:
        return None

    min_days = int((policy.get("cadence", {}) or {}).get("min_days_between_same_target", 0) or 0)
    cooldown_cutoff = now - timedelta(days=min_days)
    last_runs = state.get("last_runs", {}) or {}

    eligible: list[tuple[datetime, Target]] = []
    for target in targets:
        last = parse_iso(last_runs.get(target.key))
        if last and min_days and last > cooldown_cutoff:
            continue
        eligible.append((last or datetime.min.replace(tzinfo=timezone.utc), target))

    if not eligible:
        return None
    eligible.sort(key=lambda pair: (pair[0], pair[1].profile, pair[1].skill))
    return eligible[0][1]


def newest_success_artifact(repo_root: Path, skill: str, started_at: float) -> Path | None:
    base = repo_root / "output" / skill
    if not base.exists():
        return None
    candidates = [p for p in base.glob("20*") if p.is_dir() and (p / "metrics.json").exists()]
    candidates = [p for p in candidates if p.stat().st_mtime >= started_at - 2]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def failed_variant(repo_root: Path, skill: str, started_at: float) -> Path | None:
    path = repo_root / "output" / skill / "evolved_FAILED.md"
    if path.exists() and path.stat().st_mtime >= started_at - 2:
        return path
    return None


def copy_artifact(repo_root: Path, target: Target, timestamp: str, started_at: float) -> Path | None:
    profile_base = repo_root / "output" / target.profile / target.skill
    profile_base.mkdir(parents=True, exist_ok=True)

    success = newest_success_artifact(repo_root, target.skill, started_at)
    if success:
        dest = profile_base / success.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(success, dest)
        return dest

    failed = failed_variant(repo_root, target.skill, started_at)
    if failed:
        dest = profile_base / timestamp
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(failed, dest / "evolved_FAILED.md")
        return dest

    return None


def read_metrics(artifact: Path | None) -> dict[str, Any] | None:
    if not artifact:
        return None
    path = artifact / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _has_golden_dataset(path: Path) -> bool:
    return (path / "train.jsonl").exists() or (path / "golden.jsonl").exists()


def resolve_eval_dataset_args(repo_root: Path, target: Target, eval_cfg: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve eval source and optional dataset path for a target.

    `source: auto` prefers a hand-curated golden dataset when present, then
    falls back to synthetic generation. Explicit `synthetic` remains synthetic
    even if a golden set exists so operators can force exploration runs.
    """
    requested = str(eval_cfg.get("source", "auto") or "auto")
    dataset_path = eval_cfg.get("dataset_path")
    if dataset_path:
        return requested if requested != "auto" else "golden", str(Path(str(dataset_path)).expanduser())

    if requested in {"auto", "golden"}:
        golden_root = Path(str(eval_cfg.get("golden_root", repo_root / "datasets" / "golden" / "skills"))).expanduser()
        golden_path = golden_root / target.skill
        if _has_golden_dataset(golden_path):
            return "golden", str(golden_path)
        if requested == "golden":
            return "golden", str(golden_path)

    if requested == "auto":
        return "synthetic", None
    return requested, None


def summarize(target: Target, status: str, artifact: Path | None, log: Path, metrics: dict[str, Any] | None) -> str:
    lines = [f"Self-evolution: {target.profile}:{target.skill}", f"status: {status}"]
    if metrics:
        baseline = metrics.get("baseline_score")
        evolved = metrics.get("evolved_score")
        improvement = metrics.get("improvement")
        if isinstance(baseline, (int, float)) and isinstance(evolved, (int, float)) and isinstance(improvement, (int, float)):
            pct = improvement / max(0.001, float(baseline)) * 100
            lines.append(f"score: {baseline:.3f} -> {evolved:.3f} ({improvement:+.3f}, {pct:+.1f}%)")
        if metrics.get("constraints_passed") is True:
            lines.append("gate: constraints-passed")
        elif metrics.get("constraints_passed") is False:
            lines.append("gate: constraints-failed")
    elif status == "constraints-failed":
        lines.append("gate: constraints-failed")

    if status == "candidate":
        lines.append("review: candidate-review-needed")
    elif status == "constraints-failed":
        lines.append("review: rejected")
    elif status in {"no-improvement", "below-threshold"}:
        lines.append("review: no-application")

    if artifact:
        lines.append(f"artifact: {artifact}")
    lines.append(f"log: {log}")
    lines.append("applied: no")
    return "\n".join(lines)


def run_evolution(repo_root: Path, policy: dict[str, Any], env: dict[str, str], target: Target) -> tuple[str, Path | None, Path, dict[str, Any] | None, int]:
    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"nightly-evolve-{target.profile}-{target.skill}-{timestamp}.log"

    budget = policy.get("budget", {}) or {}
    models = policy.get("models", {}) or {}
    eval_cfg = policy.get("evaluation", {}) or {}
    timeout = int(budget.get("timeout_seconds", 1700) or 1700)
    iterations = str(int(budget.get("iterations", 5) or 5))
    eval_source, dataset_path = resolve_eval_dataset_args(repo_root, target, eval_cfg)
    eval_model = str(models.get("eval", "openai/gpt-5.4-mini"))
    optimizer_model = str(models.get("optimizer", "openai/gpt-5.4-mini"))

    child_env = env.copy()
    child_env["HERMES_AGENT_REPO"] = str(target.profile_root)

    cmd = [
        str(DEFAULT_PYTHON),
        "-u",
        "-m",
        "evolution.skills.evolve_skill",
        "--skill",
        target.skill,
        "--iterations",
        iterations,
        "--eval-source",
        eval_source,
        "--eval-model",
        eval_model,
        "--optimizer-model",
        optimizer_model,
        "--hermes-repo",
        str(target.profile_root),
    ]
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])

    started_at = time.time()
    with log_path.open("w") as log:
        log.write(f"=== Nightly Skill Evolution ===\n")
        log.write(f"Profile: {target.profile} ({target.title})\n")
        log.write(f"Skill: {target.skill}\n")
        log.write(f"Time: {datetime.now().astimezone().isoformat()}\n")
        log.write(f"Command: {' '.join(cmd)}\n\n")
        log.flush()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout}s\n")
            returncode = 124

    artifact = copy_artifact(repo_root, target, timestamp, started_at)
    metrics = read_metrics(artifact)
    min_improvement = float((policy.get("gates", {}) or {}).get("min_improvement", 0.0) or 0.0)

    if returncode != 0:
        status = f"failed rc={returncode}"
    elif metrics:
        improvement = float(metrics.get("improvement", 0.0) or 0.0)
        if improvement >= min_improvement:
            status = "candidate"
        elif improvement > 0:
            status = "below-threshold"
        else:
            status = "no-improvement"
    elif artifact and (artifact / "evolved_FAILED.md").exists():
        status = "constraints-failed"
    else:
        status = "completed-no-metrics"

    return status, artifact, log_path, metrics, returncode


def append_run_record(repo_root: Path, state: dict[str, Any], target: Target, status: str, artifact: Path | None, log: Path, attempted: bool) -> None:
    ts = now_local().isoformat(timespec="seconds")
    record = {
        "timestamp": ts,
        "target": target.key,
        "profile": target.profile,
        "skill": target.skill,
        "status": status,
        "attempted": attempted,
        "artifact": str(artifact) if artifact else None,
        "log": str(log),
    }
    state.setdefault("run_history", []).append(record)
    if attempted:
        state.setdefault("last_runs", {})[target.key] = ts
    runs_dir = repo_root / "output" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{ts.replace(':', '').replace('-', '').replace('+', '_')}-{target.profile}-{target.skill}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile-aware Hermes self-evolution nightly controller")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Self-evolution repo root")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="Policy YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print selected target without running GEPA")
    parser.add_argument("--force", action="store_true", help="Ignore cadence and weekly limits")
    parser.add_argument("--profile", help="Force/manual profile target")
    parser.add_argument("--skill", help="Force/manual skill target")
    parser.add_argument("--print-skip", action="store_true", help="Print skip reason instead of staying silent")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_policy(policy_path)
    state_path = Path(policy.get("state_file", "/home/w0lf/.hermes/skill-evolution-state.json")).expanduser()
    state = read_state(state_path)
    env = build_env(policy)
    now = now_local()

    manual = bool(args.profile or args.skill)
    if manual and not (args.profile and args.skill):
        print("Manual runs require both --profile and --skill", file=sys.stderr)
        return 2

    if manual:
        try:
            target = find_requested_target(policy, args.profile, args.skill)  # type: ignore[arg-type]
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        if not args.force:
            should_run, reason = should_run_today(policy, state, now)
            if not should_run:
                if args.dry_run or args.print_skip:
                    print(reason)
                return 0
        target = select_target(policy, state, now)
        if not target:
            if args.dry_run or args.print_skip:
                print("skip: no eligible targets")
            return 0

    missing = validate_model_env(policy, env)
    if missing:
        print(f"config-error: missing required environment key(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"dry-run: would evolve {target.profile}:{target.skill}")
        print(f"profile_root: {target.profile_root}")
        print(f"skill_path: {skill_path(target.profile_root, target.skill)}")
        print(f"policy: {policy_path}")
        print(f"state: {state_path}")
        return 0

    status, artifact, log, metrics, returncode = run_evolution(repo_root, policy, env, target)
    append_run_record(repo_root, state, target, status, artifact, log, attempted=True)
    write_state(state_path, state)
    print(summarize(target, status, artifact, log, metrics))
    # Evolution failures are expected candidate outcomes and already reported above.
    # Keep cron no_agent delivery in the normal stdout path instead of triggering a
    # generic scheduler error alert. Configuration/selection errors still return
    # non-zero earlier in main().
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
