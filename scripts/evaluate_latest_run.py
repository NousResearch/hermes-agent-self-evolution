#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUTPUT_ROOT = Path('/Users/kirniy/dev/hermes-agent-self-evolution/output')
MAX_SIZE_GROWTH_RATIO = 0.15
MAX_SUSPICIOUS_IMPROVEMENT = 0.35
MIN_HOLDOUT_EXAMPLES = 3
MIN_TOTAL_EXAMPLES = 10
MAX_RUN_AGE_SECONDS = 8 * 60 * 60


def latest_metrics() -> Path:
    candidates = sorted(OUTPUT_ROOT.glob('*/**/metrics.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit('No metrics.json files found under output/.')
    return candidates[0]


def main() -> int:
    metrics_path = latest_metrics()
    metrics = json.loads(metrics_path.read_text())

    baseline = float(metrics.get('baseline_score', 0.0))
    evolved = float(metrics.get('evolved_score', 0.0))
    improvement = float(metrics.get('improvement', evolved - baseline))
    baseline_size = int(metrics.get('baseline_size', 0))
    evolved_size = int(metrics.get('evolved_size', 0))
    constraints_passed = bool(metrics.get('constraints_passed', False))
    holdout = int(metrics.get('holdout_examples', 0))
    total_examples = int(metrics.get('train_examples', 0)) + int(metrics.get('val_examples', 0)) + holdout

    run_dir = metrics_path.parent
    baseline_skill_path = run_dir / 'baseline_skill.md'
    evolved_skill_path = run_dir / 'evolved_skill.md'

    reasons: list[str] = []
    if not constraints_passed:
        reasons.append('constraints failed')
    if improvement <= 0:
        reasons.append(f'non-positive improvement: {improvement:+.3f}')
    if metrics.get('run_status') not in (None, 'success'):
        reasons.append(f"run status not successful: {metrics.get('run_status')}")
    if baseline_size > 0 and evolved_size > baseline_size * (1 + MAX_SIZE_GROWTH_RATIO):
        reasons.append(f'size growth too large: {baseline_size} -> {evolved_size}')
    if improvement > MAX_SUSPICIOUS_IMPROVEMENT:
        reasons.append(f'suspiciously large improvement: {improvement:+.3f}')
    if holdout < MIN_HOLDOUT_EXAMPLES:
        reasons.append(f'holdout examples too low: {holdout}')
    if total_examples < MIN_TOTAL_EXAMPLES:
        reasons.append(f'total examples too low: {total_examples}')
    if baseline_skill_path.exists() and evolved_skill_path.exists():
        if baseline_skill_path.read_bytes() == evolved_skill_path.read_bytes():
            reasons.append('baseline/evolved skills are byte-identical (metric-gaming suspect)')

    run_age_seconds = time.time() - metrics_path.stat().st_mtime
    if run_age_seconds > MAX_RUN_AGE_SECONDS:
        hours = run_age_seconds / 3600.0
        reasons.append(f'stale run artifact: latest metrics is {hours:.1f}h old')

    print(f'Latest metrics: {metrics_path}')
    print(json.dumps(metrics, indent=2))

    if reasons:
        print('\nREJECTED')
        for reason in reasons:
            print(f'- {reason}')
        return 2

    print('\nACCEPTABLE')
    print(f'- improvement: {improvement:+.3f}')
    print(f'- size delta: {evolved_size - baseline_size:+d}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
