#!/usr/bin/env python3
"""Write evolution results back to Hindsight for trend tracking."""
import json, sys, requests, os
from pathlib import Path

HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL", "http://192.168.50.225:8788/v1/default/banks/hermes-clean")

def writeback(results_dir: str):
    """Read metrics.json from results dir and retain to Hindsight."""
    metrics_file = Path(results_dir) / "metrics.json"
    if not metrics_file.exists():
        print(f"ERROR: {metrics_file} not found")
        return False

    metrics = json.loads(metrics_file.read_text())

    content = f"""Evolution run completed for skill '{metrics.get('skill', 'unknown')}':
- Baseline score: {metrics.get('baseline_score', 'N/A')}
- Evolved score: {metrics.get('evolved_score', 'N/A')}
- Improvement: {metrics.get('improvement', 'N/A')}
- Iterations: {metrics.get('iterations', 'N/A')}
- Optimizer: {metrics.get('optimizer', 'N/A')}
- Eval source: {metrics.get('eval_source', 'N/A')}
"""

    resp = requests.post(
        f"{HINDSIGHT_URL}/retain",
        json={"content": content, "context": f"skill-evolution:{metrics.get('skill', 'unknown')}"},
        timeout=60
    )
    if resp.status_code == 200:
        print(f"✓ Written to Hindsight: {metrics.get('skill')} +{metrics.get('improvement', 'N/A')}")
        return True
    else:
        print(f"✗ Hindsight retain failed: {resp.status_code} {resp.text[:200]}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hindsight_writeback.py <results_dir>")
        sys.exit(1)
    writeback(sys.argv[1])