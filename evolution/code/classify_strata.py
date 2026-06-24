"""Bet #2: three-stratum classification of harvested bugs to size the PBT
(oracle-manufacture) slice. pure-input is MEASURED (armB_dr); pure-contract is
JUDGED with a cited external-spec source; state (vacuous return) is excluded from
the contract-vs-reference ratio."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Source repo to read fix diffs from — set $HERMES_REPO (no hardcoded path).
REPO = Path(_p).expanduser() if (_p := os.environ.get("HERMES_REPO")) else None
LEAK = Path("reports/asymmetry_leakage_check.json")
FUZZ = Path("reports/asymmetry_fuzz_differential.json")


def _git(*args) -> str:
    if REPO is None:
        return ""  # no source repo configured ($HERMES_REPO unset) — diffs unavailable
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def load_strata_worksheet() -> list[dict]:
    """One body-blind worksheet row per leakage organism: the MEASURED armB_dr
    (pure-input signal), the fuzz verdict (is it fuzzable?), and the parent->fix
    diff for body-blind judgement. stratum/contract_source start empty (human-filled)."""
    leak = json.loads(LEAK.read_text())["results"]
    fuzz = {r["tool"]: r for r in json.loads(FUZZ.read_text())["report"]} if FUZZ.exists() else {}
    rows = []
    for r in leak:
        tool, fix = r["tool"], r["fix"]
        fz = fuzz.get(tool, {})
        rows.append({
            "tool": tool, "fix": fix,
            "armB_dr": r["armB_dr"],                          # MEASURED pure-input signal
            "fuzz_verdict": fz.get("verdict", "unknown"),
            "fuzzable": fz.get("verdict") in ("GENERALIZES", "DIVERGES"),
            "diff": _git("show", fix, "--", tool),
            "stratum": "",          # human-filled: pure-input | pure-contract | state
            "contract_source": "",  # REQUIRED if pure-contract: cited docstring/grammar/type
            "notes": "",
        })
    return rows


def validate_strata(rows: list[dict]) -> dict:
    """Falsifiability guard + contract-vs-reference ratio.

    - pure-contract REQUIRES a non-empty contract_source (else an error).
    - if notes say the only generalization support is the fuzz differential
      (which fuzzed AGAINST the implementation), reclassify pure-contract -> pure-input.
    - state organisms and errored rows (pure-contract with no cited source) are
      EXCLUDED from the contract-vs-reference denominator.
    """
    errors, cleaned, errored_tools = [], [], set()
    for r in rows:
        r = dict(r)
        if r.get("stratum") == "pure-contract":
            if "fuzz-only" in r.get("notes", "").lower():
                r["stratum"] = "pure-input"
                r["notes"] = (r.get("notes", "") + " [reclassified: fuzz-only -> input]").strip()
            elif not r.get("contract_source", "").strip():
                errors.append(
                    f"{r['tool']}@{r['fix']}: pure-contract requires a cited contract_source"
                )
                errored_tools.add(r["tool"])
        cleaned.append(r)

    # Errored rows cannot be counted in either contract or input bucket.
    ratio_rows = [r for r in cleaned if r["tool"] not in errored_tools]
    non_state = [r for r in ratio_rows if r.get("stratum") in ("pure-input", "pure-contract")]
    contract = [r for r in non_state if r.get("stratum") == "pure-contract"]
    state_excl = [r for r in ratio_rows if r.get("stratum") not in ("pure-input", "pure-contract")]
    ratio = {
        "pure_contract": len(contract),
        "pure_input": len(non_state) - len(contract),
        "non_state_total": len(non_state),
        "state_excluded": len(state_excl),
    }
    return {"errors": errors, "rows": cleaned, "ratio": ratio}
