"""Judge-model bake-off: flash vs pro on the skill-fitness rubric.

Run:  env -u PYTHONPATH -u PYTHONHOME .venv/bin/python judge_bakeoff.py

Judges three things a fitness metric must get right:
  1. DISCRIMINATION — does it separate a good answer from a vague one?
  2. SCEPTICISM  — does it reject a confident answer that cites no evidence?
     (A judge that just rewards fluency would pass this wrongly.)
  3. STABILITY    — same input twice, same verdict? Non-determinism adds noise
     to a holdout of ~13 examples.

Reports raw scores, not a verdict on its own.
"""

import json
import re
import statistics
import urllib.request
from pathlib import Path

CFG = Path.home() / ".codex" / "config.toml"
FLASH = "openrouter/~deepseek/deepseek-flash-latest"
PRO = "openrouter/~deepseek/deepseek-pro-latest"

BASE = "http://127.0.0.1:4222/_codex-router/" + \
    re.search(r"codex-router/([A-Za-z0-9_\-]+)/v1", CFG.read_text()).group(1) + "/v1"

RUBRIC = ("cite file:line, give a PASS/FAIL/UNVERIFIED verdict, "
          "state what was not tested")

PROMPT = (
    'You are a strict grader. Reply ONLY with a JSON object, no prose, no code fence:\n'
    '{{"correctness":<0-1>,"procedure_following":<0-1>,"conciseness":<0-1>,'
    '"feedback":"one sentence"}}\n\n'
    'TASK: {task}\nRUBRIC: {rubric}\nAGENT OUTPUT: {out}'
)

# label, output text, expected band
CASES = [
    ("good: cites evidence",
     "Confirmed defect at apps/web/src/App.tsx:412. Blocking. "
     "Not tested: the Safari FSA path.",
     "high"),
    ("vague: promises only",
     "I'll cite the file and line, give a verdict of PASS or FAIL, "
     "and state not tested items.",
     "low"),
    ("fluent but unevidenced",
     "The audit is complete and everything checks out. The implementation "
     "looks correct and follows best practices throughout.",
     "low"),
    ("correct but truncated",
     "PASS. apps/web/src/App.tsx:412 reviewed.",
     "mid"),
]


def ask(model, task, rubric, out, max_tokens=1500, retries=3):
    p = PROMPT.format(task=task, rubric=rubric, out=out)
    payload = {"model": model, "stream": True, "max_output_tokens": max_tokens,
               "input": [{"role": "user",
                          "content": [{"type": "input_text", "text": p}]}]}
    req = urllib.request.Request(
        f"{BASE}/responses", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    import time
    for attempt in range(retries):
        try:
            raw = urllib.request.urlopen(req, timeout=180).read().decode()
            break
        except Exception as e:
            if attempt == retries - 1:
                return None, f"{type(e).__name__}: {str(e)[:90]}"
            time.sleep(15 * (attempt + 1))
    ans, done, usage = [], "", {}
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        b = line[6:].strip()
        if not b or b == "[DONE]":
            continue
        try:
            ev = json.loads(b)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        # ONLY the answer stream; reasoning_summary_text is separate.
        if t == "response.output_text.delta":
            ans.append(ev.get("delta", ""))
        elif t == "response.output_text.done":
            done = ev.get("text", "")
        elif t == "response.completed":
            usage = ev.get("response", {}).get("usage", {}) or {}
    s = (done or "".join(ans)).strip()
    s = s.removeprefix("```json").removesuffix("```").strip()
    if "{" not in s:
        return None, "no JSON in answer"
    try:
        d = json.loads(s[s.find("{"):s.rfind("}") + 1])
    except json.JSONDecodeError as e:
        return None, f"unparseable: {str(e)[:60]}"
    comp = (0.5 * d["correctness"] + 0.3 * d["procedure_following"]
            + 0.2 * d["conciseness"])
    return {"composite": comp, "feedback": d.get("feedback", ""),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": (usage.get("output_tokens_details") or {})
            .get("reasoning_tokens")}, None


def run(model, repeats=1):
    print(f"\n{'='*72}\n{model}   (repeats={repeats})\n{'='*72}")
    scores, errs = {}, 0
    for label, out, band in CASES:
        vals = []
        for _ in range(repeats):
            r, err = ask(model, "audit the download gate", RUBRIC, out)
            if err:
                print(f"  {label:26} ERROR {err}")
                errs += 1
                break
            vals.append(r["composite"])
        if not vals:
            continue
        scores[label] = (statistics.mean(vals), band,
                         min(vals) == max(vals), r)
        spread = "" if len(set(vals)) == 1 else f"  spread={min(vals):.2f}-{max(vals):.2f}"
        rt = r.get("reasoning_tokens")
        print(f"  {label:26} {statistics.mean(vals):.3f}  (want {band}){spread}")
        print(f"  {'':26} out_tok={r.get('output_tokens')} reason_tok={rt}")
        print(f"  {'':26} {r['feedback'][:88]}")
    if scores:
        g = scores.get("good: cites evidence", (0,))[0]
        v = scores.get("vague: promises only", (0,))[0]
        f = scores.get("fluent but unevidenced", (0,))[0]
        print(f"\n  discrimination (good - vague) = {g - v:+.3f}")
        print(f"  scepticism(good - fluent)     = {g - f:+.3f}")
    return scores, errs


def stability(model, repeats=3):
    """Same input, repeated. Non-determinism adds noise to a~13 holdout."""
    print(f"\n--- stability: {model.split('/')[-1]} (n={repeats}) ---")
    for label, out in [("good", CASES[0][1]), ("mid/truncated", CASES[3][1])]:
        vals = []
        for _ in range(repeats):
            r, err = ask(model, "audit the download gate", RUBRIC, out)
            vals.append(None if err else round(r["composite"], 3))
        ok = [v for v in vals if v is not None]
        sd = statistics.pstdev(ok) if len(ok) > 1 else 0.0
        print(f"  {label:16} {vals}  spread={max(ok)-min(ok):.3f} sd={sd:.3f}")
    return


if __name__ == "__main__":
    allres = {}
    for m in (FLASH, PRO):
        allres[m] = run(m)
    for m in (FLASH, PRO):
        stability(m)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    hdr = f"{'model':46} {'discrim':>9} {'sceptic':>9} {'errors':>7}"
    print(hdr)
    for m, (sc, errs) in allres.items():
        g = sc.get("good: cites evidence", (0,))[0]
        v = sc.get("vague: promises only", (0,))[0]
        f = sc.get("fluent but unevidenced", (0,))[0]
        print(f"{m.split('/')[-1]:46} {g - v:+9.3f} {g - f:+9.3f} {errs:7}")
    print("\nHigher discrimination/scepticism = the judge better separates a")
    print("real answer from a confident-sounding non-answer.")