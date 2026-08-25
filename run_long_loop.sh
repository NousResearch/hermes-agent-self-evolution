#!/usr/bin/env bash
# Long evolution loop: Phase 1 (skills) + Phase 2 (tool descriptions).
# Eval model: openrouter/openai/gpt-4.1-nano (chosen: weak-but-stable = real GEPA signal).
# Optimizer:  openrouter/openai/gpt-4.1 for both phases.
set -a
. /home/raul/.hermes/.env >/dev/null 2>&1
set +a
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
cd /home/raul/dev/active/hermes/hermes-agent-self-evolution

echo "=== LOOP START $(date -Is) ==="

# ---------- PHASE 1: skills ----------
for skill in github-code-review systematic-debugging; do
  echo "--- [phase1] skill=$skill synthetic $(date -Is)"
  .venv/bin/python -m evolution.skills.evolve_skill \
    --skill "$skill" --iterations 10 --eval-source synthetic \
    --optimizer-model "openrouter/openai/gpt-4.1" \
    --eval-model "openrouter/openai/gpt-4.1-nano" \
    > "loop_phase1_${skill}_synthetic.log" 2>&1
  echo "exit=$?"

  echo "--- [phase1] skill=$skill sessiondb $(date -Is)"
  .venv/bin/python -m evolution.skills.evolve_skill \
    --skill "$skill" --iterations 10 --eval-source sessiondb \
    --optimizer-model "openrouter/openai/gpt-4.1" \
    --eval-model "openrouter/openai/gpt-4.1-nano" \
    > "loop_phase1_${skill}_sessiondb.log" 2>&1
  echo "exit=$?"
done

# ---------- PHASE 2: tool descriptions (hardened dataset) ----------
for group in "read_file,write_file,search_files,terminal" "web_search,web_extract" ; do
  tag=$(echo "$group" | tr ',' '_')
  echo "--- [phase2] tools=$group $(date -Is)"
  .venv/bin/python -m evolution.tools.evolve_tool_descriptions \
    --tools "$group" --iterations 12 --cases-per-tool 4 \
    --optimizer-model "openrouter/openai/gpt-4.1" \
    --eval-model "openrouter/openai/gpt-4.1-nano" \
    > "loop_phase2_${tag}.log" 2>&1
  echo "exit=$?"
done

echo "=== LOOP DONE $(date -Is) ==="
grep -H '"improvement"' output/*/*/*/metrics.json 2>/dev/null | tail -20
grep -HE "Holdout|Change|improvement" loop_*.log | tail -30
