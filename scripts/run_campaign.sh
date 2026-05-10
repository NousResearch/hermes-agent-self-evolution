#!/usr/bin/env bash
# Deploy-gate calibration campaign runbook.
#
# This is a checkpoint-driven driver, not a fire-and-forget script: 24
# evolve runs at ~$10 each is too much money to launch unattended. Each
# phase pauses for explicit confirmation; cost monitoring happens via
# `scripts/campaign_status.py` between phases.
#
# Stages map to ~/.claude/plans/i-want-to-implement-peaceful-stearns.md.
#
# Pre-reqs:
#   - Stage 1 code prep merged (this branch).
#   - SKILL_SOURCES_HERMES_REPO set so --skill resolves to the hermes-agent
#     skills directory.
#   - OPENAI_API_KEY populated.
#
# Usage:
#   bash scripts/run_campaign.sh

set -euo pipefail

CAMPAIGN_START_TS=$(date +%Y%m%d_%H%M%S)
CAP_USD=200
STATUS_CMD="uv run python scripts/campaign_status.py --since ${CAMPAIGN_START_TS} --cap ${CAP_USD}"
EVOLVE="uv run python -m evolution.skills.evolve_skill --iterations 10 --evaluate-band-on-holdout"

# Calibration corpus (Stage 2)
SKILLS_CALIB_SMALL=(nano-pdf apple-notes)
SKILLS_CALIB_MED=(polymarket huggingface-hub)
SKILLS_CALIB_LARGE=(notion spotify)
SKILLS_CALIB_ALL=("${SKILLS_CALIB_SMALL[@]}" "${SKILLS_CALIB_MED[@]}" "${SKILLS_CALIB_LARGE[@]}")

# Validation corpus (Stage 7) — must NOT appear in output/ at campaign start
SKILLS_VALIDATION=(plan maps linear)

confirm() {
    local prompt="$1"
    read -r -p "${prompt} [y/N] " answer
    case "${answer}" in
        y|Y|yes|YES) return 0 ;;
        *) echo "Aborted."; exit 1 ;;
    esac
}

echo "=== Deploy-gate calibration campaign ==="
echo "Campaign start: ${CAMPAIGN_START_TS}"
echo "Cost cap:       \$${CAP_USD}"
echo

# ----------------------------------------------------------------------
# Stage 4 — Study A: generate large holdout pools, then pick N* and ratio*
# ----------------------------------------------------------------------
echo "--- Stage 4: Study A (dataset gen + N*/ratio* analysis) ---"
confirm "Generate N=400 holdout pools for the 6 calibration skills (~\$20)?"
for skill in "${SKILLS_CALIB_ALL[@]}"; do
    echo ">>> Generating pool for ${skill}"
    uv run python scripts/generate_large_holdout.py --skill "${skill}" --n 400 --seed 42
done
echo
echo "Now run: uv run python scripts/analysis/study_a_pick_n.py"
echo "Then export the chosen values:"
echo "    export N_STAR=...        # one of {50,100,150,250,400}"
echo "    export RATIO_STAR=...    # one of {0.36,0.50,0.65}"
confirm "Have you exported N_STAR and RATIO_STAR?"
: "${N_STAR:?N_STAR not exported}"
: "${RATIO_STAR:?RATIO_STAR not exported}"
${STATUS_CMD}
confirm "Proceed to Stage 5 (Study C — 12 evolve runs at the picked N*/ratio*)?"

# ----------------------------------------------------------------------
# Stage 5 — Study C: 12 growth/control runs (smoke = run 1/12)
# ----------------------------------------------------------------------
echo "--- Stage 5: Study C (12 evolve runs) ---"

# Smoke run = nano-pdf seed=42 lenient + bap-safety=0.0. If growth_pct < 0,
# abort — the lenient/BAP-off setup didn't produce growth and (free, slope)
# calibration is not viable on this corpus.
echo ">>> SMOKE: nano-pdf seed=42 (= run 1/12 of Study C)"
${EVOLVE} \
    --skill nano-pdf --seed 42 \
    --quality-gate lenient --bap-safety-margin 0.0 \
    --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
echo "Inspect: jq '.growth_pct' output/nano-pdf/<latest>/gate_decision.json"
confirm "Did the smoke run produce growth_pct > 0?"

# Remaining 7 growth runs (3 skills × 2 seeds + 1 skill × 1 seed)
declare -a REMAINING_GROWTH=(
    "nano-pdf 7"
    "apple-notes 42" "apple-notes 7"
    "polymarket 42" "polymarket 7"
    "huggingface-hub 42" "huggingface-hub 7"
)
for entry in "${REMAINING_GROWTH[@]}"; do
    read -r skill seed <<<"${entry}"
    echo ">>> GROWTH: ${skill} seed=${seed}"
    ${EVOLVE} \
        --skill "${skill}" --seed "${seed}" \
        --quality-gate lenient --bap-safety-margin 0.0 \
        --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
    ${STATUS_CMD}
    confirm "Continue Stage 5?"
done

# 4 control runs at default preset — confirms BAP-off is the variable
# causing growth, not anything else in the lenient setup.
declare -a CONTROL_PAIRS=(
    "nano-pdf 42"
    "apple-notes 42"
    "polymarket 42"
    "huggingface-hub 42"
)
for entry in "${CONTROL_PAIRS[@]}"; do
    read -r skill seed <<<"${entry}"
    echo ">>> CONTROL: ${skill} seed=${seed} (--quality-gate default)"
    ${EVOLVE} \
        --skill "${skill}" --seed "${seed}" \
        --quality-gate default \
        --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
done
${STATUS_CMD}

# ----------------------------------------------------------------------
# Stage 6 — Study B + (free, slope) analysis (no new evolve runs)
# ----------------------------------------------------------------------
echo "--- Stage 6: analysis on Study C outputs ---"
echo "Run: uv run python scripts/analysis/study_b_pick_epsilon.py"
echo "Run: uv run python scripts/analysis/study_c_pick_curve.py"
echo "Then export:"
echo "    export EPSILON_STAR=..."
echo "    export FREE_STAR=...     # or 'KEEP_CURRENT' if Study C verdict was INSUFFICIENT_DATA"
echo "    export SLOPE_STAR=..."
confirm "Have you exported EPSILON_STAR / FREE_STAR / SLOPE_STAR?"
: "${EPSILON_STAR:?}" "${FREE_STAR:?}" "${SLOPE_STAR:?}"

# ----------------------------------------------------------------------
# Stage 7 — Study D: 12 validation runs (current vs proposed defaults)
# ----------------------------------------------------------------------
echo "--- Stage 7: Study D (12 validation runs) ---"
confirm "Proceed to Stage 7?"
for skill in "${SKILLS_VALIDATION[@]}"; do
    for seed in 42 7; do
        echo ">>> VALIDATION current: ${skill} seed=${seed}"
        ${EVOLVE} \
            --skill "${skill}" --seed "${seed}" \
            --quality-gate default
        echo ">>> VALIDATION proposed: ${skill} seed=${seed}"
        if [[ "${FREE_STAR}" == "KEEP_CURRENT" ]]; then
            # Study C verdict was INSUFFICIENT_DATA — only N*/ratio*/ε* changed
            ${EVOLVE} \
                --skill "${skill}" --seed "${seed}" \
                --quality-gate default \
                --knee-point-epsilon "${EPSILON_STAR}" \
                --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
        else
            ${EVOLVE} \
                --skill "${skill}" --seed "${seed}" \
                --quality-gate default \
                --growth-free-threshold "${FREE_STAR}" \
                --growth-quality-slope "${SLOPE_STAR}" \
                --knee-point-epsilon "${EPSILON_STAR}" \
                --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
        fi
        ${STATUS_CMD}
        confirm "Continue Stage 7?"
    done
done

echo
echo "=== Campaign runs complete ==="
echo "Run: uv run python scripts/analysis/study_d_compare.py --since ${CAMPAIGN_START_TS}"
${STATUS_CMD}
