#!/bin/bash
# ============================================================
# Self-Evolution Environment Bootstrap — hardened with timeouts
# Uses filesystem checks only for heavy packages to avoid import hangs
# ============================================================
set -euo pipefail

REPO="/mnt/d/hermes-agent-self-evolution"
VENV="$REPO/.venv"
PYTHON="$VENV/bin/python3"
SITE="$VENV/lib/python3.12/site-packages"

# ── 1. Ensure virtualenv exists ────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "[BOOTSTRAP] Creating venv at $VENV..."
    python3 -m venv "$VENV"
    SITE="$VENV/lib/python3.12/site-packages"
fi

# ── 2. Ensure pip is available (30s timeout) ─────────────
if ! timeout 10 "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "[BOOTSTRAP] Installing pip..."
    timeout 20 "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1
    timeout 30 "$PYTHON" -m pip install --upgrade pip setuptools wheel -q >/dev/null 2>&1 || true
fi

# ── 3. Filesystem-based dep check (no imports) ───────────
needs_install=()

# Map package names to expected directories in site-packages
declare -a PKG_MAP=(
    "dspy:dspy"
    "openai:openai"
    "click:click"
    "rich:rich"
    "pyyaml:yaml"
)

for entry in "${PKG_MAP[@]}"; do
    pkg="${entry%%:*}"
    dir="${entry##*:}"
    if [ ! -d "$SITE/$dir" ]; then
        needs_install+=("$pkg")
    fi
done

if [ ${#needs_install[@]} -gt 0 ]; then
    echo "[BOOTSTRAP] Installing: ${needs_install[*]}"
    for pkg in "${needs_install[@]}"; do
        timeout 90 "$PYTHON" -m pip install "$pkg" -q >/dev/null 2>&1 || {
            echo "[BOOTSTRAP] ⚠️ $pkg install timed out — may need manual fix"
        }
    done
fi

# ── 4. Verify (lightweight imports, 5s timeout) ───────────
if timeout 5 "$PYTHON" -c "import click, rich, yaml" 2>/dev/null; then
    if [ -d "$SITE/dspy" ] && [ -d "$SITE/openai" ]; then
        echo "[BOOTSTRAP] ✅ All deps ready"
        exit 0
    fi
fi

echo "[BOOTSTRAP] ❌ Failed to verify deps"
exit 1
