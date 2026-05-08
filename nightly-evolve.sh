#!/usr/bin/env bash
# Thin wrapper for the profile-aware Python controller.
set -euo pipefail
cd /home/w0lf/dev/hermes-agent-self-evolution
exec /usr/bin/python /home/w0lf/dev/hermes-agent-self-evolution/scripts/nightly_evolve_cron.py "$@"
