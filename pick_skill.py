#!/usr/bin/env python3
"""Pick the next skill to evolve, prioritized by actual usage frequency.

Reads from the dedicated skill_usage.db (clean, no false positives).
Falls back to session DB FTS if usage DB is empty (cold start).

State file: /Users/eric/Playground/hermes-agent-self-evolution/.evolve_state.json
"""
import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

STATE_FILE = Path("/Users/eric/Playground/hermes-agent-self-evolution/.evolve_state.json")
SKILLS_DIR = Path.home() / ".hermes" / "skills"
USAGE_DB = Path.home() / ".hermes" / "skill_usage.db"
STATE_DB = Path.home() / ".hermes" / "state.db"
SELF_EVOLVE_DIR = Path("/Users/eric/Playground/hermes-agent-self-evolution")

# Skills not worth evolving
SKIP = {
    # Too small / no procedure to optimize
    "apple-notes", "apple-reminders", "findmy", "imessage",
    "macos-icloud-documents-folder-offload",
    "apple-silicon-mlx-model-shortlist", "codebase-inspection",
    "github-auth", "agent-browser-cdp-smoke-test",
    # Infrastructure / meta skills (evolving these is circular)
    "hermes-agent-setup", "dogfood", "hermes-skill-self-evolution",
    "gateway-agent-timeout-autoresume", "complete-agent-profile-creation",
    "cron-monitor-dedup-state", "nous-portal-infrastructure-audit",
    "bookmark-bucket-reliability-triage",
    "hermes-telegram-session-hygiene-debugging",
    "hermes-oess-seat-installation", "oess-base-image-setup",
    "nora-operating-model", "oess-executive-team-orchestrator",
    # Very broad matches — not useful to evolve
    "hermes-agent", "cli", "plan", "codex", "guidance",
}

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"history": [], "last_run": None}

def discover_skills():
    """Find all skills with substantial content (>500 chars body)."""
    skills = {}
    for skill_md in SKILLS_DIR.rglob("SKILL.md"):
        name = skill_md.parent.name
        if name in SKIP:
            continue
        raw = skill_md.read_text()
        body = raw.split("---", 2)[-1].strip() if raw.strip().startswith("---") else raw
        if len(body) > 500:
            skills[name] = {
                "name": name,
                "path": str(skill_md),
                "size": len(raw),
                "body_size": len(body),
            }
    return skills

def get_usage_from_tracker():
    """Read clean usage counts from skill_usage.db."""
    if not USAGE_DB.exists():
        return {}
    try:
        conn = sqlite3.connect(str(USAGE_DB))
        cursor = conn.execute(
            "SELECT skill_name, COUNT(*) FROM skill_invocations GROUP BY skill_name ORDER BY COUNT(*) DESC"
        )
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return counts
    except Exception:
        return {}

def get_usage_from_sessiondb(skills):
    """Fallback: mine session DB for skill load events (cold start)."""
    counts = {}
    try:
        conn = sqlite3.connect(str(STATE_DB))
        cursor = conn.cursor()
        for name in skills:
            patterns = [
                f'%invoked the "{name}" skill%',
                f'%invoked the \\"{name}\\" skill%',
                f'%/skill {name}%',
            ]
            total = 0
            for p in patterns:
                cursor.execute("SELECT COUNT(*) FROM messages_fts WHERE content LIKE ?", (p,))
                total += cursor.fetchone()[0]
            if total > 0:
                counts[name] = total
        conn.close()
    except Exception:
        pass
    return counts

def pick_next(skills, usage_counts, state):
    """Pick the most-used skill that hasn't been evolved, or was evolved longest ago."""
    history = {h["name"]: h for h in state["history"]}

    never_evolved = []
    previously_evolved = []

    for name, s in skills.items():
        usage = usage_counts.get(name, 0)
        if name not in history:
            never_evolved.append((usage, name, s))
        else:
            h = history[name]
            last_result = h.get("result", "")
            last_ts = h.get("timestamp", "")
            # 24h cooldown on failures
            if last_result == "failed" and last_ts:
                try:
                    dt = datetime.fromisoformat(last_ts)
                    hours_ago = (datetime.now() - dt).total_seconds() / 3600
                    if hours_ago < 24:
                        continue
                except ValueError:
                    pass
            staleness = 0
            if last_ts:
                try:
                    staleness = (datetime.now() - datetime.fromisoformat(last_ts)).total_seconds() / 3600
                except ValueError:
                    pass
            previously_evolved.append((usage, staleness, name, s))

    # Never-evolved first (most used), then stalest previously-evolved
    never_evolved.sort(key=lambda x: -x[0])
    previously_evolved.sort(key=lambda x: (-x[0], -x[1]))

    if never_evolved:
        return never_evolved[0][1], never_evolved[0][2]
    elif previously_evolved:
        return previously_evolved[0][2], previously_evolved[0][3]
    else:
        return None, None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of skills to pick")
    args = parser.parse_args()

    state = load_state()
    skills = discover_skills()

    if not skills:
        print("ERROR: No skills found", file=sys.stderr)
        sys.exit(1)

    # Primary: clean tracker DB
    usage_counts = get_usage_from_tracker()

    # Cold start fallback: if tracker is empty, mine session DB
    if not usage_counts:
        usage_counts = get_usage_from_sessiondb(skills)

    evolved_set = {h["name"] for h in state["history"]}
    
    # Pick N skills in priority order
    picks = []
    remaining = dict(skills)
    remaining_counts = dict(usage_counts)
    
    for _ in range(min(args.count, len(remaining))):
        name, skill = pick_next(remaining, remaining_counts, state)
        if not name:
            break
        picks.append(name)
        del remaining[name]  # Don't pick the same one twice in a batch

    # Print ranking to stderr
    ranked = sorted(usage_counts.items(), key=lambda x: -x[1])[:15]
    print(f"Picking {len(picks)} skill(s):", file=sys.stderr)
    for i, (n, c) in enumerate(ranked, 1):
        mark = "✓" if n in evolved_set else " "
        pick_marker = " ◀ PICKED" if n in picks else ""
        if n in skills:
            print(f"  {i:2d}. [{mark}] {n}: {c} invocations{pick_marker}", file=sys.stderr)

    print("\n".join(picks))

if __name__ == "__main__":
    main()
