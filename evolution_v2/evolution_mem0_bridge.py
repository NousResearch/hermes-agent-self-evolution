#!/usr/bin/env python3
"""
Evolution → Mem0 Bridge (REST API)
Pushes actionable insights into Mem0 via direct REST calls.
Avoids local dependency hell — uses what works.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
import requests

# ── CONFIG ─────────────────────────────────────────────
EVOLUTION_DIR = Path("/mnt/d/gbrain-tan/evolution")
# Read API key from config at runtime
API_KEY = json.loads(
    Path.home().joinpath(".mem0/config.json").read_text()
)["platform"]["api_key"]

BASE = "https://api.mem0.ai/v1"
HEADERS = {"Authorization": f"Token {API_KEY}", "Content-Type": "application/json"}
USER_ID = "pablo-kru"

# ── HELPERS ────────────────────────────────────────────

def read_latest_report():
    """Parse the latest evolution report."""
    latest = None
    for f in EVOLUTION_DIR.glob("daily-*.md"):
        if not latest or f.stat().st_mtime > latest.stat().st_mtime:
            latest = f
    if not latest:
        return {}
    
    text = latest.read_text(encoding='utf-8')
    skills = {}
    for match in re.finditer(r'\|\s*(\S+)\s*\|\s*(\d+)\s*\|', text):
        name, count = match.group(1), match.group(2)
        if name not in ('Skill', 'Uses', 'Error', 'Count', 'Sequence', 'Frequency'):
            try:
                skills[name] = int(count)
            except:
                pass
    
    errors = {}
    for match in re.finditer(r'###\s+(\w+)\s+\((\d+)\s+hits\)', text):
        errors[match.group(1)] = int(match.group(2))
    
    commands = []
    for line in text.split('\n'):
        if line.strip().startswith('- ') and len(commands) < 10:
            cmd = line.strip()[2:]
            if len(cmd) > 10 and not cmd.startswith('['):
                commands.append(cmd[:200])
    
    return {
        "date": latest.stem.replace("daily-", ""),
        "skills": skills,
        "errors": errors,
        "commands": commands,
    }


def mem0_add(text: str, category: str) -> bool:
    """Push insight to Mem0 via REST."""
    try:
        r = requests.post(
            f"{BASE}/memories/",
            headers=HEADERS,
            json={
                "messages": [{"role": "user", "content": text}],
                "user_id": USER_ID,
                "metadata": {"namespace": "kru:evolution", "category": category},
            },
            timeout=15,
        )
        return r.status_code in (200, 201)
    except Exception as e:
        print(f"  Mem0 error ({category}): {e}")
        return False


# ── MAIN ───────────────────────────────────────────────

def main():
    print(f"[{datetime.now().isoformat()}] Evolution → Mem0 Bridge")
    
    report = read_latest_report()
    if not report:
        print("No report found. Exit.")
        return
    
    print(f"Report: {report['date']} | Skills: {len(report['skills'])} | Errors: {len(report['errors'])}")
    
    stored = 0
    
    if report["skills"]:
        top = max(report["skills"], key=report["skills"].get)
        if mem0_add(
            f"Priority skill focus: {top} ({report['skills'][top]} uses in 7d). "
            f"Top request pattern — improve reliability/docs for {top}.",
            "skill_priority"
        ):
            stored += 1
            print(f"  ✓ skill_priority: {top}")
    
    if report["errors"]:
        top = max(report["errors"], key=report["errors"].get)
        if mem0_add(
            f"Top error pattern: {top} ({report['errors'][top]} hits in 7d). "
            f"Review retry/timeout logic for affected tools.",
            "error_pattern"
        ):
            stored += 1
            print(f"  ✓ error_pattern: {top}")
    
    if report["commands"]:
        themes = " | ".join(report["commands"][:5])
        if mem0_add(
            f"Command themes: {themes}. Package recurring requests into reusable skills.",
            "workflow_patterns"
        ):
            stored += 1
            print(f"  ✓ workflow_patterns")
    
    # Meta insight
    total_skills = sum(report["skills"].values()) if report["skills"] else 0
    if mem0_add(
        f"Evolution cycle {report['date']}: {total_skills} skill refs, "
        f"{len(report['errors'])} error categories, {len(report['commands'])} commands extracted.",
        "evolution_meta"
    ):
        stored += 1
        print(f"  ✓ evolution_meta")
    
    print(f"\n✅ Bridge complete. {stored}/4 insights stored to Mem0.")


if __name__ == "__main__":
    main()
