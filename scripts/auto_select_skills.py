#!/usr/bin/env python3
"""Auto-select which skills need evolution based on past performance."""
import json, requests, os, subprocess, sys
from datetime import datetime

HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL", "http://192.168.50.225:8788/v1/default/banks/hermes-clean")

def get_skill_failure_patterns() -> list[dict]:
    """Query Hindsight for skill failure patterns."""
    queries = [
        "Which skills or workflows have failed or caused problems recently?",
        "What mistakes or errors have been corrected in skills?",
        "Which skills are stale or outdated based on recent experiences?"
    ]

    patterns = []
    for q in queries:
        try:
            resp = requests.post(
                f"{HINDSIGHT_URL}/reflect",
                json={"query": q, "mode": "compact"},
                timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("text", data.get("answer", ""))
                if text:
                    patterns.append({"query": q, "findings": text})
        except Exception as e:
            print(f"  ⚠️  Query failed: {e}")

    return patterns


def rank_skills(candidate_skills: list[str], patterns: list[dict]) -> list[str]:
    """Simple heuristic: count mentions of each skill in Hindsight findings."""
    scores = {s: 0 for s in candidate_skills}

    for p in patterns:
        text = p["findings"].lower()
        for skill in candidate_skills:
            if skill.lower() in text:
                scores[skill] += 1

    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


def main():
    # List available skills
    result = subprocess.run(
        ["/Volumes/1TB/AI_Workspace/Hermes/.hermes/hermes-agent/venv/bin/hermes",
         "curator", "list", "--json"],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        print("ERROR: Could not list skills")
        sys.exit(1)

    skills_data = json.loads(result.stdout)
    active_skills = []

    for s in skills_data.get("skills", []):
        if s.get("stale") or s.get("archived"):
            continue
        name = s.get("name", "")
        active_skills.append(name)

    print(f"📋 {len(active_skills)} active skills found")

    # Get Hindsight patterns
    print("🔍 Querying Hindsight for failure patterns...")
    patterns = get_skill_failure_patterns()

    if patterns:
        ranked = rank_skills(active_skills, patterns)
        print(f"\n🎯 Priority queue (top 5):")
        for i, skill in enumerate(ranked[:5]):
            print(f"  {i+1}. {skill}")
    else:
        print("  ⚠️  No patterns found (Hindsight not responding?)")
        ranked = active_skills

    # Output as JSON for automation
    return ranked


if __name__ == "__main__":
    ranked = main()
    json.dump(ranked[:5], sys.stdout)
    print()