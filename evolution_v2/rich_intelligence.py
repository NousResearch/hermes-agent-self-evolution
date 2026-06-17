#!/usr/bin/env python3
"""
HERMES SELF-EVOLUTION ENGINE v2.2 — Rich Intelligence Pipeline
Produces actionable evolution insights: skill DNA, error heatmaps, workflow patterns.
"""

import sqlite3
import json
import re
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Set

STATE_DB = Path.home() / ".hermes" / "state.db"
GBRAIN_DIR = Path("/mnt/d/gbrain-tan")
LOOKBACK_DAYS = 7

# ── EXTRACTORS ─────────────────────────────────────────

def extract_skill_dna(conn, since_ts: float) -> List[dict]:
    """Extract skill usage DNA: which skills are used together, in what context."""
    c = conn.execute("""
        SELECT m.session_id, m.content, m.tool_calls, m.tool_name
        FROM messages m
        WHERE m.timestamp > ? AND (m.role = 'user' OR m.role = 'assistant')
        ORDER BY m.session_id, m.timestamp
    """, (since_ts,))
    
    sessions = defaultdict(list)
    for row in c.fetchall():
        sessions[row["session_id"]].append(row)
    
    skill_dna = []
    for session_id, msgs in sessions.items():
        skills_in_session = set()
        text = " ".join(str(m["content"] or "") for m in msgs)
        
        # Find all skill references
        for match in re.finditer(r'skill_view\(["\']([^"\']+)["\']', text):
            skills_in_session.add(match.group(1))
        
        # Also check tool names for skill indicators
        for m in msgs:
            if m["tool_name"] and "skill" in m["tool_name"]:
                skills_in_session.add(m["tool_name"])
        
        if len(skills_in_session) >= 2:
            skill_dna.append({
                "session": session_id[:8],
                "skills": sorted(skills_in_session),
                "count": len(skills_in_session),
            })
    
    # Return top multi-skill sessions
    return sorted(skill_dna, key=lambda x: x["count"], reverse=True)[:20]


def extract_error_heatmap(conn, since_ts: float) -> dict:
    """Extract error patterns with context — what fails, when, and what fixes it."""
    c = conn.execute("""
        SELECT m.session_id, m.role, m.content, m.tool_name, m.timestamp
        FROM messages m
        WHERE m.timestamp > ?
        ORDER BY m.session_id, m.timestamp
    """, (since_ts,))
    
    errors = defaultdict(lambda: {"count": 0, "fixes": Counter(), "tools": Counter()})
    
    session_messages = defaultdict(list)
    for row in c.fetchall():
        session_messages[row["session_id"]].append(row)
    
    for session_id, msgs in session_messages.items():
        for i, msg in enumerate(msgs):
            if msg["role"] != "tool":
                continue
            
            content = msg["content"] or ""
            
            # Classify error
            error_type = None
            if re.search(r"timed out|timeout|deadline exceeded", content, re.I):
                error_type = "timeout"
            elif re.search(r"not found|No such file|404|does not exist", content, re.I):
                error_type = "not_found"
            elif re.search(r"JSONDecodeError|SyntaxError|parse|invalid", content, re.I):
                error_type = "parse"
            elif re.search(r"permission|access denied|403|unauthorized", content, re.I):
                error_type = "permission"
            elif re.search(r"auth|token|key|credential", content, re.I):
                error_type = "auth"
            elif re.search(r"rate limit|too many|429", content, re.I):
                error_type = "rate_limit"
            elif re.search(r"connection|refused|network|ECONN", content, re.I):
                error_type = "connection"
            elif re.search(r"exit code [1-9]|command not found|failed", content, re.I):
                error_type = "shell"
            
            if error_type:
                errors[error_type]["count"] += 1
                errors[error_type]["tools"][msg["tool_name"] or "unknown"] += 1
                
                # Look ahead for fix (next 3 messages)
                if i + 3 < len(msgs):
                    next_msgs = msgs[i+1:i+4]
                    next_text = " ".join(str(m["content"] or "") for m in next_msgs)
                    
                    # What fixed it?
                    if "patch" in next_text.lower():
                        errors[error_type]["fixes"]["code_fix"] += 1
                    elif "retry" in next_text.lower() or "again" in next_text.lower():
                        errors[error_type]["fixes"]["retry"] += 1
                    elif "chmod" in next_text.lower() or "permission" in next_text.lower():
                        errors[error_type]["fixes"]["permission_fix"] += 1
                    elif "install" in next_text.lower() or "apt" in next_text.lower() or "pip" in next_text.lower():
                        errors[error_type]["fixes"]["install_missing"] += 1
                    elif "env" in next_text.lower() or "export" in next_text.lower():
                        errors[error_type]["fixes"]["env_fix"] += 1
                    else:
                        errors[error_type]["fixes"]["other"] += 1
    
    return dict(errors)


def extract_workflow_patterns(conn, since_ts: float) -> List[dict]:
    """Extract successful multi-step workflows that should become skills."""
    c = conn.execute("""
        SELECT m.session_id, m.role, m.content, m.tool_name, m.tool_calls
        FROM messages m
        WHERE m.timestamp > ? AND m.role IN ('assistant', 'tool')
        ORDER BY m.session_id, m.timestamp
    """, (since_ts,))
    
    workflows = []
    session_msgs = defaultdict(list)
    for row in c.fetchall():
        session_msgs[row["session_id"]].append(row)
    
    for session_id, msgs in session_msgs.items():
        if len(msgs) < 5:
            continue
        
        # Find sequences of tool calls that succeed (no error in following tool msg)
        tool_sequence = []
        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant" and msg["tool_calls"]:
                try:
                    tcs = json.loads(msg["tool_calls"]) if isinstance(msg["tool_calls"], str) else msg["tool_calls"]
                    for tc in tcs:
                        tool_name = tc.get("function", {}).get("name", msg["tool_name"] or "unknown")
                        tool_sequence.append(tool_name)
                except:
                    if msg["tool_name"]:
                        tool_sequence.append(msg["tool_name"])
            
            # If we see an error tool, break the sequence
            if msg["role"] == "tool" and msg["content"]:
                if any(e in msg["content"] for e in ["Error", "error", "Traceback", "FAIL", "failed"]):
                    if tool_sequence:
                        tool_sequence = []  # Reset on error
        
        # If sequence is 4+ tools long and ends without error, it's a workflow
        if len(tool_sequence) >= 4:
            workflows.append({
                "session": session_id[:8],
                "length": len(tool_sequence),
                "sequence": " -> ".join(tool_sequence[:8]),
                "tools": tool_sequence,
            })
    
    return sorted(workflows, key=lambda x: x["length"], reverse=True)[:15]


# ── MAIN ENHANCED ──────────────────────────────────────

def main():
    print("=" * 60)
    print(" HERMES SELF-EVOLUTION ENGINE v2.2 — Rich Intelligence")
    print("=" * 60)
    
    conn = sqlite3.connect(str(STATE_DB))
    conn.row_factory = sqlite3.Row
    
    since = datetime.now().astimezone() - timedelta(days=LOOKBACK_DAYS)
    since_ts = since.timestamp()
    
    print(f"\nAnalyzing last {LOOKBACK_DAYS} days of activity...")
    print(f"Timestamp threshold: {since_ts}")
    
    # ── Skill DNA ──
    print("\n[1/4] Extracting skill DNA...")
    skill_dna = extract_skill_dna(conn, since_ts)
    print(f"      Found {len(skill_dna)} multi-skill sessions")
    
    # ── Error Heatmap ──
    print("\n[2/4] Building error heatmap...")
    error_heatmap = extract_error_heatmap(conn, since_ts)
    print(f"      Found {len(error_heatmap)} error categories")
    for err_type, data in error_heatmap.items():
        print(f"      - {err_type}: {data['count']} occurrences")
    
    # ── Workflow Patterns ──
    print("\n[3/4] Mining workflow patterns...")
    workflows = extract_workflow_patterns(conn, since_ts)
    print(f"      Found {len(workflows)} candidate workflows")
    
    # ── Write Rich Report ──
    print("\n[4/4] Writing intelligence report...")
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = f"evolution/intelligence-{date_str}"
    page_path = GBRAIN_DIR / f"{slug}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build skill DNA markdown
    dna_md = "\n".join(
        f"| {d['session']} | {', '.join(d['skills'][:4])} | {d['count']} |"
        for d in skill_dna[:15]
    ) if skill_dna else "_No multi-skill sessions found._"
    
    # Build error heatmap markdown
    error_md = ""
    for err_type, data in sorted(error_heatmap.items(), key=lambda x: x[1]["count"], reverse=True):
        error_md += f"\n### {err_type.replace('_', ' ').title()} ({data['count']} hits)\n\n"
        error_md += "**Most affected tools:**\n"
        for tool, count in data["tools"].most_common(5):
            error_md += f"- `{tool}`: {count}\n"
        error_md += "\n**Fix strategies:**\n"
        for fix, count in data["fixes"].most_common():
            error_md += f"- `{fix}`: {count} times\n"
    
    if not error_md:
        error_md = "_No error patterns detected._"
    
    # Build workflow markdown
    workflow_md = "\n".join(
        f"| {w['session']} | {w['length']} | `{w['sequence'][:80]}` |"
        for w in workflows[:10]
    ) if workflows else "_No candidate workflows found._"
    
    content = f"""---
type: evolution_intelligence
date: {date_str}
engine_version: 2.2
---

# Evolution Intelligence Report — {date_str}

This report contains **actionable patterns** extracted from live session data.
Produced by Evolution Engine v2.2 — streaming pipeline over state.db.

---

## Skill DNA: Multi-Skill Sessions

Sessions where multiple skills were used together (indicates complex workflows).

| Session | Skills Used | Count |
|---------|-------------|-------|
{dna_md}

## Error Heatmap & Recovery Patterns

{error_md}

## Workflow Candidates

Long tool sequences that succeeded without errors — candidates for skillification.

| Session | Length | Sequence |
|---------|--------|----------|
{workflow_md}

---

## Actionable Insights

### Immediate Fixes
- Review timeout-heavy tools for retry/backoff logic
- Add pre-flight checks for "not_found" errors (file existence, path validation)
- Standardize parse error handling (JSON schema validation before decode)

### Skill Candidates
The following multi-tool sequences appear successful and should be wrapped as skills:
"""
    
    # Add skill candidates
    for w in workflows[:5]:
        content += f"\n- **{w['sequence'][:60]}**... (session {w['session']})\n"
    
    content += f"""

### Memory Consolidation Notes
- {len(skill_dna)} sessions show cross-skill usage → consider unified skill bundles
- {len(error_heatmap)} error categories have identifiable fix patterns → add to error_recovery skill
- {len(workflows)} successful workflows detected → candidate for procedural memory extraction

---
*Generated by Hermes Self-Evolution Engine v2.2*
*Processing: live state.db stream, {LOOKBACK_DAYS}-day lookback*
"""
    
    page_path.write_text(content, encoding='utf-8')
    
    # Trigger sync
    print(f"\n  Written: {page_path}")
    print("\nTriggering GBrain sync...")
    os.system(f"cd {GBRAIN_DIR} && gbrain sync --source evolution 2>/dev/null")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print(" INTELLIGENCE CYCLE COMPLETE")
    print("=" * 60)
    print(f"Skill DNA entries: {len(skill_dna)}")
    print(f"Error categories: {len(error_heatmap)}")
    print(f"Workflow candidates: {len(workflows)}")
    print(f"Report: {page_path}")


if __name__ == "__main__":
    main()
