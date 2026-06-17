#!/usr/bin/env python3
"""
HERMES SELF-EVOLUTION ENGINE v2 — Streaming Pipeline
Reads jsonl session files directly, extracts patterns, writes to GBrain.
Built for Pablo's AI Lab. 2026-06-09.

Key differences from v1:
- v1: Assumed SQLite sessions.db, queried ghost tables, found nothing
- v2: Reads raw jsonl files, streams through all 216 files in parallel
- Extracts: skill usage patterns, error recovery sequences, successful workflows
- Writes: Consolidated findings to GBrain pages, triggers autopilot consolidation
"""

import json
import gzip
import re
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional

# ── CONFIG ─────────────────────────────────────────────
SESSIONS_DIR = Path.home() / ".hermes" / "sessions"
GBRAIN_DIR = Path("/mnt/d/gbrain-tan")
OUTPUT_DIR = Path("/mnt/d/hermes-agent-self-evolution/evolution_v2/output")
MAX_WORKERS = 8  # Parallel session processing
MIN_SESSION_MESSAGES = 3  # Skip trivial sessions
LOOKBACK_DAYS = 7  # Only process recent sessions

# Skills to track evolution for
TRACKED_SKILLS = [
    "codex-image-gen", "pablo-campaign", "pablo-cinematic-engine",
    "fx-rich", "trading", "security", "marketing", "devops",
    "open-source-bounty-hunter", "hermes-agent", "god-mode-protocol"
]

# ── SESSION READER ─────────────────────────────────────

def read_session_file(path: Path) -> List[dict]:
    """Read a session file (jsonl or gzipped jsonl)."""
    messages = []
    try:
        if path.suffix == '.gz':
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
    return messages


def parse_session_metadata(messages: List[dict]) -> dict:
    """Extract metadata from session meta message."""
    for msg in messages:
        if msg.get("role") == "session_meta":
            return {
                "tools": msg.get("tools", []),
                "model": msg.get("model", "unknown"),
                "platform": msg.get("platform", "unknown"),
                "timestamp": msg.get("timestamp", ""),
            }
    return {}


# ── PATTERN EXTRACTORS ─────────────────────────────────

def extract_skill_usage(messages: List[dict]) -> Counter:
    """Count skill_view calls and other skill references."""
    skills = Counter()
    text = " ".join(str(m.get("content", "")) for m in messages)
    
    # Find skill_view calls
    for match in re.finditer(r'skill_view\(["\']([^"\']+)["\']', text):
        skills[match.group(1)] += 1
    
    # Find skill mentions in natural language
    for skill in TRACKED_SKILLS:
        if skill.replace('-', ' ') in text.lower() or skill in text.lower():
            skills[skill] += text.lower().count(skill.replace('-', ' '))
    
    return skills


def extract_error_recovery(messages: List[dict]) -> List[dict]:
    """Find sequences where errors were fixed."""
    recoveries = []
    
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        
        content = str(msg.get("content", ""))
        output = str(msg.get("tool_output", ""))
        
        # Error patterns
        error_patterns = [
            "Error:", "error:", "Traceback", "FAIL", "failed",
            "timed out", "timeout", "Connection refused", "404",
            "permission denied", "No such file", "not found"
        ]
        
        has_error = any(p in content or p in output for p in error_patterns)
        
        if has_error and i + 2 < len(messages):
            # Look ahead for success
            next_msgs = messages[i+1:i+4]
            next_text = " ".join(str(m.get("content", "")) for m in next_msgs)
            
            success_markers = ["success", "ok", "done", "completed", "fixed", "working"]
            if any(s in next_text.lower() for s in success_markers):
                # Found error → fix sequence
                tool_name = msg.get("tool_calls", [{}])[0].get("function", {}).get("name", "unknown") if msg.get("tool_calls") else "unknown"
                
                recoveries.append({
                    "error_tool": tool_name,
                    "error_summary": content[:200],
                    "fix_summary": next_text[:300],
                    "sequence_length": len(messages[i:i+4]),
                })
    
    return recoveries


def extract_tool_sequences(messages: List[dict]) -> Counter:
    """Find common tool call sequences (what tools are used together)."""
    sequences = Counter()
    
    # Extract tool calls from assistant messages
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "unknown")
                tool_calls.append(name)
    
    # Find pairs and triplets
    for i in range(len(tool_calls) - 1):
        pair = f"{tool_calls[i]} → {tool_calls[i+1]}"
        sequences[pair] += 1
    
    for i in range(len(tool_calls) - 2):
        triplet = f"{tool_calls[i]} → {tool_calls[i+1]} → {tool_calls[i+2]}"
        sequences[triplet] += 1
    
    return sequences


def extract_user_preferences(messages: List[dict]) -> Dict[str, List[str]]:
    """Extract corrections and preferences from user messages."""
    prefs = defaultdict(list)
    
    for msg in messages:
        if msg.get("role") != "user":
            continue
        
        content = str(msg.get("content", "")).lower()
        
        # Preference patterns
        if "remember" in content or "don't forget" in content:
            prefs["memory_requests"].append(content[:200])
        
        if "always" in content or "never" in content:
            prefs["rules"].append(content[:200])
        
        if "fix" in content or "wrong" in content or "incorrect" in content:
            prefs["corrections"].append(content[:200])
        
        if "better" in content or "improve" in content or "upgrade" in content:
            prefs["improvements"].append(content[:200])
    
    return dict(prefs)


# ── SESSION PROCESSOR ──────────────────────────────────

def process_session_file(path: Path) -> Optional[dict]:
    """Process a single session file and return extracted insights."""
    messages = read_session_file(path)
    
    if len(messages) < MIN_SESSION_MESSAGES:
        return None
    
    meta = parse_session_metadata(messages)
    
    # Skip old sessions
    try:
        ts = meta.get("timestamp", "")
        if ts:
            session_date = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if session_date < datetime.now().astimezone() - timedelta(days=LOOKBACK_DAYS):
                return None
    except:
        pass
    
    return {
        "file": str(path.name),
        "message_count": len(messages),
        "metadata": meta,
        "skills_used": extract_skill_usage(messages),
        "recoveries": extract_error_recovery(messages),
        "tool_sequences": extract_tool_sequences(messages),
        "preferences": extract_user_preferences(messages),
    }


# ── AGGREGATOR ─────────────────────────────────────────

def aggregate_results(results: List[dict]) -> dict:
    """Aggregate patterns across all processed sessions."""
    all_skills = Counter()
    all_recoveries = []
    all_sequences = Counter()
    all_prefs = defaultdict(list)
    
    total_messages = 0
    total_sessions = 0
    platform_counts = Counter()
    model_counts = Counter()
    
    for r in results:
        if not r:
            continue
        
        total_sessions += 1
        total_messages += r["message_count"]
        all_skills.update(r["skills_used"])
        all_recoveries.extend(r["recoveries"])
        all_sequences.update(r["tool_sequences"])
        
        for key, vals in r["preferences"].items():
            all_prefs[key].extend(vals)
        
        meta = r.get("metadata", {})
        platform_counts[meta.get("platform", "unknown")] += 1
        model_counts[meta.get("model", "unknown")] += 1
    
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "top_skills": all_skills.most_common(20),
        "recoveries": all_recoveries[:50],  # Top 50 recovery patterns
        "top_sequences": all_sequences.most_common(30),
        "preferences": dict(all_prefs),
        "platforms": dict(platform_counts),
        "models": dict(model_counts),
        "generated_at": datetime.now().isoformat(),
    }


# ── GBRAIN WRITER ──────────────────────────────────────

def write_gbrain_page(data: dict) -> str:
    """Write evolution findings as a GBrain page."""
    ts = datetime.now().strftime("%Y-%m-%d")
    slug = f"evolution/daily-{ts}"
    
    # Build markdown content
    skills_md = "\n".join(
        f"| {skill} | {count} |"
        for skill, count in data["top_skills"]
    )
    
    sequences_md = "\n".join(
        f"| {seq} | {count} |"
        for seq, count in data["top_sequences"]
    )
    
    recoveries_md = "\n".join(
        f"- **{r['error_tool']}**: {r['error_summary'][:80]}... → fixed"
        for r in data["recoveries"][:20]
    ) if data["recoveries"] else "_No error recoveries found in this cycle._"
    
    content = f"""---
type: evolution_report
date: {ts}
total_sessions: {data['total_sessions']}
total_messages: {data['total_messages']}
---

# Daily Evolution Report — {ts}

## Summary
- **Sessions processed:** {data['total_sessions']}
- **Messages analyzed:** {data['total_messages']}
- **Platforms:** {', '.join(f'{k} ({v})' for k, v in data['platforms'].items())}
- **Models used:** {', '.join(f'{k} ({v})' for k, v in data['models'].items())}

## Top Skills Used

| Skill | Uses |
|-------|------|
{skills_md}

## Common Tool Sequences

| Sequence | Frequency |
|----------|-----------|
{sequences_md}

## Error Recovery Patterns

{recoveries_md}

## User Preferences Extracted

"""
    
    for pref_type, values in data["preferences"].items():
        content += f"\n### {pref_type.replace('_', ' ').title()} ({len(values)} found)\n\n"
        for v in values[:10]:
            content += f"- {v[:150]}...\n"
    
    # Write to GBrain source
    page_path = GBRAIN_DIR / f"{slug}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(content, encoding='utf-8')
    
    return str(page_path)


# ── MAIN ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" HERMES SELF-EVOLUTION ENGINE v2 — Streaming Pipeline")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Session dir: {SESSIONS_DIR}")
    print(f"Workers: {MAX_WORKERS}")
    print()
    
    # Find all session files
    session_files = []
    for ext in ['*.jsonl', '*.jsonl.gz']:
        session_files.extend(SESSIONS_DIR.glob(ext))
    
    # Filter to recent files
    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    recent_files = []
    for f in session_files:
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime > cutoff:
                recent_files.append(f)
        except:
            recent_files.append(f)
    
    print(f"Found {len(session_files)} total session files")
    print(f"Recent (last {LOOKBACK_DAYS} days): {len(recent_files)}")
    print()
    
    # Process in parallel
    results = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_session_file, f): f for f in recent_files}
        
        for future in as_completed(futures):
            f = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                processed += 1
                if processed % 20 == 0:
                    print(f"  Progress: {processed}/{len(recent_files)} files...")
            except Exception as e:
                print(f"  Error processing {f.name}: {e}")
    
    print(f"\n✅ Processed {processed} files, extracted {len(results)} valid sessions")
    
    # Aggregate
    print("\nAggregating patterns...")
    aggregated = aggregate_results(results)
    
    print(f"  Total messages: {aggregated['total_messages']}")
    print(f"  Top skill: {aggregated['top_skills'][0] if aggregated['top_skills'] else 'none'}")
    print(f"  Recoveries found: {len(aggregated['recoveries'])}")
    
    # Write to GBrain
    print("\nWriting to GBrain...")
    page_path = write_gbrain_page(aggregated)
    print(f"  Written: {page_path}")
    
    # Trigger GBrain sync
    print("\nTriggering GBrain sync...")
    os.system(f"cd {GBRAIN_DIR} && gbrain sync --source evolution 2>/dev/null || echo 'GBrain sync attempted'")
    
    print("\n" + "=" * 60)
    print(" EVOLUTION CYCLE COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
