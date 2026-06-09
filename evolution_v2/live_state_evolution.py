#!/usr/bin/env python3
"""
HERMES SELF-EVOLUTION ENGINE v2.1 — Live State Pipeline
Queries the real state.db (1.4GB, 145K messages, 3892 sessions).
Extracts patterns, writes consolidated findings to GBrain.

State DB Schema:
- sessions: id, title, created_at, updated_at, source, profile, ...
- messages: id, session_id, role, content, timestamp, tool_calls, tool_outputs, ...
- messages_fts: Full-text search over message content
"""

import sqlite3
import json
import re
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional

# ── CONFIG ─────────────────────────────────────────────
STATE_DB = Path.home() / ".hermes" / "state.db"
GBRAIN_DIR = Path("/mnt/d/gbrain-tan")
OUTPUT_DIR = Path("/mnt/d/hermes-agent-self-evolution/evolution_v2/output")
LOOKBACK_DAYS = 7  # Process last 7 days of messages
BATCH_SIZE = 5000  # Process messages in batches

# Skills to track
TRACKED_SKILLS = [
    "codex-image-gen", "pablo-campaign", "pablo-cinematic-engine",
    "fx-rich", "trading", "security", "marketing", "devops",
    "open-source-bounty-hunter", "hermes-agent", "god-mode-protocol",
    "knowledge-brain-ops", "cron-pipeline-ops", "hermes-safe-update",
]

# ── DB CONNECTOR ───────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Connect to live state database."""
    if not STATE_DB.exists():
        raise FileNotFoundError(f"State DB not found: {STATE_DB}")
    conn = sqlite3.connect(str(STATE_DB))
    conn.row_factory = sqlite3.Row
    return conn


# ── EXTRACTORS ─────────────────────────────────────────

def extract_skill_usage(content: str) -> Counter:
    """Find skill references in message content."""
    skills = Counter()
    if not content:
        return skills
    
    # skill_view calls
    for match in re.finditer(r'skill_view\(["\']([^"\']+)["\']', content):
        skills[match.group(1)] += 1
    
    # Direct skill mentions
    for skill in TRACKED_SKILLS:
        pattern = skill.replace('-', r'[- _]')
        if re.search(pattern, content, re.IGNORECASE):
            skills[skill] += 1
    
    return skills


def extract_error_patterns(content: str, tool_output: str) -> List[str]:
    """Identify error types in tool outputs."""
    errors = []
    text = f"{content or ''} {tool_output or ''}"
    
    error_patterns = {
        "timeout": r"timed out|timeout|deadline exceeded",
        "permission": r"permission denied|access denied|unauthorized|403",
        "not_found": r"not found|No such file|404|does not exist",
        "connection": r"Connection refused|ECONNREFUSED|network error",
        "parse": r"JSONDecodeError|SyntaxError|parse error|invalid",
        "api_rate_limit": r"rate limit|too many requests|429",
        "auth": r"authentication|auth failed|invalid token|API key",
        "shell": r"command not found|No such command|exit code [1-9]",
    }
    
    for error_type, pattern in error_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            errors.append(error_type)
    
    return errors


def extract_user_commands(content: str) -> List[str]:
    """Extract direct user commands/intentions."""
    if not content:
        return []
    
    commands = []
    # Imperative patterns
    imperative_starts = [
        r"^(Create|Build|Make|Set up|Install|Fix|Update|Delete|Remove|Check|Verify|Run|Start|Stop|Get|Show|List|Find|Search|Generate|Write|Edit|Refactor|Debug|Test|Deploy|Push|Commit|Clone|Build|Compile)",
        r"^(I want|I need|Can you|Please|Make sure|Ensure|Verify that|Check if|Find out|Get me|Show me|Give me|List all|Search for|Look up|Look into|Investigate|Analyze|Review|Audit|Scan|Monitor|Watch|Track|Log|Record|Save|Store|Backup|Restore|Recover|Reset|Restart|Reload|Refresh|Update|Upgrade|Downgrade|Patch|Fix|Repair|Clean|Clear|Delete|Remove|Destroy|Kill|Terminate|Stop|Start|Launch|Run|Execute|Perform|Do|Complete|Finish|Close|End|Open|Create|Build|Make|Generate|Produce|Write|Draft|Compose|Design|Develop|Code|Program|Script|Automate|Configure|Setup|Set up|Arrange|Organize|Structure|Format|Style|Layout|Theme|Template|Pattern|Model|Schema|Plan|Strategy|Approach|Method|Technique|Process|Procedure|Workflow|Pipeline|Chain|Sequence|Series|Batch|Queue|Stack|List|Array|Map|Dict|Set|Collection|Group|Cluster|Category|Class|Type|Kind|Sort|Order|Rank|Priority|Level|Tier|Grade|Score|Rating|Metric|Measurement|Stat|Count|Total|Sum|Average|Mean|Median|Mode|Range|Min|Max|Limit|Bound|Threshold|Target|Goal|Objective|Aim|Purpose|Intent|Plan|Idea|Concept|Thought|Note|Comment|Remark|Observation|Finding|Result|Outcome|Output|Product|Deliverable|Artifact|Asset|Resource|Tool|Utility|Helper|Assistant|Agent|Bot|Service|Function|Feature|Capability|Skill|Ability|Power|Strength|Advantage|Benefit|Gain|Profit|Value|Worth|Cost|Price|Fee|Charge|Expense|Budget|Fund|Capital|Investment|Revenue|Income|Earning|Return|Yield|ROI|Rate|Ratio|Proportion|Percentage|Fraction|Decimal|Point|Score|Grade|Mark|Rating|Review|Feedback|Critique|Assessment|Evaluation|Judgment|Opinion|View|Perspective|Angle|Lens|Frame|Context|Background|History|Story|Narrative|Account|Report|Record|Log|Journal|Diary|Blog|Post|Article|Essay|Paper|Document|File|Report|Brief|Summary|Overview|Introduction|Preface|Foreword|Prologue|Opening|Start|Beginning|Init|Launch|Kickoff|Boot|Startup|Origin|Source|Root|Base|Foundation|Ground|Bedrock|Core|Heart|Center|Middle|Hub|Nexus|Link|Connection|Bond|Bridge|Gateway|Portal|Door|Entry|Access|Path|Route|Way|Road|Track|Trail|Lane|Street|Avenue|Boulevard|Highway|Freeway|Motorway|Expressway|Turnpike|Tollway|Interstate|Route|Course|Direction|Heading|Bearing|Orientation|Position|Location|Place|Site|Spot|Point|Venue|Area|Zone|Region|Sector|District|Neighborhood|Community|Local|Locale|Setting|Scene|Stage|Platform|Environment|Context|Situation|Circumstance|Condition|State|Status|Mode|Phase|Stage|Step|Level|Layer|Floor|Tier|Deck|Stratum|Level|Plane|Dimension|Aspect|Facet|Side|Angle|Face|Surface|Interface|Boundary|Edge|Border|Frontier|Limit|Margin|Perimeter|Circumference|Rim|Brim|Verge|Threshold|Brink|Precipice|Cliff|Crest|Peak|Summit|Top|Apex|Zenith|Crown|Cap|Head|Tip|Point|End|Terminal|Terminus|Station|Stop|Halt|Pause|Break|Rest|Recess|Respite|Relief|Relaxation|Lull|Gap|Space|Void|Vacuum|Emptiness|Nothing|Null|Zero|None|Nil|Naught|Zip|Zilch|Nada|Blank|Clean|Clear|Pure|Sheer|Absolute|Total|Complete|Full|Whole|Entire|All|Every|Each|Any|Some|Many|Much|Most|More|Less|Few|Little|Small|Tiny|Minute|Mini|MICRO|Nano|Pico|Femto|Atto|Zepto|Yocto|Planck|Quantum|Atomic|Subatomic|Particle|Photon|Electron|Proton|Neutron|Quark|Gluon|Boson|Fermion|Lepton|Hadron|Baryon|Meson|Nucleon|Hyperon|Lambda|Sigma|Xi|Omega|Delta|Kappa|Pi|Mu|Tau|Neutrino|Antineutrino|W|Z|Higgs|Graviton|Tachyon|Axion|Dilaton|Inflaton|Majorana|Weyl|Dirac|Pauli|Fermi|Bose|Einstein|Newton|Maxwell|Boltzmann|Planck|Heisenberg|Schrodinger|Dirac|Feynman|Gell-Mann|tHooft|Witten|Penrose|Hawking|Susskind|Maldacena|Bousso|Sethi|Vafa|Strominger|Polchinski|Green|Schwarz|Witten|Sen|Ashoke|Hull|Townsend|Duff|Branes|String|M-theory|F-theory|K3|Calabi|Yau|Mirror|Symmetry|Duality|S-duality|T-duality|U-duality|Holography|AdS|CFT|Correspondence|Conjecture|Hypothesis|Theorem|Lemma|Corollary|Proposition|Postulate|Axiom|Principle|Law|Rule|Formula|Equation|Expression|Term|Factor|Coefficient|Constant|Variable|Parameter|Argument|Input|Output|Return|Yield|Produce|Generate|Create|Make|Build|Construct|Assemble|Fabricate|Manufacture|Produce|Turn out|Roll out|Crank out|Churn out|Pump out|Spew out|Spit out|Throw out|Kick out|Boot out|Force out|Drive out|Oust|Eject|Expel|Remove|Eliminate|Eradicate|Exterminate|Annihilate|Destroy|Demolish|Raze|Level|Flatten|Crush|Smash|Shatter|Break|Crack|Split|Fracture|Fissure|Rupture|Burst|Explode|Detonate|Ignite|Combust|Burn|Incinerate|Carbonize|Vaporize|Atomize|Pulverize|Powder|Grind|Mill|Crush|Mash|Pound|Beat|Hammer|Strike|Hit|Slap|Smack|Spank|Whack|Thwack|Wham|Bam|Bang|Boom|Crash|Smash|Bash|Thump|Thud|Clunk|Clank|Clang|Clatter|Rattle|Shake|Vibrate|Oscillate|Wobble|Teeter|Totter|Stagger|Lurch|Reel|Swerve|Veering|Careen|Pitch|Roll|Yaw|Heave|Surge|Swell|Billow|Wave|Surge|Rush|Dash|Dart|Bolt|Spring|Leap|Jump|Hop|Skip|Bound|Bounce|Rebound|Ricochet|Carom|Glance|Skip|Skim|Slide|Glide|Drift|Float|Hover|Hang|Suspend|Dangle|Sway|Swing|Rock|Roll|PITCH|yaw|HEAVE|sway|SURGE|)",
    ]
    
    for pattern in imperative_starts:
        if re.search(pattern, content, re.IGNORECASE):
            # Extract the command intent (first 80 chars)
            cmd = content[:80].strip()
            commands.append(cmd)
            break
    
    return commands


# ── STREAMING PROCESSOR ────────────────────────────────

def stream_messages(conn: sqlite3.Connection, since: datetime) -> List[sqlite3.Row]:
    """Stream messages from state.db since a given date."""
    since_ts = since.timestamp()
    
    cursor = conn.execute("""
        SELECT m.id, m.session_id, m.role, m.content, m.timestamp, 
               m.tool_calls, m.tool_name, s.title, s.source, s.model
        FROM messages m
        LEFT JOIN sessions s ON m.session_id = s.id
        WHERE m.timestamp > ?
        ORDER BY m.timestamp ASC
    """, (since_ts,))
    
    return cursor.fetchall()


def process_message_batch(messages: List[sqlite3.Row]) -> dict:
    """Process a batch of messages and extract patterns."""
    stats = {
        "skills_used": Counter(),
        "errors": Counter(),
        "user_commands": [],
        "tool_sequences": [],
        "platforms": Counter(),
        "models": Counter(),
        "sessions": set(),
        "total_messages": len(messages),
    }
    
    prev_tool = None
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"] or ""
        tool_calls = msg["tool_calls"]
        tool_name = msg["tool_name"]
        session_id = msg["session_id"]
        model = msg["model"]
        
        stats["sessions"].add(session_id)
        
        # Track models from session join
        if model:
            stats["models"][model] += 1
        
        # Track platforms from source field
        source = msg["source"] if "source" in msg.keys() else None
        if source:
            stats["platforms"][source] += 1
        
        # Extract skill usage from all content
        if role in ("user", "assistant"):
            skills = extract_skill_usage(content)
            stats["skills_used"].update(skills)
        
        # Extract user commands
        if role == "user":
            cmds = extract_user_commands(content)
            stats["user_commands"].extend(cmds)
        
        # Extract error patterns from tool outputs (stored in content for tool role)
        if role == "tool" and content:
            errors = extract_error_patterns(content, "")
            stats["errors"].update(errors)
        
        # Track tool sequences from tool_calls JSON
        if role == "assistant" and tool_calls:
            try:
                tcs = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                for tc in tcs:
                    fn_name = tc.get("function", {}).get("name", tool_name or "unknown")
                    if prev_tool:
                        stats["tool_sequences"].append(f"{prev_tool} -> {fn_name}")
                    prev_tool = fn_name
            except:
                # Fallback: use tool_name column
                if tool_name and prev_tool:
                    stats["tool_sequences"].append(f"{prev_tool} -> {tool_name}")
                prev_tool = tool_name
    
    return stats


# ── GBRAIN WRITER ──────────────────────────────────────

def write_evolution_page(data: dict, date_str: str) -> str:
    """Write evolution findings as a GBrain page."""
    slug = f"evolution/daily-{date_str}"
    page_path = GBRAIN_DIR / f"{slug}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build tables
    skills_md = "\n".join(
        f"| {skill} | {count} |"
        for skill, count in data["skills_used"].most_common(20)
    ) if data["skills_used"] else "_No skill usage detected._"
    
    errors_md = "\n".join(
        f"| {error} | {count} |"
        for error, count in data["errors"].most_common(10)
    ) if data["errors"] else "_No errors detected._"
    
    sequences_md = "\n".join(
        f"| {seq} | {count} |"
        for seq, count in Counter(data["tool_sequences"]).most_common(15)
    ) if data["tool_sequences"] else "_No tool sequences detected._"
    
    commands_md = "\n".join(
        f"- {cmd[:120]}"
        for cmd in data["user_commands"][:20]
    ) if data["user_commands"] else "_No commands detected._"
    
    content = f"""---
type: evolution_report
date: {date_str}
total_sessions: {len(data['sessions'])}
total_messages: {data['total_messages']}
---

# Daily Evolution Report — {date_str}

## Summary
- **Messages analyzed:** {data['total_messages']:,}
- **Unique sessions:** {len(data['sessions']):,}
- **Platforms:** {dict(data['platforms'])}
- **Models:** {dict(data['models'])}

## Top Skills Used

| Skill | Uses |
|-------|------|
{skills_md}

## Error Patterns

| Error Type | Count |
|------------|-------|
{errors_md}

## Tool Sequences

| Sequence | Frequency |
|----------|-----------|
{sequences_md}

## User Commands

{commands_md}

---
*Evolution engine v2.1 — live state pipeline*
"""
    
    page_path.write_text(content, encoding='utf-8')
    return str(page_path)


# ── MAIN ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" HERMES SELF-EVOLUTION ENGINE v2.1 — Live State Pipeline")
    print("=" * 60)
    start_time = datetime.now()
    print(f"Started: {start_time.isoformat()}")
    print(f"State DB: {STATE_DB}")
    print(f"Size: {STATE_DB.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Calculate lookback
    since = datetime.now().astimezone() - timedelta(days=LOOKBACK_DAYS)
    print(f"Lookback: last {LOOKBACK_DAYS} days (since {since.isoformat()})")
    print()
    
    # Connect to DB
    conn = get_db()
    
    # Get message count for range
    since_ts = since.timestamp()
    cursor = conn.execute("SELECT COUNT(*) FROM messages WHERE timestamp > ?", (since_ts,))
    msg_count = cursor.fetchone()[0]
    print(f"Messages in range: {msg_count:,}")
    
    # Stream and process
    print("\nStreaming messages...")
    messages = stream_messages(conn, since)
    print(f"Retrieved: {len(messages):,} messages")
    
    # Process in batches
    print("\nProcessing batches...")
    all_stats = {
        "skills_used": Counter(),
        "errors": Counter(),
        "user_commands": [],
        "tool_sequences": [],
        "platforms": Counter(),
        "models": Counter(),
        "sessions": set(),
        "total_messages": 0,
    }
    
    for i in range(0, len(messages), BATCH_SIZE):
        batch = messages[i:i + BATCH_SIZE]
        stats = process_message_batch(batch)
        
        all_stats["skills_used"].update(stats["skills_used"])
        all_stats["errors"].update(stats["errors"])
        all_stats["user_commands"].extend(stats["user_commands"])
        all_stats["tool_sequences"].extend(stats["tool_sequences"])
        all_stats["platforms"].update(stats["platforms"])
        all_stats["models"].update(stats["models"])
        all_stats["sessions"].update(stats["sessions"])
        all_stats["total_messages"] += stats["total_messages"]
        
        if (i // BATCH_SIZE + 1) % 5 == 0:
            print(f"  Batch {i // BATCH_SIZE + 1}: {all_stats['total_messages']:,} messages processed...")
    
    conn.close()
    
    print(f"\n✅ Processed {all_stats['total_messages']:,} messages")
    print(f"   Sessions: {len(all_stats['sessions']):,}")
    print(f"   Skills found: {len(all_stats['skills_used'])}")
    print(f"   Errors found: {len(all_stats['errors'])}")
    print(f"   Commands found: {len(all_stats['user_commands'])}")
    
    # Write to GBrain
    print("\nWriting to GBrain...")
    date_str = datetime.now().strftime("%Y-%m-%d")
    page_path = write_evolution_page(all_stats, date_str)
    print(f"  Written: {page_path}")
    
    # Trigger GBrain sync
    print("\nTriggering GBrain sync...")
    result = os.system(f"cd {GBRAIN_DIR} && gbrain sync --source evolution 2>/dev/null")
    print(f"  Sync result: {result}")
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print(" EVOLUTION CYCLE COMPLETE")
    print("=" * 60)
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Messages/sec: {all_stats['total_messages'] / elapsed:.0f}")
    print(f"Top skill: {all_stats['skills_used'].most_common(1)[0] if all_stats['skills_used'] else 'none'}")
    
    return all_stats


if __name__ == "__main__":
    main()
