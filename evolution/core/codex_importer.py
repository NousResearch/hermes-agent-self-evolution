"""Import Codex CLI / Codex App session history into golden eval datasets.

Codex stores conversation history in `~/.codex/thread_history_1.sqlite` with
three tables:

    thread_items  — every item (user message, agent message, reasoning, tool
                    call, file change), keyed by (thread_id, turn_id,
                    rollout_ordinal)
    thread_turns  — turn boundaries and status
    thread_realtime_items — streaming-state mirror (unused here)

Unlike Claude Code (user inputs only), Codex retains full turn structure with
the assistant's own messages, so this yields user + final-assistant pairs like
the Copilot and Hermes importers.

This is the richest available source on a Codex-heavy machine: it includes the
full tool-call trace and the *corrections* the user issued after a wrong turn,
which are exactly the failure patterns an evolved skill should learn to avoid.

Safety: this DB is read-only opened. Secret-pattern filtering reuses the
shared `SECRET_PATTERNS` detector from external_importers.

Usage as standalone CLI:
    python -m evolution.core.codex_importer --skill my-skill --dry-run
    python -m evolution.core.codex_importer --skill my-skill \
        --model openrouter/google/gemini-2.5-flash
"""

import json
import re
import sqlite3
import statistics
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress

from evolution.core.dataset_builder import EvalExample, EvalDataset
from evolution.core.external_importers import (
    RelevanceFilter,
    _contains_secret,
    MIN_DATASET_SIZE,
    RECOMMENDED_DATASET_SIZE,
)

console = Console()

DEFAULT_DB = Path.home() / ".codex" / "thread_history_1.sqlite"

# ── Injected-context markers ─────────────────────────────────────────────
# Codex prepends harness state to the *first* user turn of a session. These
# are not user requests and must never become eval examples. Matched as a
# prefix after stripping.
INJECTED_PREFIXES = (
    "<environment_details>",
    "<model_switch>",
    "<multi_agent_role>",
    "<multi_agent_mode>",
    "<INSTRUCTIONS>",
    "<codex_app_instructions>",
    "<permissions instructions>",
    "<user_instructions>",
    "<collaboration_mode>",
    "<apps_instructions>",
    "<plugins_instructions>",
    "<skills_instructions>",
    "<hooks>",
    "<environment>",
    "# AGENTS.md instructions",
)

# Prefixes that mark a real user turn which merely *begins* with harness
# context. These are kept but the context block is stripped.
USER_PREFIX_KEEP = (
    "# Files mentioned by the user:",
    "# Files pasted by the user:",
)

MAX_TASK_INPUT = 2000
MAX_ASSISTANT = 3000
MIN_TASK_LEN = 15


def _strip_injected_context(text: str) -> str:
    """Remove harness-injected blocks from a user message body.

    Keeps the user's actual request, drops <model_switch>/<multi_agent_*>/
    <environment_details>/<INSTRUCTIONS> blocks that Codex prepends.
    """
    text = text.strip()

    # Drop whole blocks for tags Codex wraps its injected state in.
    for tag in (
        "environment_details", "model_switch", "multi_agent_role",
        "multi_agent_mode", "INSTRUCTIONS", "codex_app_instructions",
        "permissions instructions", "user_instructions",
        "collaboration_mode", "apps_instructions", "plugins_instructions",
        "skills_instructions", "hooks",
    ):
        text = re.sub(
            rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>",
            "", text, flags=re.DOTALL,
        )

    text = re.sub(r"^#\s*AGENTS\.md instructions\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_text(item: dict) -> str:
    """Pull plain text out of a userMessage item."""
    content = item.get("content", [])
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _is_real_user_turn(text: str) -> bool:
    """Reject harness state and trivially short turns."""
    stripped = text.strip()
    if len(stripped) < MIN_TASK_LEN:
        return False
    # A message that is *only* injected context is not a request.
    for prefix in INJECTED_PREFIXES:
        if stripped.startswith(prefix):
            return False
    return True


class CodexImporter:
    """Import user/final-assistant pairs from the Codex thread history DB."""

    DB_PATH = DEFAULT_DB

    @staticmethod
    def extract_messages(limit: int = 0, db_path: Optional[Path] = None) -> list[dict]:
        """Read user/assistant turn pairs from the Codex history DB.

        Args:
            limit: Maximum pairs to return (0 = no limit).
            db_path: Override the DB location.

        Returns:
            List of dicts with keys: source, task_input, assistant_response,
            session_id, turn_id.
        """
        path = Path(db_path) if db_path else CodexImporter.DB_PATH
        if not path.exists():
            console.print(f"[yellow]Codex DB not found at {path}[/yellow]")
            return []

        # Read-only URI — never mutate the user's Codex state.
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        try:
            # Pull every user and agent message, ordered so a turn's messages
            # are adjacent and in sequence.
            cur.execute(
                "SELECT thread_id, turn_id, rollout_ordinal, item_type, item_json "
                "FROM thread_items "
                "WHERE item_type IN ('userMessage', 'agentMessage') "
                "ORDER BY thread_id, rollout_ordinal"
            )
            rows = cur.fetchall()
        except sqlite3.Error as e:
            console.print(f"[red]Codex DB read failed: {e}[/red]")
            conn.close()
            return []

        # Group by (thread, turn) preserving ordinal order.
        turns: dict[tuple[str, str], list[tuple[int, str, dict]]] = {}
        for r in rows:
            try:
                item = json.loads(r["item_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            key = (r["thread_id"], r["turn_id"])
            turns.setdefault(key, []).append((r["rollout_ordinal"], r["item_type"], item))

        console.print(f"  Raw items: {len(rows)} across {len(turns)} turns")

        messages: list[dict] = []
        skipped_injected = 0
        skipped_secret = 0
        skipped_no_reply = 0

        with Progress() as progress:
            task = progress.add_task("Parsing Codex turns...", total=len(turns))
            for (thread_id, turn_id), items in turns.items():
                items.sort(key=lambda t: t[0])

                user_text = None
                final_response = ""
                for _, item_type, item in items:
                    if item_type == "userMessage" and user_text is None:
                        user_text = _extract_text(item)
                    elif item_type == "agentMessage":
                        text = (item.get("text") or "").strip()
                        # Keep the last substantive assistant message as the
                        # response of record; Codex emits commentary before it.
                        if text:
                            final_response = text

                if not user_text:
                    progress.update(task, advance=1)
                    continue

                cleaned = _strip_injected_context(user_text)
                if not _is_real_user_turn(cleaned):
                    skipped_injected += 1
                    progress.update(task, advance=1)
                    continue

                if _contains_secret(cleaned) or (
                    final_response and _contains_secret(final_response)
                ):
                    skipped_secret += 1
                    progress.update(task, advance=1)
                    continue

                messages.append({
                    "source": "codex",
                    "task_input": cleaned[:MAX_TASK_INPUT],
                    "assistant_response": final_response[:MAX_ASSISTANT],
                    "session_id": thread_id,
                    "turn_id": turn_id,
                })

                if not final_response:
                    skipped_no_reply += 1

                progress.update(task, advance=1)

                if limit and len(messages) >= limit:
                    break

        conn.close()

        if skipped_injected or skipped_secret:
            console.print(
                f"  Filtered: {skipped_injected} injected-context, "
                f"{skipped_secret} secret-bearing"
            )
        if skipped_no_reply:
            console.print(
                f"  Note: {skipped_no_reply} turns had no assistant reply "
                f"(user request retained, response empty)"
            )

        return messages

    @staticmethod
    def dataset_stats(messages: list[dict]) -> dict:
        """Summarize a message set for a dry-run report."""
        if not messages:
            return {}
        lens = [len(m["task_input"]) for m in messages]
        with_reply = sum(1 for m in messages if m.get("assistant_response"))
        return {
            "pairs": len(messages),
            "sessions": len({m["session_id"] for m in messages}),
            "median_task_len": int(statistics.median(lens)),
            "max_task_len": max(lens),
            "with_assistant_reply": with_reply,
        }


# ── CLI ───────────────────────────────────────────────────────────────────


@click.command()
@click.option("--skill", required=True, help="Skill name to generate eval data for")
@click.option("--output", type=click.Path(), default=None,
              help="Output directory (default: datasets/skills/<skill>/)")
@click.option("--model", default="openrouter/google/gemini-2.5-flash",
              help="LiteLLM model string for relevance scoring")
@click.option("--max-examples", default=RECOMMENDED_DATASET_SIZE, type=int)
@click.option("--db", type=click.Path(), default=None, help="Override Codex DB path")
@click.option("--dry-run", is_flag=True, help="Show counts without LLM scoring")
def main(skill, output, model, max_examples, db, dry_run):
    """Build an eval dataset for a skill from Codex session history."""
    console.print(f"\n[bold cyan]Codex Session Importer[/bold cyan] — skill: [bold]{skill}[/bold]\n")

    from evolution.core.external_importers import _load_skill_text
    try:
        skill_name, skill_text = _load_skill_text(skill)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    console.print(f"  Loaded skill: {skill_name} ({len(skill_text):,} chars)")
    console.print(f"  DB: {db or DEFAULT_DB}")

    messages = CodexImporter.extract_messages(db_path=db)

    if not messages:
        console.print("[red]No Codex messages found.[/red]")
        raise SystemExit(1)

    stats = CodexImporter.dataset_stats(messages)
    console.print(f"\n[bold]Extracted pairs: {stats['pairs']}[/bold]")
    console.print(f"  sessions: {stats['sessions']}")
    console.print(f"  median task length: {stats['median_task_len']} chars")
    console.print(f"  with assistant reply: {stats['with_assistant_reply']}/{stats['pairs']}")

    if dry_run:
        console.print("\n[bold green]DRY RUN — no LLM calls made.[/bold green]")
        return

    if output is None:
        output = Path("datasets") / "skills" / skill_name
    else:
        output = Path(output)

    relevance_filter = RelevanceFilter(model=model)
    examples = relevance_filter.filter_and_score(
        messages, skill_name, skill_text, max_examples=max_examples,
    )

    console.print(f"\n[bold green]Found {len(examples)} relevant examples[/bold green]")
    if not examples:
        console.print("[yellow]No relevant examples found.[/yellow]")
        return

    if len(examples) < MIN_DATASET_SIZE:
        console.print(
            f"[yellow]⚠ Only {len(examples)} examples (min {MIN_DATASET_SIZE} "
            f"recommended)[/yellow]"
        )

    import random
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * 0.5))
    n_val = max(1, int(n * 0.25))

    dataset = EvalDataset(
        train=examples[:n_train],
        val=examples[n_train:n_train + n_val],
        holdout=examples[n_train + n_val:],
    )
    dataset.save(output)
    console.print(f"\n[bold]Saved to {output}/[/bold]")
    console.print(
        f"  train: {len(dataset.train)}  val: {len(dataset.val)}  "
        f"holdout: {len(dataset.holdout)}"
    )


if __name__ == "__main__":
    main()