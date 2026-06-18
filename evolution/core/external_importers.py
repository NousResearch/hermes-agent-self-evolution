"""Import session data from external AI tools into golden eval datasets.

Bridges the gap between existing tool usage (Claude Code, GitHub Copilot)
and Hermes self-evolution by mining real session history for skill-relevant
evaluation examples. Solves the cold-start problem: new Hermes users don't
have golden datasets, but they do have session history from tools they
already use.

Supported sources:
  - Claude Code (~/.claude/history.jsonl) — user inputs only
  - GitHub Copilot (~/.copilot/session-state/*/events.jsonl) — full conversations
  - Hermes Agent (~/.hermes/sessions/*.json) — user + assistant + tool context

Usage as standalone CLI:
    python -m evolution.core.external_importers \\
        --source all --skill my-skill --dry-run

    python -m evolution.core.external_importers \\
        --source claude-code --skill my-skill --model openrouter/google/gemini-2.5-flash

Usage from evolve_skill.py:
    python -m evolution.skills.evolve_skill --skill my-skill --eval-source sessiondb
"""

import json
import re
import random
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.progress import Progress

from evolution.core.dataset_builder import EvalExample, EvalDataset

console = Console()

# ── Secret Detection ──────────────────────────────────────────────────────

# Patterns that indicate secrets — NEVER include these in datasets.
# Each pattern is intentionally anchored to known key formats to minimize
# false positives on normal prose.
SECRET_PATTERNS = re.compile(
    r'('
    r'sk-ant-api\S+'           # Anthropic API keys
    r'|sk-or-v1-\S+'          # OpenRouter API keys
    r'|sk-\S{20,}'            # Generic OpenAI-style keys (20+ chars after sk-)
    r'|ghp_\S+'               # GitHub personal access tokens
    r'|ghu_\S+'               # GitHub user tokens
    r'|xoxb-\S+'              # Slack bot tokens
    r'|xapp-\S+'              # Slack app tokens
    r'|ntn_\S+'               # Notion integration tokens
    r'|AKIA[0-9A-Z]{16}'      # AWS access key IDs
    r'|Bearer\s+\S{20,}'      # Bearer auth headers (20+ char tokens)
    r'|-----BEGIN\s+(RSA\s+)?PRIVATE\sKEY-----'  # PEM private keys
    r'|ANTHROPIC_API_KEY'      # Known env var names (exact match)
    r'|OPENAI_API_KEY'
    r'|OPENROUTER_API_KEY'
    r'|SLACK_BOT_TOKEN'
    r'|GITHUB_TOKEN'
    r'|AWS_SECRET_ACCESS_KEY'
    r'|DATABASE_URL'
    r'|\bpassword\s*[=:]\s*\S+' # password assignments (password=xxx, password: xxx)
    r'|\bsecret\s*[=:]\s*\S+'   # secret assignments (secret=xxx, secret: xxx)
    r'|\btoken\s*[=:]\s*\S{10,}' # token assignments with 10+ char values
    r')',
    re.IGNORECASE,
)


VALID_DIFFICULTIES = {"easy", "medium", "hard"}

MIN_DATASET_SIZE = 3  # Minimum examples needed to produce a meaningful split


def _contains_secret(text: str) -> bool:
    """Check if text contains potential API keys or tokens."""
    return bool(SECRET_PATTERNS.search(text))


def _validate_eval_example(
    task_input: str,
    expected_behavior: str,
    difficulty: str,
    category: str,
) -> Optional[dict]:
    """Validate and normalize fields before creating an EvalExample.

    Returns:
        Dict of validated fields, or None if the example should be skipped.
    """
    # task_input and expected_behavior must be non-empty
    if not task_input or not task_input.strip():
        return None
    if not expected_behavior or not expected_behavior.strip():
        return None

    # Normalize difficulty to a known value
    difficulty = difficulty.strip().lower() if difficulty else "medium"
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "medium"

    # Category must be non-empty
    category = category.strip() if category else "general"
    if not category:
        category = "general"

    # Cap task_input length to prevent bloated datasets
    task_input = task_input[:2000]

    return {
        "task_input": task_input,
        "expected_behavior": expected_behavior.strip(),
        "difficulty": difficulty,
        "category": category,
    }


def _is_relevant_to_skill(text: str, skill_name: str, skill_text: str) -> bool:
    """Quick heuristic check if a message might be relevant to a skill.

    Uses keyword overlap between the message and skill description/name.
    This is a cheap pre-filter before the LLM does proper relevance scoring.
    Returns True if the message shares enough vocabulary with the skill.
    """
    text_lower = text.lower()
    skill_lower = skill_name.lower().replace("-", " ").replace("_", " ")

    # Exact full skill name match (handles short names like "mcp", "tdd", "git")
    if skill_lower in text_lower:
        return True

    # Individual word match (only words > 3 chars to avoid false positives
    # from short fragments like "run", "use", etc.)
    for word in skill_lower.split():
        if len(word) > 3 and word in text_lower:
            return True

    # Extract meaningful keywords from skill text (first 500 chars)
    skill_keywords = set()
    for word in skill_text[:500].lower().split():
        word = re.sub(r'[^a-z]', '', word)
        if len(word) > 4:
            skill_keywords.add(word)

    # Require at least 2 keyword matches
    message_words = set(re.sub(r'[^a-z\s]', '', text_lower).split())
    overlap = message_words & skill_keywords
    return len(overlap) >= 2


# ── Importers ─────────────────────────────────────────────────────────────


class ClaudeCodeImporter:
    """Import user prompts from Claude Code history.jsonl.

    Claude Code stores a flat JSONL of user messages at ~/.claude/history.jsonl.
    Each line has: display (user text), timestamp, project, sessionId.
    Only user inputs are available — no assistant responses.
    """

    HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"

    @staticmethod
    def extract_messages(limit: int = 0) -> list[dict]:
        """Read user messages from Claude Code history.

        Args:
            limit: Maximum messages to return (0 = no limit).

        Returns:
            List of dicts with keys: source, task_input, project, session_id, timestamp.
        """
        if not ClaudeCodeImporter.HISTORY_PATH.exists():
            return []

        messages = []
        with open(ClaudeCodeImporter.HISTORY_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = entry.get("display", "")
                if not text or len(text) < 10:
                    continue
                if _contains_secret(text):
                    continue

                messages.append({
                    "source": "claude-code",
                    "task_input": text,
                    "project": entry.get("project", ""),
                    "session_id": entry.get("sessionId", ""),
                    "timestamp": entry.get("timestamp", 0),
                })

                if limit and len(messages) >= limit:
                    break

        return messages


class CopilotImporter:
    """Import conversations from GitHub Copilot session events.

    Copilot stores sessions at ~/.copilot/session-state/<session-id>/.
    Each session has workspace.yaml (project context) and events.jsonl
    (chronological stream of user.message / assistant.message events).
    Files can be 100MB+ so we stream line-by-line.

    Note: This path is the default Copilot CLI session storage location.
    Override SESSION_DIR for non-standard installations.
    """

    SESSION_DIR = Path.home() / ".copilot" / "session-state"

    @staticmethod
    def extract_messages(limit: int = 0) -> list[dict]:
        """Read user/assistant message pairs from Copilot sessions.

        Args:
            limit: Maximum messages to return (0 = no limit).

        Returns:
            List of dicts with keys: source, task_input, assistant_response,
            project, session_id.
        """
        if not CopilotImporter.SESSION_DIR.exists():
            return []

        messages = []
        event_files = list(CopilotImporter.SESSION_DIR.glob("*/events.jsonl"))

        with Progress() as progress:
            task = progress.add_task("Reading Copilot sessions...", total=len(event_files))

            for events_path in event_files:
                session_id = events_path.parent.name
                project = _read_copilot_workspace(events_path.parent / "workspace.yaml")

                pairs = _parse_copilot_events(events_path, session_id, project)
                messages.extend(pairs)

                progress.update(task, advance=1)

                if limit and len(messages) >= limit:
                    messages = messages[:limit]
                    break

        return messages


def _read_copilot_workspace(workspace_path: Path) -> str:
    """Extract cwd from a Copilot workspace.yaml file."""
    if not workspace_path.exists():
        return ""
    try:
        for line in workspace_path.read_text().split("\n"):
            if line.startswith("cwd:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _parse_copilot_events(
    events_path: Path, session_id: str, project: str,
) -> list[dict]:
    """Parse a single Copilot events.jsonl into user/assistant pairs."""
    pairs = []
    current_user_msg = None
    current_assistant_msg = None

    try:
        with open(events_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
                data = event.get("data", {})

                if event_type == "user.message":
                    # Save previous pair before starting new one
                    if current_user_msg and current_assistant_msg:
                        if not _contains_secret(current_user_msg) and not _contains_secret(current_assistant_msg):
                            pairs.append({
                                "source": "copilot",
                                "task_input": current_user_msg,
                                "assistant_response": current_assistant_msg,
                                "project": project,
                                "session_id": session_id,
                            })

                    current_user_msg = data.get("content", "")
                    current_assistant_msg = None

                elif event_type == "assistant.message":
                    content = data.get("content", "")
                    if content and current_user_msg:
                        if current_assistant_msg:
                            current_assistant_msg += "\n" + content
                        else:
                            current_assistant_msg = content

        # Don't forget the last pair in the file
        if current_user_msg and current_assistant_msg:
            if not _contains_secret(current_user_msg) and not _contains_secret(current_assistant_msg):
                pairs.append({
                    "source": "copilot",
                    "task_input": current_user_msg,
                    "assistant_response": current_assistant_msg,
                    "project": project,
                    "session_id": session_id,
                })

    except Exception as e:
        console.print(f"[dim]Skipped {session_id}: {e}[/dim]")

    return pairs


class HermesSessionImporter:
    """Import conversations from Hermes Agent session files.

    Hermes stores session transcripts as JSON files in ~/.hermes/sessions/.
    Each file contains an OpenAI-format message list with user, assistant,
    and tool messages — providing richer signal than Claude Code (user-only)
    or Copilot (user+assistant without tool context).

    This mines user messages paired with the assistant's final response,
    giving the LLM judge both the task and how it was actually handled.
    """

    SESSION_DIR = Path.home() / ".hermes" / "sessions"

    @staticmethod
    def extract_messages(limit: int = 0) -> list[dict]:
        """Read user/assistant pairs from Hermes session files.

        Args:
            limit: Maximum messages to return (0 = no limit).

        Returns:
            List of dicts with keys: source, task_input, assistant_response,
            session_id.
        """
        if not HermesSessionImporter.SESSION_DIR.exists():
            return []

        messages = []
        session_files = sorted(
            HermesSessionImporter.SESSION_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # newest first
        )

        for session_file in session_files:
            try:
                data = json.loads(session_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            msg_list = data.get("messages", [])
            if not msg_list:
                continue

            session_id = data.get("session_id", session_file.stem)

            # Walk messages: pair each user message with the next assistant
            # response (skipping tool messages in between).
            for i, msg in enumerate(msg_list):
                if msg.get("role") != "user":
                    continue
                user_text = msg.get("content", "")
                if not user_text or len(user_text) < 10:
                    continue
                if _contains_secret(user_text):
                    continue

                # Find the next assistant response
                assistant_text = ""
                for j in range(i + 1, len(msg_list)):
                    if msg_list[j].get("role") == "assistant":
                        content = msg_list[j].get("content", "")
                        if content:
                            assistant_text = content
                            break
                    elif msg_list[j].get("role") == "user":
                        break  # next user turn, no assistant response found

                if assistant_text and _contains_secret(assistant_text):
                    continue

                messages.append({
                    "source": "hermes",
                    "task_input": user_text,
                    "assistant_response": assistant_text,
                    "session_id": session_id,
                })

                if limit and len(messages) >= limit:
                    return messages

        return messages


# ── Relevance Filtering ───────────────────────────────────────────────────


class RelevanceFilter:
    """Use LLM-as-judge to determine which messages are relevant to a skill.

    Two-stage pipeline:
      1. Cheap heuristic pre-filter (_is_relevant_to_skill)
      2. LLM scoring for final relevance + eval metadata generation
    """

    class ScoreRelevance(dspy.Signature):
        """Score whether a user message is relevant to a specific agent skill.

        Return a JSON object with:
        - relevant: boolean (true if the message relates to what this skill does)
        - expected_behavior: string (if relevant, what should a good response do?)
        - difficulty: string (easy, medium, or hard)
        - category: string (what aspect of the skill this tests)
        """
        skill_name: str = dspy.InputField(desc="Name of the skill")
        skill_description: str = dspy.InputField(desc="First 800 chars of the skill file")
        user_message: str = dspy.InputField(desc="The user's message to evaluate")
        assistant_response: str = dspy.InputField(desc="The assistant's actual response (may be empty)")
        scoring: str = dspy.OutputField(desc="JSON object with: relevant, expected_behavior, difficulty, category")

    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.scorer = dspy.ChainOfThought(self.ScoreRelevance)
        self.model = model
        self.api_base = api_base
        self.api_key = api_key

    def filter_and_score(
        self,
        messages: list[dict],
        skill_name: str,
        skill_text: str,
        max_examples: int = 50,
    ) -> list[EvalExample]:
        """Filter messages by relevance and generate eval examples.

        Args:
            messages: Raw messages from importers.
            skill_name: Name of the target skill.
            skill_text: Full text of the SKILL.md file.
            max_examples: Maximum eval examples to produce.

        Returns:
            List of EvalExample objects for relevant messages.
        """
        skill_desc = skill_text[:800]

        # Stage 0: drop messages missing required fields
        messages = [m for m in messages if m.get("task_input") and m.get("source")]

        # Stage 1: cheap heuristic pre-filter
        candidates = [
            m for m in messages
            if _is_relevant_to_skill(m["task_input"], skill_name, skill_text)
        ]

        # If heuristics found too few, sample remaining messages
        if len(candidates) < max_examples:
            candidate_ids = {id(m) for m in candidates}
            remaining = [m for m in messages if id(m) not in candidate_ids]
            random.shuffle(remaining)
            candidates.extend(remaining[:max_examples * 2])

        # Cap candidates to control LLM costs
        candidates = candidates[:max_examples * 3]

        console.print(f"  Pre-filtered to {len(candidates)} candidates (from {len(messages)} total)")

        # Stage 2: LLM relevance scoring
        examples = []
        errors = 0
        lm = dspy.LM(
            self.model,
            api_base=self.api_base,
            api_key=self.api_key,
        )

        with Progress() as progress:
            task = progress.add_task("Scoring relevance...", total=len(candidates))

            for msg in candidates:
                try:
                    with dspy.context(lm=lm):
                        result = self.scorer(
                            skill_name=skill_name,
                            skill_description=skill_desc,
                            user_message=msg["task_input"][:1000],
                            assistant_response=msg.get("assistant_response", "")[:1000],
                        )

                    scoring = _parse_scoring_json(result.scoring)
                    if scoring is None:
                        errors += 1
                        progress.update(task, advance=1)
                        continue

                    if scoring.get("relevant", False):
                        validated = _validate_eval_example(
                            task_input=msg["task_input"],
                            expected_behavior=scoring.get("expected_behavior", ""),
                            difficulty=scoring.get("difficulty", "medium"),
                            category=scoring.get("category", "general"),
                        )
                        if validated:
                            examples.append(EvalExample(
                                source=msg["source"],
                                **validated,
                            ))

                except Exception:
                    errors += 1

                progress.update(task, advance=1)

                if len(examples) >= max_examples:
                    break

        # Report error rate so users know if the LLM is misbehaving
        total_scored = len(candidates)
        if errors > 0:
            console.print(
                f"  [yellow]LLM scoring: {errors}/{total_scored} failed "
                f"({errors / max(1, total_scored) * 100:.0f}% error rate)[/yellow]"
            )

        return examples


def _parse_scoring_json(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM scoring output.

    Strategy:
      1. Try direct json.loads (handles clean LLM output)
      2. Fall back to regex extraction (handles text-wrapped or fenced JSON)

    Returns:
        Parsed dict or None if no valid JSON found.
    """
    if not text:
        return None

    # Fast path: LLM returned clean JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Slow path: find balanced {...} block using brace counting.
    # Simple regex like r'\{[^}]+\}' breaks on nested braces
    # (e.g. "handle {edge} cases" in a string value).
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None

    return None


# ── Orchestration ─────────────────────────────────────────────────────────


def build_dataset_from_external(
    skill_name: str,
    skill_text: str,
    sources: list[str],
    output_path: Path,
    model: str,
    max_examples: int = 50,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EvalDataset:
    """Extract messages from external tools, filter for relevance, and save.

    This is the main entry point called by both the standalone CLI and
    evolve_skill.py when --eval-source sessiondb is used.

    Args:
        skill_name: Name of the target skill.
        skill_text: Full text of the SKILL.md file.
        sources: List of source names ("claude-code", "copilot").
        output_path: Directory to write train/val/holdout JSONL files.
        model: LiteLLM model string for relevance scoring.
        max_examples: Maximum eval examples to generate.
        api_base: Optional API base URL for custom endpoints.
        api_key: Optional API key for custom endpoints.

    Returns:
        EvalDataset with train/val/holdout splits.
    """
    all_messages = []

    importers = {
        "claude-code": ("Claude Code", ClaudeCodeImporter),
        "copilot": ("Copilot", CopilotImporter),
        "hermes": ("Hermes Agent", HermesSessionImporter),
    }

    for source in sources:
        if source not in importers:
            continue
        label, importer_cls = importers[source]
        console.print(f"\n[bold]Importing from {label}...[/bold]")
        msgs = importer_cls.extract_messages()
        console.print(f"  Found {len(msgs)} messages")
        all_messages.extend(msgs)

    if not all_messages:
        console.print("[red]No messages found from any source.[/red]")
        return EvalDataset()

    console.print(f"\n[bold]Total messages: {len(all_messages)}[/bold]")
    console.print(f"[bold]Filtering for relevance to skill: {skill_name}[/bold]")

    relevance_filter = RelevanceFilter(model=model, api_base=api_base, api_key=api_key)
    examples = relevance_filter.filter_and_score(
        all_messages, skill_name, skill_text, max_examples=max_examples,
    )

    console.print(f"\n[bold green]Found {len(examples)} relevant examples[/bold green]")

    if not examples:
        console.print("[yellow]No relevant examples found. Try a different skill or broader sources.[/yellow]")
        return EvalDataset()

    if len(examples) < MIN_DATASET_SIZE:
        console.print(
            f"[yellow]⚠ Only {len(examples)} examples found (minimum {MIN_DATASET_SIZE} "
            f"recommended for meaningful train/val/holdout split)[/yellow]"
        )

    # Split into train/val/holdout (50/25/25)
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * 0.5))
    n_val = max(1, int(n * 0.25))

    dataset = EvalDataset(
        train=examples[:n_train],
        val=examples[n_train:n_train + n_val],
        holdout=examples[n_train + n_val:],
    )

    dataset.save(output_path)
    console.print(f"\n[bold]Saved to {output_path}/[/bold]")
    console.print(f"  train: {len(dataset.train)}  val: {len(dataset.val)}  holdout: {len(dataset.holdout)}")

    source_counts: dict[str, int] = {}
    for ex in examples:
        source_counts[ex.source] = source_counts.get(ex.source, 0) + 1
    for src, count in sorted(source_counts.items()):
        console.print(f"  {src}: {count}")

    return dataset


def _load_skill_text(skill_name: str, skills_dir: Optional[Path] = None) -> tuple[str, str]:
    """Load skill text from the installed Hermes skills directory.

    This is used by the standalone CLI only. When called via evolve_skill.py,
    skill loading goes through skill_module.find_skill() + load_skill() instead,
    which searches the hermes-agent repo path rather than installed skills.

    Args:
        skill_name: Name of the skill directory.
        skills_dir: Override skills directory (default: ~/.hermes/skills).

    Returns:
        Tuple of (skill_name, skill_file_contents).

    Raises:
        FileNotFoundError: If no SKILL.md found for the given name.
    """
    if skills_dir is None:
        skills_dir = Path.home() / ".hermes" / "skills"

    # Try direct match, then subdirectory search
    for pattern in [skill_name, f"*/{skill_name}"]:
        for skill_dir in skills_dir.glob(pattern):
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                return skill_name, skill_file.read_text()

    raise FileNotFoundError(f"Skill '{skill_name}' not found in {skills_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--source",
    type=click.Choice(["claude-code", "copilot", "hermes", "all"]),
    default="all",
    help="Which tool to import from",
)
@click.option("--skill", required=True, help="Skill name to generate eval data for")
@click.option("--output", type=click.Path(), default=None,
              help="Output directory (default: datasets/skills/<skill>/)")
@click.option("--model", default="openrouter/google/gemini-2.5-flash",
              help="LiteLLM model string for relevance scoring")
@click.option("--max-examples", default=50, help="Max eval examples to generate")
@click.option("--dry-run", is_flag=True, help="Show message counts without LLM scoring")
@click.option("--api-base", default=None,
              help="Optional API base URL for custom endpoints")
@click.option("--api-key", default=None,
              help="Optional API key for custom endpoints")
def main(source, skill, output, model, max_examples, dry_run, api_base, api_key):
    """Import external session data into golden eval datasets for self-evolution."""
    console.print(f"\n[bold cyan]External Session Importer[/bold cyan] — skill: [bold]{skill}[/bold]\n")

    try:
        skill_name, skill_text = _load_skill_text(skill)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    console.print(f"  Loaded skill: {skill_name} ({len(skill_text):,} chars)")

    sources = [source] if source != "all" else ["claude-code", "copilot", "hermes"]

    if dry_run:
        importers = {
            "claude-code": ClaudeCodeImporter,
            "copilot": CopilotImporter,
            "hermes": HermesSessionImporter,
        }
        for src in sources:
            msgs = importers[src].extract_messages()
            console.print(f"  {src}: {len(msgs)} messages")
        console.print("\n[bold green]DRY RUN — no LLM calls made.[/bold green]")
        return

    if output is None:
        output = Path(__file__).parent.parent.parent / "datasets" / "skills" / skill_name
    else:
        output = Path(output)

    build_dataset_from_external(
        skill_name=skill_name,
        skill_text=skill_text,
        sources=sources,
        output_path=output,
        model=model,
        max_examples=max_examples,
        api_base=api_base,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
