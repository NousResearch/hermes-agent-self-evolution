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
# false positives on normal prose. This is a defence-in-depth heuristic, not
# an authoritative scanner — pair with detect-secrets/gitleaks for production
# scans of any output that ships externally.
SECRET_PATTERNS = re.compile(
    r'('
    # OpenAI / Anthropic / OpenRouter
    r'sk-ant-api\S+'                                    # Anthropic
    r'|sk-or-v1-\S+'                                    # OpenRouter
    r'|sk-\S{8,}'                                       # OpenAI-style (and Stripe sk_)
    # GitHub — keep prefix-based detection loose so short tokens still trip
    r'|gh[pousr]_\S+'                                   # PAT / user / oauth / server / refresh
    # GitLab
    r'|glpat-[A-Za-z0-9_\-]{20,}'
    # Slack — separate alternations so xapp- / xoxb- / xoxp- all match
    r'|xoxb-\S+'
    r'|xoxp-\S+'
    r'|xoxa-\S+'
    r'|xoxr-\S+'
    r'|xoxs-\S+'
    r'|xapp-\S+'
    # Notion (modern + legacy)
    r'|ntn_[A-Za-z0-9]+'
    r'|secret_[A-Za-z0-9]{43}'
    # AWS
    r'|AKIA[0-9A-Z]{16}'                                # access key id
    r'|ASIA[0-9A-Z]{16}'                                # session/temporary access key id
    # Google API key
    r'|AIza[0-9A-Za-z_\-]{35}'
    # Stripe — explicit live/test variants (also covered by sk-\S{8,} above)
    r'|rk_(?:live|test)_[A-Za-z0-9]{20,}'
    r'|pk_(?:live|test)_[A-Za-z0-9]{20,}'
    # Twilio
    r'|AC[a-f0-9]{32}'
    # SendGrid
    r'|SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}'
    # Mailgun
    r'|key-[a-f0-9]{32}'
    # Databricks
    r'|dapi[a-f0-9]{32}'
    # DigitalOcean
    r'|dop_v1_[a-f0-9]{64}'
    # npm
    r'|npm_[A-Za-z0-9]{36}'
    # PyPI
    r'|pypi-[A-Za-z0-9_\-]{16,}'
    # Hashicorp Vault
    r'|hvs\.[A-Za-z0-9_\-]{24,}'
    # Telegram bot tokens
    r'|\d{8,10}:[A-Za-z0-9_\-]{35}'
    # Supabase
    r'|sbp_[A-Za-z0-9]{40,}'
    # Vercel
    r'|vercel_[A-Za-z0-9_\-]{24,}'
    # Generic Bearer / private key / JWT
    r'|Bearer\s+\S{20,}'
    r'|-----BEGIN\s+(?:(?:RSA|DSA|EC|OPENSSH|PGP)\s+)?PRIVATE\s+KEY-----'
    r'|eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+'   # JWT (3-part)
    # Connection strings with embedded credentials
    r'|(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:]+:[^\s@]+@\S+'
    # Known env var names — flag presence even without a value, so a transcript
    # describing key handling does not slip through.
    r'|\b(?:'
    r'ANTHROPIC_API_KEY|OPENAI_API_KEY|OPENROUTER_API_KEY|MINIMAX_API_KEY'
    r'|SLACK_BOT_TOKEN|GITHUB_TOKEN|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID'
    r'|DATABASE_URL|REDIS_URL|MONGO_URI|HF_TOKEN|HUGGINGFACE_TOKEN'
    r'|STRIPE_SECRET_KEY|TWILIO_AUTH_TOKEN|SENDGRID_API_KEY'
    r')\b'
    # Generic key/secret/password assignments — last so prefixed patterns win.
    r'|\b(?:password|passwd|pwd)\s*[=:]\s*\S+'
    r'|\b(?:api[_-]?key|secret|token|credential)\s*[=:]\s*\S{10,}'
    r')',
    re.IGNORECASE,
)

# PII patterns — applied as a second-pass filter alongside secret detection.
# Catches personal data that SECRET_PATTERNS intentionally does not cover.
PII_PATTERNS = re.compile(
    r'('
    # Email addresses
    r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'
    # IPv4 addresses (not 127.0.0.1 or 0.0.0.0 which are benign)
    r'|(?<!\d)(?!127\.0\.0\.1|0\.0\.0\.0)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)'
    # Phone numbers (international formats)
    r'|(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'
    # SSN-like patterns (US)
    r'|\b\d{3}-\d{2}-\d{4}\b'
    r')',
)


VALID_DIFFICULTIES = {"easy", "medium", "hard"}

MIN_DATASET_SIZE = 3  # Minimum examples needed to produce a meaningful split


def _contains_secret(text: str) -> bool:
    """Check if text contains potential API keys, tokens, or PII."""
    return bool(SECRET_PATTERNS.search(text)) or bool(PII_PATTERNS.search(text))


def scrub_secrets(text: str, replacement: str = "[REDACTED]") -> str:
    """Replace any matched secret or PII patterns with a placeholder.

    Defence-in-depth scan on artifacts about to be persisted to disk (evolved
    skill bodies, error messages, log lines). Not a substitute for the
    `_contains_secret` ingest filter — this is the last-resort layer for
    content the model may have paraphrased into plausible secret-shaped text.
    """
    text = SECRET_PATTERNS.sub(replacement, text)
    text = PII_PATTERNS.sub(replacement, text)
    return text


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
    """Import sessions from Claude Code.

    Claude Code stores data in two locations:

    1. ``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` — full session
       transcripts. Each line is one event (``user``, ``assistant``,
       ``attachment``, ``permission-mode``, etc.). When present these yield
       paired ``(task_input, assistant_response)`` examples comparable to the
       Copilot/Hermes importers.

    2. ``~/.claude/history.jsonl`` — flat log of user prompts only. Used as a
       fallback when ``projects/`` is empty or missing (older Claude Code
       installations, or fresh machines).

    The default behaviour is ``source="auto"``: prefer rich project transcripts
    when available, fall back to ``history.jsonl`` otherwise. Pass
    ``source="history"`` to force the legacy user-only path, or
    ``source="projects"`` to read transcripts only.
    """

    HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"
    PROJECTS_DIR = Path.home() / ".claude" / "projects"

    @staticmethod
    def extract_messages(limit: int = 0, source: str = "auto", project_filter: Optional[str] = None) -> list[dict]:
        """Read messages from Claude Code session storage.

        Args:
            limit: Maximum messages to return (0 = no limit).
            source: "auto" (default), "projects", or "history".
            project_filter: If set, only mine sessions from project dirs
                whose name contains this substring (encoded CWD matching).

        Returns:
            List of dicts. Always include ``source``, ``task_input``,
            ``project``, ``session_id``, ``timestamp``. Project transcripts
            additionally include ``assistant_response``.
        """
        if source not in ("auto", "projects", "history"):
            raise ValueError(
                f"source must be 'auto', 'projects', or 'history' (got {source!r})"
            )

        if source in ("auto", "projects"):
            messages = ClaudeCodeImporter._extract_from_projects(limit, project_filter=project_filter)
            if messages or source == "projects":
                return messages

        return ClaudeCodeImporter._extract_from_history(limit)

    @staticmethod
    def _extract_from_history(limit: int = 0) -> list[dict]:
        """Read user prompts from the flat ``history.jsonl`` log."""
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

    @staticmethod
    def _extract_from_projects(limit: int = 0, project_filter: Optional[str] = None) -> list[dict]:
        """Read paired user/assistant turns from project session transcripts.

        Args:
            limit: Maximum messages to return (0 = no limit).
            project_filter: If set, only mine sessions from project dirs
                whose encoded name contains this substring.
        """
        if not ClaudeCodeImporter.PROJECTS_DIR.exists():
            return []

        session_files = sorted(ClaudeCodeImporter.PROJECTS_DIR.rglob("*.jsonl"))
        if not session_files:
            return []

        messages: list[dict] = []
        for session_path in session_files:
            project = session_path.parent.name
            if project_filter and project_filter not in project:
                continue
            messages.extend(_parse_claude_code_session(session_path, project))
            if limit and len(messages) >= limit:
                messages = messages[:limit]
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


def _parse_claude_code_session(session_path: Path, project: str) -> list[dict]:
    """Parse one Claude Code session JSONL into (user, assistant) pairs.

    Claude Code project transcripts interleave many record types. We keep only
    real user prompts (``type == "user"`` with string content — array content
    means a tool result, which we skip) and concatenate the text blocks of all
    assistant turns that follow until the next user prompt.

    Records lacking ``type``, malformed JSON, or events containing detected
    secrets are skipped. A session that yields no clean pairs returns an empty
    list rather than raising.
    """
    pairs: list[dict] = []
    current_user: Optional[str] = None
    current_assistant_parts: list[str] = []

    session_id = session_path.stem

    def flush() -> None:
        if current_user and current_assistant_parts:
            assistant = "\n".join(current_assistant_parts).strip()
            if assistant and not _contains_secret(current_user) and not _contains_secret(assistant):
                pairs.append({
                    "source": "claude-code",
                    "task_input": current_user,
                    "assistant_response": assistant,
                    "project": project,
                    "session_id": session_id,
                    "timestamp": 0,
                })

    try:
        with open(session_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                message = event.get("message") or {}

                if event_type == "user":
                    content = message.get("content")
                    # array content == tool_result, skip it
                    if not isinstance(content, str):
                        continue
                    text = content.strip()
                    if len(text) < 10:
                        continue
                    # close out the previous turn before starting a new one
                    flush()
                    current_user = text
                    current_assistant_parts = []

                elif event_type == "assistant" and current_user is not None:
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "").strip()
                            if text:
                                current_assistant_parts.append(text)

        flush()
    except OSError as e:
        console.print(f"[dim]Skipped {session_path.name}: {e}[/dim]")

    return pairs


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

    def __init__(self, model: str):
        self.scorer = dspy.ChainOfThought(self.ScoreRelevance)
        self.model = model

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
        lm = dspy.LM(self.model)

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
    source_project: Optional[str] = None,
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
        source_project: If set, limit Claude Code mining to sessions whose
            project directory name contains this substring.

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
        if source == "claude-code" and source_project:
            console.print(f"  Filtering to projects matching: {source_project!r}")
            msgs = importer_cls.extract_messages(project_filter=source_project)
        else:
            msgs = importer_cls.extract_messages()
        console.print(f"  Found {len(msgs)} messages")
        all_messages.extend(msgs)

    if not all_messages:
        console.print("[red]No messages found from any source.[/red]")
        return EvalDataset()

    console.print(f"\n[bold]Total messages: {len(all_messages)}[/bold]")
    console.print(f"[bold]Filtering for relevance to skill: {skill_name}[/bold]")

    relevance_filter = RelevanceFilter(model=model)
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


_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


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
        ValueError: If skill_name contains path separators or shell metacharacters.
    """
    if not skill_name or not _SKILL_NAME_RE.match(skill_name):
        raise ValueError(
            f"Invalid skill name {skill_name!r} — must match [A-Za-z0-9_.-]+ "
            "(no path separators or shell metacharacters)"
        )

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
@click.option(
    "--consent-external-ingest",
    is_flag=True,
    help=(
        "Required. Acknowledges that local chat transcripts from ~/.claude, "
        "~/.copilot, and ~/.hermes will be read and sent to the configured "
        "LLM for relevance scoring."
    ),
)
@click.option(
    "--source-project",
    default=None,
    help="Limit Claude Code mining to sessions from this project directory name only",
)
def main(source, skill, output, model, max_examples, dry_run, consent_external_ingest, source_project):
    """Import external session data into golden eval datasets for self-evolution."""
    console.print(f"\n[bold cyan]External Session Importer[/bold cyan] — skill: [bold]{skill}[/bold]\n")

    if not dry_run and not consent_external_ingest:
        console.print(
            "[red]This command reads chat transcripts from ~/.claude, ~/.copilot, "
            "and ~/.hermes.[/red]\n"
            "[red]Transcripts may contain:[/red]\n"
            "[red]  - Full source code and terminal output from all projects[/red]\n"
            "[red]  - Personal data, credentials, and business-confidential content[/red]\n"
            "[red]  - Assistant responses quoting sensitive file contents[/red]\n"
            "[red]Relevant content (up to 1000 chars per message) is sent to the[/red]\n"
            f"[red]configured LLM ({model}) for relevance scoring.[/red]\n\n"
            "[red]Re-run with --consent-external-ingest to proceed.[/red]"
        )
        raise SystemExit(2)

    try:
        skill_name, skill_text = _load_skill_text(skill)
    except (FileNotFoundError, ValueError) as e:
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
        source_project=source_project,
    )


if __name__ == "__main__":
    main()
