"""Content retention analysis for evolved artifacts.

The ``growth_limit`` constraint only bounds how much an evolved artifact may
*grow* over its baseline. It cannot detect the opposite failure mode: an
optimizer that **deletes** expensive-to-learn content (incident lessons,
runnable commands, structural sections) and is then rewarded for it by an
LLM-as-judge that only sees whether the answer "looks right" for a task.

This module provides a *deletion* detector: it compares an evolved artifact
against its baseline and reports how much semantically-load-bearing content
survived. It is deliberately deterministic and dependency-free so it can run
as a hard constraint, not as a heuristic score.

Two families of content are tracked:

``commands``
    Runnable shell commands found in fenced code blocks. These are the part
    of a skill an agent can copy-paste and execute; losing them turns an
    executable playbook into prose.

``lessons``
    Lines carrying an imperative guard marker (a prohibition, a warning, a
    pitfall, or a mandatory step). These encode the institutional memory
    that is usually the most expensive part of a skill to rediscover.

Both are matched on *semantic groups* rather than exact strings, so a
reworded-but-equivalent lesson still counts as retained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

# ── command extraction ──────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)

# A shell-ish line: starts with a word, may contain flags, paths, pipes.
_CMD_RE = re.compile(
    r"^[ \t]*(?:"
    r"\$ ?"
    r"|(?:sudo|bash|sh|zsh|env|cd|ls|cat|grep|rg|find|echo|printf|tee|"
    r"mkdir|rm|cp|mv|ln|chmod|chown|touch|head|tail|sort|uniq|wc|diff|"
    r"sed|awk|cut|tr|jq|curl|wget|git|gh|docker|docker-compose|podman|"
    r"kubectl|helm|systemctl|journalctl|ss|lsof|ps|top|kill|nohup|ssh|"
    r"scp|rsync|tar|unzip|gzip|python3?|pip3?|node|npm|npx|yarn|pnpm|"
    r"pytest|make|cmake|go|cargo|rustc|java|mvn|gradle|brew|apt|apt-get|"
    r"dpkg|yum|dnf|apk|openssl|gpg|base64|xxd|jq|export|source|alias|"
    r"tmux|screen|rsync|watch|timeout|nohup|xargs|tee|diff|patch"
    r")\b"
    r")",
    re.IGNORECASE,
)


def extract_commands(text: str) -> Set[str]:
    """Return the set of normalised shell commands found in ``text``.

    Only fenced code blocks are considered. Lines that look like Markdown
    prose or bold-label markup (``**[apt summary]**``) are excluded, and
    each command is normalised so that trivial argument changes do not read
    as a different command.
    """
    commands: Set[str] = set()
    if not text:
        return commands
    for block in _FENCE_RE.findall(text):
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Skip markdown emphasis labels that are not commands.
            if line.startswith("*") or line.startswith("**"):
                continue
            if not _CMD_RE.match(line):
                continue
            commands.add(_normalise_command(line))
    return commands


def _normalise_command(line: str) -> str:
    """Collapse a command to a stable identity.

    The executable name and its subcommand are kept; flags, positional
    arguments and paths are dropped, because a reworded invocation of the
    same tool is still the same capability.
    """
    line = line.lstrip("$ ").strip()
    # Drop leading env assignments: FOO=bar cmd -> cmd
    while re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line):
        line = line.split("=", 1)[1].lstrip()
    parts = line.split()
    if not parts:
        return line
    exe = parts[0].rsplit("/", 1)[-1]
    sub = ""
    for token in parts[1:]:
        if token.startswith("-"):
            continue
        if "/" in token and not token.startswith("-"):
            # a path argument, not a subcommand
            continue
        sub = token
        break
    return f"{exe} {sub}".strip()


# ── lesson extraction ───────────────────────────────────────────────────────

# Semantic groups of guard markers. A lesson is a line that contains at least
# one marker from any group. Grouping by *meaning* rather than by exact
# wording means "do not" / "never" / "must not" are treated as equivalent.
_LESSON_GROUPS: Dict[str, Tuple[str, ...]] = {
    "prohibition": (
        "do not", "dont", "never", "must not", "mustn't", "shall not",
        "should not", "shouldn't", "avoid", "forbidden", "prohibited",
        "not allowed", "cannot", "can't", "no longer",
    ),
    "warning": (
        "warning", "caution", "careful", "beware", "gotcha", "pitfall",
        "trap", "hazard", "risk", "danger",
    ),
    "requirement": (
        "must ", "required", "require ", "always ", "make sure",
        "ensure ", "remember to", "important:", "critical",
    ),
    "instruction": (
        "do not skip", "never skip", "first run", "before you", "instead of",
        "prefer ", "use ", "run ",
    ),
}

# Chinese equivalents — the same semantic groups, for zh skills.
_LESSON_GROUPS_ZH: Dict[str, Tuple[str, ...]] = {
    "prohibition": ("不要", "不得", "禁止", "严禁", "别", "避免", "不能", "不可", "切勿"),
    "warning": ("警告", "注意", "小心", "陷阱", "坑", "风险", "危险", "后果"),
    "requirement": ("必须", "务必", "一定要", "需要", "应当", "应该", "要求"),
    "instruction": ("先", "改用", "优先", "使用", "执行", "运行"),
}


@dataclass
class LessonGroup:
    """One semantic group of guard markers and the lines that carry it."""

    name: str
    markers: Tuple[str, ...]

    def find(self, text: str) -> List[str]:
        found: List[str] = []
        for line in text.splitlines():
            low = line.lower()
            if any(m in low for m in self.markers):
                stripped = line.strip()
                if stripped:
                    found.append(stripped)
        return found


def lesson_groups() -> List[LessonGroup]:
    """All lesson groups, English and Chinese."""
    groups = [LessonGroup(name, markers) for name, markers in _LESSON_GROUPS.items()]
    groups += [LessonGroup(name, markers) for name, markers in _LESSON_GROUPS_ZH.items()]
    return groups


# ── retention report ────────────────────────────────────────────────────────

@dataclass
class RetentionReport:
    """How much load-bearing content survived an evolution pass."""

    command_retention: float = 1.0
    lesson_retention: float = 1.0
    block_retention: float = 1.0
    commands_total: int = 0
    commands_kept: int = 0
    commands_lost: List[str] = field(default_factory=list)
    lessons_total: int = 0
    lessons_kept: int = 0
    lessons_lost: Dict[str, int] = field(default_factory=dict)
    blocks_total: int = 0
    blocks_kept: int = 0
    size_ratio: float = 1.0

    @property
    def worst_retention(self) -> float:
        """The lowest structural retention rate.

        Only *structural* signals count here (commands, blocks). Lesson
        retention is reported for diagnosis but deliberately excluded from
        the verdict: rewriting a guard line can change which marker word it
        uses ("must" -> "have to"), which makes lesson counting a poor
        pass/fail signal even though it is a useful diagnostic.
        """
        rates = []
        if self.commands_total:
            rates.append(self.command_retention)
        if self.blocks_total:
            rates.append(self.block_retention)
        return min(rates) if rates else 1.0


def extract_blocks(text: str) -> List[str]:
    """Split a document into content blocks separated by blank lines.

    Headings become their own block and each paragraph becomes a block.
    This is what makes a *removed callout* visible: when an optimizer drops
    a standalone paragraph, the block count drops with it.
    """
    blocks: List[str] = []
    current: List[str] = []
    for line in (text or "").splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        if line.lstrip().startswith("#"):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            blocks.append(line.strip())
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b.strip()]


def _blocks_match(base_block: str, evo_blocks: Sequence[str]) -> bool:
    """Is this baseline block still represented in the evolved text?

    Heading blocks match on their heading text. Paragraph blocks match when
    at least half of their distinctive tokens survive, so rephrasing a
    paragraph does not read as a deletion while gutting it does.
    """
    base = base_block.strip()
    if base.lstrip().startswith("#"):
        heading = base.lstrip("#").strip().lower()
        for other in evo_blocks:
            if other.lstrip().startswith("#") and other.lstrip("#").strip().lower() == heading:
                return True
        return False

    base_tokens = _distinctive_tokens(base)
    if not base_tokens:
        return False
    needed = max(1, len(base_tokens) // 2)
    for other in evo_blocks:
        if other.lstrip().startswith("#"):
            continue
        if len(base_tokens & _distinctive_tokens(other)) >= needed:
            return True
    return False


def analyse_retention(baseline: str, evolved: str) -> RetentionReport:
    """Compare an evolved artifact against its baseline.

    ``command_retention`` is the fraction of baseline commands still present.
    ``lesson_retention`` is the fraction of baseline guard-marker lines still
    present, compared per semantic group and averaged over groups that were
    present in the baseline.
    """
    base_cmds = extract_commands(baseline)
    evo_cmds = extract_commands(evolved)
    kept_cmds = base_cmds & evo_cmds
    lost_cmds = sorted(base_cmds - evo_cmds)

    report = RetentionReport(
        command_retention=(len(kept_cmds) / len(base_cmds)) if base_cmds else 1.0,
        commands_total=len(base_cmds),
        commands_kept=len(kept_cmds),
        commands_lost=lost_cmds,
    )

    # Per-group lesson retention.
    #
    # Retention is measured per *semantic group* by counting the guard lines
    # the evolved text still carries in that group, not by matching
    # individual sentences. Sentence-level matching produces false
    # positives: rewording "always check X" as "always inspect X" is a
    # legitimate rewrite, yet a token comparison reports the lesson as
    # deleted. Counting per group keeps the signal ("this kind of guard
    # disappeared") without punishing vocabulary changes.
    group_rates: List[float] = []
    for group in lesson_groups():
        base_lines = group.find(baseline)
        if not base_lines:
            continue
        evo_lines = group.find(evolved)
        kept = min(len(base_lines), len(evo_lines))
        rate = kept / len(base_lines)
        group_rates.append(rate)
        report.lessons_total += len(base_lines)
        report.lessons_kept += kept
        if kept < len(base_lines):
            report.lessons_lost[group.name] = len(base_lines) - kept

    report.lesson_retention = (
        sum(group_rates) / len(group_rates) if group_rates else 1.0
    )

    # Structural block retention: how many baseline sections survived.
    base_blocks = extract_blocks(baseline)
    if base_blocks:
        evo_blocks = extract_blocks(evolved)
        kept = sum(1 for b in base_blocks if _blocks_match(b, evo_blocks))
        report.blocks_total = len(base_blocks)
        report.blocks_kept = kept
        report.block_retention = kept / len(base_blocks)

    report.size_ratio = len(evolved) / max(1, len(baseline))
    return report


def _lesson_present(base_line: str, evolved_lines: Sequence[str]) -> bool:
    """Is this baseline lesson still present in the evolved text?

    Exact match wins. Otherwise the lesson counts as retained when its
    distinctive tokens survive, so a reworded-but-equivalent lesson is not
    reported as a deletion.
    """
    if base_line in evolved_lines:
        return True
    tokens = _distinctive_tokens(base_line)
    if not tokens:
        return False
    for line in evolved_lines:
        if tokens <= _distinctive_tokens(line):
            return True
    return False


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}|[一-鿿]{2,4}")


def _distinctive_tokens(line: str) -> Set[str]:
    """Tokens that carry the meaning of a guard line.

    Common English stop-words and generic Chinese function words are dropped
    so that a reworded sentence still overlaps on its distinctive terms.
    """
    tokens = {t.lower() for t in _TOKEN_RE.findall(line)}
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= 3}


_STOPWORDS = {
    "the", "and", "for", "you", "your", "not", "but", "with", "this", "that",
    "from", "are", "was", "has", "have", "had", "will", "can", "should",
    "must", "always", "never", "when", "what", "which", "then", "than",
    "all", "any", "each", "its", "use", "using", "used", "into", "out",
    "不要", "不得", "必须", "应该", "以及", "这个", "那个", "如果", "然后",
}


def format_report(report: RetentionReport, max_examples: int = 8) -> str:
    """Human-readable one-paragraph summary for a constraint message."""
    parts: List[str] = []
    if report.commands_total:
        parts.append(
            f"commands {report.commands_kept}/{report.commands_total} retained "
            f"({report.command_retention:.0%})"
        )
        if report.commands_lost:
            shown = ", ".join(report.commands_lost[:max_examples])
            more = len(report.commands_lost) - max_examples
            suffix = f" (+{more} more)" if more > 0 else ""
            parts.append(f"lost: {shown}{suffix}")
    if report.lessons_total:
        parts.append(
            f"guard lines {report.lessons_kept}/{report.lessons_total} retained "
            f"({report.lesson_retention:.0%})"
        )
        if report.lessons_lost:
            parts.append(
                "lost per group: "
                + ", ".join(f"{k}×{v}" for k, v in sorted(report.lessons_lost.items()))
            )
    parts.append(f"size {report.size_ratio:.0%} of baseline")
    return "; ".join(parts)
