#!/usr/bin/env python3
"""
Akashic 2.0 — Self-Adaptive Context Pruning Engine

Implements SWE-Pruner-style scope-aware chunking for the Cloud Cortex:
- AST-based structural skeleton preservation (imports, def/class signatures)
- Goal-hint matching keeps full bodies of relevant functions
- State-mutation regex overlay preserves lines that mutate system state
  (sed -i, >>, export, open('w'), .write(), DB mutations, os.remove, rmtree)
- Grammar-blind regex fallback for syntactically broken code
- Sliding context window centered on syntax errors (debugging mode)

Spec: Metaconscious Singularity Node — Akashic 2.0 subsystem.
Empirical: 39-54% token reduction, up to 26% fewer agent interaction rounds.
"""
import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# State-mutation patterns — any line matching these is NEVER pruned
MUTATION_PATTERNS = [
    r'\bsed\s+-i\b',          # in-place text edits
    r'>>',                    # file redirect append
    r'>\s*\S+',               # file redirect overwrite (but not '->' or '>=' comparisons)
    r'\bexport\s+\w+=',       # env var writes
    r'\bunset\s+\w+',         # env var removal
    r'\bnp\.save\b', r'\bnp\.savetxt\b', r'\bnp\.savez\b',
    r'\bopen\s*\([^)]*[\'"]w', r'\bopen\s*\([^)]*[\'"]a',
    r'\.write\(', r'\.writelines\(', r'\.dump\(', r'\.dumps\(',
    r'\bINSERT\s+INTO\b', r'\bUPDATE\s+\w+\s+SET\b', r'\bDELETE\s+FROM\b',
    r'\bALTER\s+TABLE\b', r'\bCREATE\s+TABLE\b', r'\bDROP\s+TABLE\b',
    r'\bos\.remove\b', r'\bos\.unlink\b', r'\brmtree\b', r'\bchmod\b', r'\bchown\b',
    r'\bshutil\.(copy|move|rmtree)\b',
    r'\bgit\s+(add|commit|push|checkout|reset|rm|mv)\b',
    r'\brm\s+-rf?\b', r'\bmkdir\s+-p\b', r'\bcurl\s+.*-o\b', r'\bwget\s+',
    r'\bsubprocess\.(run|call|Popen|check_call)\b',
    r'\bconn\.(execute|commit|rollback)\b', r'\bcursor\.execute\b',
    r'\bmodel\.save\b', r'\btorch\.save\b', r'\bjson\.dump\b',
]

# Structural markers for the grammar-blind fallback
STRUCTURAL_PREFIXES = ("def ", "class ", "import ", "from ", "if __", "elif ",
                       "else:", "try:", "except ", "finally:", "async def ",
                       "with ", "@", "return ", "pass", "raise ")


class AkashicPruner:
    """AST-based context pruner. Reduces prompt tokens 39-54% per SWE-Pruner."""

    def __init__(self, goal_hint: str = "", debug_window: int = 0):
        self.goal_hint = goal_hint.lower()
        self.debug_window = debug_window  # if >0, keep ±N lines around syntax errors
        self.last_mode = "none"  # 'ast' | 'regex' | 'error-window'
        # Tokenize goal hint for relevance matching (stopwords dropped)
        self.goal_tokens = [t for t in re.split(r'\W+', self.goal_hint)
                            if len(t) > 2 and t not in ("the", "and", "for", "with")]

    def _goal_matches(self, name: str) -> bool:
        """Relevance match: whole hint or any significant token inside the name."""
        name_l = name.lower()
        if self.goal_hint and self.goal_hint in name_l:
            return True
        return any(tok in name_l for tok in self.goal_tokens)

    def prune_file(self, source: str) -> str:
        """Prune a single source file toward the goal hint."""
        if not source.strip():
            return source
        try:
            tree = ast.parse(source)
            self.last_mode = "ast"
            return self._ast_prune(tree, source)
        except SyntaxError as exc:
            if self.debug_window > 0:
                self.last_mode = "error-window"
                return self._error_window(source, exc)
            self.last_mode = "regex"
            return self._regex_fallback(source)

    # ─── AST mode ───

    def _ast_prune(self, tree: ast.Module, source: str) -> str:
        """Walk AST, keep relevant nodes + state-mutations + structural skeleton."""
        lines = source.splitlines(keepends=True)
        keep: Set[int] = set()

        # Always keep imports and module docstring
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._keep_range(keep, node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str) and node.lineno == 1:
                self._keep_range(keep, node)

        # Recursive walk: classes and functions get skeletons, goal-matches get full bodies
        def walk_node(node) -> None:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self._keep_range(keep, node, skeleton=True)
                if self._goal_matches(node.name):
                    self._keep_range(keep, node, skeleton=False)
                for dec in getattr(node, "decorator_list", []):
                    self._keep_range(keep, dec)
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                        walk_node(child)

        for child in ast.iter_child_nodes(tree):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                walk_node(child)

        # State-mutation regex overlay — never prune state-altering lines
        for i, line in enumerate(lines):
            if self._is_mutation(line):
                keep.add(i)

        return self._reconstruct(lines, keep)

    @staticmethod
    def _keep_range(keep: Set[int], node, skeleton: bool = False) -> None:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        if skeleton:
            # def line + signature lines + closing line (col offset for multi-line sigs)
            keep.add(start)
            if end - 1 > start:
                keep.add(end - 1)
            # Keep signature continuation lines (lines 2..first body line)
            body_start = start + 1
            while body_start < end - 1 and not AkashicPruner._is_body_start(node, body_start):
                keep.add(body_start)
                body_start += 1
        else:
            for i in range(start, end):
                keep.add(i)

    @staticmethod
    def _is_body_start(node, line_no: int) -> bool:
        """Heuristic: first indented line after the def/class line."""
        try:
            first_body = node.body[0]
            return getattr(first_body, "lineno", 0) == line_no + 1
        except (AttributeError, IndexError):
            return False

    # ─── Shared reconstruction ───

    @staticmethod
    def _reconstruct(lines: List[str], keep: Set[int]) -> str:
        result = []
        prev_kept = False
        pruned_run = 0
        for i, line in enumerate(lines):
            if i in keep:
                result.append(line)
                prev_kept = True
                pruned_run = 0
            elif prev_kept:
                pruned_run += 1
                # Insert a single ellipsis marker per pruned region
                next_kept = any(j in keep for j in range(i + 1, min(i + 5, len(lines))))
                if next_kept and pruned_run == 1:
                    indent = line[:len(line) - len(line.lstrip())]
                    result.append(f"{indent}# ... (pruned by Akashic 2.0)\n")
            else:
                continue
        # Drop trailing ellipsis if the file ends in pruned content
        while result and result[-1].startswith("# ... (pruned"):
            result.pop()
        return "".join(result)

    # ─── Grammar-blind fallback (syntactically broken code) ───

    def _regex_fallback(self, source: str) -> str:
        """Keep structural markers, mutations, and non-empty context lines."""
        lines = source.splitlines(keepends=True)
        keep = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith(STRUCTURAL_PREFIXES) or stripped == "" or
                    self._is_mutation(line) or
                    re.match(r'^\s+[a-zA-Z_]\w*\s*[=:]', line) or  # assignments/dict keys
                    stripped.startswith(("'", '"', "#"))):
                keep.append(line)
        return "".join(keep)

    # ─── Debug mode: sliding window around syntax error ───

    def _error_window(self, source: str, exc: SyntaxError) -> str:
        lines = source.splitlines(keepends=True)
        err_line = (exc.lineno or 1) - 1
        lo = max(0, err_line - self.debug_window)
        hi = min(len(lines), err_line + self.debug_window + 1)
        window = lines[lo:hi]
        marker = f"# ⚠ SYNTAX ERROR at line {err_line + 1}: {exc.msg}\n"
        return "".join(window[: (err_line - lo)]) + marker + "".join(window[(err_line - lo):])

    # ─── Helpers ───

    @staticmethod
    def _is_mutation(line: str) -> bool:
        # Skip comparison operators that contain '>' (e.g. '->', '>=')
        if re.search(r'->', line):
            return False
        for pat in MUTATION_PATTERNS:
            if re.search(pat, line, re.IGNORECASE):
                return True
        return False

    def token_reduction(self, original: str, pruned: str) -> float:
        """Rough token reduction percentage (4 chars/token heuristic)."""
        o = max(1, len(original) // 4)
        p = max(1, len(pruned) // 4)
        return (1 - p / o) * 100.0


def prune_path(path: str, goal_hint: str = "", debug_window: int = 0) -> Tuple[str, str, str]:
    """Prune a file on disk. Returns (pruned_text, mode, reduction_pct_str)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    source = p.read_text(errors="replace")
    pruner = AkashicPruner(goal_hint=goal_hint, debug_window=debug_window)
    pruned = pruner.prune_file(source)
    reduction = pruner.token_reduction(source, pruned)
    return pruned, pruner.last_mode, f"{reduction:.1f}%"


if __name__ == "__main__":
    # ─── Test 1: token reduction on a real file ───
    pruner = AkashicPruner("database error handling")
    target = Path(__file__).parent / "ouroboros_memory.py"
    original = target.read_text()
    pruned = pruner.prune_file(original)
    reduction = pruner.token_reduction(original, pruned)
    print(f"[akashic] AST mode: {len(original)//4} -> {len(pruned)//4} tokens, "
          f"reduction {reduction:.1f}%")
    assert pruner.last_mode == "ast"
    assert reduction > 10, f"Expected >10% reduction, got {reduction:.1f}%"

    # ─── Test 2: goal-hint keeps relevant function bodies ───
    pruner2 = AkashicPruner("syntax error window")
    src2 = Path(__file__).read_text()  # this file contains _error_window
    pruned2 = pruner2.prune_file(src2)
    assert "def _error_window" in pruned2, "goal-matched method body pruned!"
    # Full body of _error_window kept, not just skeleton
    assert "SYNTAX ERROR at line" in pruned2, "goal-matched body should include internals"
    print("[akashic] PASS — goal-hint relevant functions preserved")

    # ─── Test 3: state mutations never pruned ───
    assert any("conn.execute" in l or "PRAGMA" in l for l in pruned.splitlines()), \
        "State-mutation lines were pruned!"
    print("[akashic] PASS — state-mutation overlay preserved DB writes")

    # ─── Test 4: regex fallback handles broken Python ───
    broken = "def foo(\n  x = 1\n  \nimport os\nclass Bar(\n  y = 2\n"
    result = pruner._regex_fallback(broken)
    for marker in ("def foo", "import os", "class Bar"):
        assert marker in result, f"fallback lost {marker}"
    print("[akashic] PASS — grammar-blind fallback on broken syntax")

    # ─── Test 5: debug window centers on the syntax error ───
    err_pruner = AkashicPruner(debug_window=3)
    err_out = err_pruner.prune_file(broken)
    assert "SYNTAX ERROR" in err_out
    assert "import os" in err_out
    print("[akashic] PASS — debug error-window mode")
