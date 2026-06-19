"""Local-first code evolver — no cloud API required.

A lightweight evolutionary optimizer for Python skill code that runs entirely
on a local LLM endpoint (OpenAI-compatible, e.g. llama.cpp, Ollama).
Extracted and generalised from JoeCLI (https://github.com/applelai001/joe).

Three-signal fitness function (ADAS-inspired):
  Signal 1 — AST structural complexity   (weight 0.25, instant, no LLM)
  Signal 2 — Pattern keyword coverage    (weight 0.25, instant, no LLM)
  Signal 3 — LLM-as-judge               (weight 0.50, ~2-5s via local LLM)

A pure LLM wrapper scores ~0.10–0.20.
A real implementation scores ~0.60–0.90.

Usage (standalone):
    from evolution.code.local_evolver import LocalCodeEvolver
    evolver = LocalCodeEvolver(llm_base_url="http://127.0.0.1:11434/v1")
    best_code, score = evolver.evolve(
        skill_name="react_agent",
        topic="ReAct agent pattern",
        initial_code=open("skills/react_agent.py").read(),
        iterations=3,
    )

Usage (as a fitness backend for evolve_skill.py):
    from evolution.code.local_evolver import LocalCodeEvolver, LocalFitnessScore
    evolver = LocalCodeEvolver()
    result = evolver.score(code, tasks=["perform a reasoning task"])
    print(result.composite)  # 0.0–1.0
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import urllib.request
import json

# ---------------------------------------------------------------------------
# Pattern keyword vocabulary (extend as needed)
# ---------------------------------------------------------------------------

PATTERN_KEYWORDS: dict[str, list[str]] = {
    "react":       ["thought", "action", "observation", "loop", "step"],
    "reflexion":   ["reflect", "retry", "attempt", "evaluate", "trial"],
    "memory":      ["sqlite", "store", "recall", "embed", "retrieve"],
    "self_heal":   ["diagnose", "fix", "verify", "recover", "rollback"],
    "planning":    ["plan", "decompose", "step", "subtask", "execute"],
    "tool":        ["select", "dispatch", "execute", "observe", "result"],
    "coding":      ["parse", "compile", "test", "patch", "diff"],
    "search":      ["query", "rank", "fetch", "index", "result"],
    "summarize":   ["chunk", "extract", "compress", "key", "brief"],
}


# ---------------------------------------------------------------------------
# Fitness score dataclass (mirrors hermes FitnessScore interface)
# ---------------------------------------------------------------------------

@dataclass
class LocalFitnessScore:
    """Three-signal fitness breakdown."""

    ast_complexity: float = 0.0       # Signal 1: structural richness (0-1)
    keyword_coverage: float = 0.0     # Signal 2: pattern keyword match (0-1)
    llm_judge: float = 0.5            # Signal 3: LLM-as-judge (0-1)
    exec_penalty: float = 0.0         # 0 = ran fine, 0.2 = external crash, 1.0 = hard fail
    feedback: str = ""                # LLM judge's one-sentence reason

    @property
    def composite(self) -> float:
        raw = 0.25 * self.ast_complexity + 0.25 * self.keyword_coverage + 0.50 * self.llm_judge
        return round(max(0.0, raw * (1.0 - self.exec_penalty)), 3)


# ---------------------------------------------------------------------------
# Main evolver class
# ---------------------------------------------------------------------------

class LocalCodeEvolver:
    """Lightweight local-first evolutionary code optimizer.

    Parameters
    ----------
    llm_base_url:
        OpenAI-compatible endpoint base URL. Defaults to Ollama on localhost.
    model:
        Model name to use for LLM-as-judge and variant generation.
    db_path:
        SQLite path for variant history. None = in-memory only.
    """

    def __init__(
        self,
        llm_base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen2.5-coder:14b",
        db_path: Optional[Path] = None,
    ):
        self.llm_base_url = llm_base_url.rstrip("/")
        self.model = model
        self.db_path = db_path
        if db_path:
            self._db_init()

    # ── DB ────────────────────────────────────────────────────────────────

    def _db_init(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS variants (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                skill     TEXT,
                code      TEXT,
                score     REAL DEFAULT 0,
                iteration INTEGER DEFAULT 0,
                created   REAL DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def _db_save(self, skill: str, code: str, score: float, iteration: int):
        if not self.db_path:
            return
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.execute(
            "INSERT INTO variants (skill,code,score,iteration,created) VALUES (?,?,?,?,?)",
            (skill, code[:6000], score, iteration, time.time()),
        )
        conn.commit()
        conn.close()

    # ── LLM call ──────────────────────────────────────────────────────────

    def _llm(self, messages: list[dict], timeout: int = 30) -> str:
        """Single OpenAI-compatible chat completion (non-streaming)."""
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(
            f"{self.llm_base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # ── Signal 1: AST complexity ──────────────────────────────────────────

    @staticmethod
    def ast_score(code: str) -> float:
        """Structural complexity via AST — instant, no LLM needed.

        Heuristic: a pure LLM wrapper has 1 function, 0 loops, <20 lines.
        A real implementation has 2+ functions, loops, try/except, 30+ lines.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0
        loops     = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
        branches  = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
        try_nodes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))
        funcs     = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        lines     = len([l for l in code.splitlines() if l.strip()])

        structure = loops + branches + try_nodes + max(0, funcs - 1)
        line_score = min(1.0, lines / 60)
        return min(1.0, (structure / 6) * 0.7 + line_score * 0.3)

    # ── Signal 2: keyword coverage ────────────────────────────────────────

    @staticmethod
    def keyword_score(code: str, pattern: str) -> float:
        """Pattern keyword overlap — instant, no LLM needed."""
        lower = code.lower()
        kws = PATTERN_KEYWORDS.get(pattern.lower().split()[0] if pattern else "", [])
        if not kws:
            # Generic: penalise obvious LLM-wrapper patterns
            is_wrapper = (
                ("requests.post" in lower or "urlopen" in lower)
                and "for " not in lower
                and "while " not in lower
                and lower.count("def ") <= 1
            )
            return 0.1 if is_wrapper else 0.5
        found = sum(1 for kw in kws if kw in lower)
        return found / len(kws)

    # ── Signal 3: LLM-as-judge ────────────────────────────────────────────

    def llm_judge(self, code: str, pattern_desc: str) -> tuple[float, str]:
        """LLM-as-judge scorer. Returns (score 0-1, one-sentence feedback)."""
        prompt = (
            f'Evaluate if this Python code truly implements "{pattern_desc}" as a real algorithm.\n\n'
            "SCORING GUIDE:\n"
            "  0-2: Pure LLM wrapper — just calls an HTTP endpoint and returns the response\n"
            "  3-5: Has pattern elements but core logic is placeholder/incomplete\n"
            "  6-8: Implements the pattern with real logic (loops, state, conditions)\n"
            "  9-10: Complete, robust implementation with error handling\n\n"
            f"Code ({len(code)} chars):\n{code[:2500]}\n\n"
            'Return ONLY valid JSON: {"score": <0-10>, "reason": "<one sentence>"}'
        )
        raw = self._llm([{"role": "user", "content": prompt}], timeout=25)
        try:
            m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
            r = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
            if m:
                return min(10.0, float(m.group(1))) / 10.0, r.group(1) if r else ""
        except Exception:
            pass
        return 0.5, ""  # neutral on failure

    # ── Combined fitness ──────────────────────────────────────────────────

    def score(
        self,
        code: str,
        tasks: Optional[list[str]] = None,
        pattern: str = "",
    ) -> LocalFitnessScore:
        """Compute three-signal fitness for a code string.

        Parameters
        ----------
        code:   Python source string to evaluate.
        tasks:  List of string tasks to pass to code's run(task=...) function.
        pattern: Pattern name (e.g. "react", "planning") for keyword scoring.
        """
        if not code:
            return LocalFitnessScore(exec_penalty=1.0, feedback="empty code")

        # Syntax check
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        if subprocess.run(["python3", "-m", "py_compile", tmp_path],
                          capture_output=True).returncode != 0:
            Path(tmp_path).unlink(missing_ok=True)
            return LocalFitnessScore(exec_penalty=1.0, feedback="syntax error")

        # Execution check
        exec_penalty = 0.0
        try:
            spec = importlib.util.spec_from_file_location("_evo_probe", tmp_path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if tasks and hasattr(mod, "run"):
                result = str(mod.run(task=tasks[0]))
                if len(result.strip()) <= 3:
                    exec_penalty = 0.3  # ran but returned near-empty
        except (ConnectionRefusedError, OSError, TimeoutError):
            exec_penalty = 0.2  # external dependency unavailable — not a code bug
        except Exception:
            exec_penalty = 0.5  # logic error
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if exec_penalty >= 1.0:
            return LocalFitnessScore(exec_penalty=1.0, feedback="hard execution failure")

        s1 = self.ast_score(code)
        s2 = self.keyword_score(code, pattern)
        s3, feedback = self.llm_judge(
            code, pattern or "a real algorithm (not just an LLM wrapper)"
        )

        return LocalFitnessScore(
            ast_complexity=s1,
            keyword_coverage=s2,
            llm_judge=s3,
            exec_penalty=exec_penalty,
            feedback=feedback,
        )

    # ── Variant generation ────────────────────────────────────────────────

    def generate_variant(
        self,
        base_code: str,
        topic: str,
        iteration: int,
        prev_score: float,
        context: str = "",
    ) -> str:
        """Ask the local LLM to produce an improved variant of base_code."""
        hint = (
            f"Previous version scored {prev_score:.0%}. Iteration {iteration}: improve it."
            if prev_score < 1.0 else "Optimise for clarity and robustness."
        )
        prompt = (
            f"You are evolving a Python skill. Topic: {topic}.\n"
            + (f"Reference context:\n{context[:800]}\n\n" if context else "")
            + f"Current version ({hint}):\n\n{base_code[:3000]}\n\n"
            "Rules:\n"
            "  • Keep the same SKILL_NAME constant and def run(task, **kwargs) -> str interface.\n"
            "  • Make run() actually compute something — no bare LLM-wrapper.\n"
            "  • No project-specific imports (no joe, no hermes internals).\n"
            "  • Output ONLY valid Python, no markdown fences.\n"
        )
        raw = self._llm([{"role": "user", "content": prompt}], timeout=120)
        if not raw:
            return base_code
        # Strip accidental markdown fences
        return re.sub(r"```\w*\n?", "", raw).strip().strip("`").strip()

    # ── Main evolution loop ───────────────────────────────────────────────

    def evolve(
        self,
        skill_name: str,
        topic: str,
        initial_code: str,
        iterations: int = 3,
        tasks: Optional[list[str]] = None,
        context: str = "",
        verbose: bool = True,
    ) -> tuple[str, float]:
        """Run the full evolutionary loop.

        Parameters
        ----------
        skill_name:   Identifier used for DB storage and keyword matching.
        topic:        Natural-language topic for LLM prompts (e.g. "ReAct agent").
        initial_code: Starting Python source to evolve.
        iterations:   Number of mutation rounds (3 is usually enough).
        tasks:        Tasks to pass to run() during execution checks.
        context:      Extra context for the mutation prompt (e.g. repo README snippet).
        verbose:      Print progress to stdout.

        Returns
        -------
        (best_code, best_score)  — best_score is composite 0.0–1.0.
        """
        if tasks is None:
            tasks = [
                f"perform a {topic} task",
                "hello world",
                f"analyze {topic}",
            ]

        pattern = re.sub(r"[_\d]+", " ", skill_name).strip()

        result = self.score(initial_code, tasks=tasks, pattern=pattern)
        best_code, best_score = initial_code, result.composite
        self._db_save(skill_name, initial_code, best_score, 0)

        if verbose:
            print(f"  [local-evo] baseline={best_score:.2f}  "
                  f"(ast={result.ast_complexity:.2f} kw={result.keyword_coverage:.2f} "
                  f"llm={result.llm_judge:.2f})")

        for i in range(1, iterations + 1):
            if verbose:
                print(f"  [local-evo] iteration {i}/{iterations}…")

            variant = self.generate_variant(best_code, topic, i, best_score, context)
            res = self.score(variant, tasks=tasks, pattern=pattern)
            score = res.composite
            self._db_save(skill_name, variant, score, i)

            if verbose:
                print(f"  [local-evo]   score={score:.2f} (best={best_score:.2f})  {res.feedback}")

            if score > best_score:
                best_code, best_score = variant, score
                if verbose:
                    print(f"  [local-evo]   ✓ improved → {best_score:.0%}")

        return best_code, best_score
