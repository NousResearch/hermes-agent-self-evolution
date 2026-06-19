"""Tests for LocalCodeEvolver — run offline, no LLM required for signal 1+2."""

import pytest
from evolution.code.local_evolver import LocalCodeEvolver, LocalFitnessScore, PATTERN_KEYWORDS

WRAPPER_CODE = '''
SKILL_NAME = "wrapper"
import urllib.request, json

def run(task, **kwargs):
    req = urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions",
        data=json.dumps({"messages":[{"role":"user","content":task}]}).encode(),
        headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]
'''

REAL_CODE = '''
SKILL_NAME = "react_agent"
import sqlite3, time

def parse_intent(task):
    for kw in ["search", "find", "get"]:
        if kw in task.lower():
            return "search"
    return "analyze"

def observe(result):
    return bool(result and len(result) > 5)

def next_action(observation):
    return "complete" if observation else "retry"

def run(task, **kwargs):
    intent = parse_intent(task)
    state = "attempt"
    retries = 0
    result = f"[thought] intent={intent}"
    while state != "complete" and retries < 3:
        observation = observe(result)
        action = next_action(observation)
        if action == "retry":
            retries += 1
            time.sleep(0.01)
        else:
            state = "complete"
    return f"[done] {result}"
'''

SYNTAX_ERROR_CODE = "def run(task\n    return task"


class TestAstScore:
    def test_wrapper_scores_low(self):
        evolver = LocalCodeEvolver()
        score = evolver.ast_score(WRAPPER_CODE)
        assert score < 0.4, f"Wrapper should score below 0.4, got {score}"

    def test_real_impl_scores_high(self):
        evolver = LocalCodeEvolver()
        score = evolver.ast_score(REAL_CODE)
        assert score > 0.5, f"Real impl should score above 0.5, got {score}"

    def test_syntax_error_returns_zero(self):
        evolver = LocalCodeEvolver()
        assert evolver.ast_score(SYNTAX_ERROR_CODE) == 0.0

    def test_empty_code_returns_zero(self):
        evolver = LocalCodeEvolver()
        assert evolver.ast_score("") == 0.0


class TestKeywordScore:
    def test_react_keywords_present(self):
        evolver = LocalCodeEvolver()
        score = evolver.keyword_score(REAL_CODE, "react")
        assert score > 0.4, f"React keywords should score above 0.4, got {score}"

    def test_wrapper_penalised_generic(self):
        evolver = LocalCodeEvolver()
        score = evolver.keyword_score(WRAPPER_CODE, "")
        assert score <= 0.5

    def test_all_patterns_have_keywords(self):
        for pattern in PATTERN_KEYWORDS:
            assert len(PATTERN_KEYWORDS[pattern]) >= 3


class TestLocalFitnessScore:
    def test_composite_weights(self):
        fs = LocalFitnessScore(ast_complexity=1.0, keyword_coverage=1.0, llm_judge=1.0)
        assert fs.composite == 1.0

    def test_exec_penalty_applied(self):
        fs = LocalFitnessScore(ast_complexity=1.0, keyword_coverage=1.0, llm_judge=1.0, exec_penalty=0.2)
        assert fs.composite == pytest.approx(0.8, abs=0.01)

    def test_full_penalty_zeroes_score(self):
        fs = LocalFitnessScore(ast_complexity=1.0, keyword_coverage=1.0, llm_judge=1.0, exec_penalty=1.0)
        assert fs.composite == 0.0


class TestScoreWithoutLLM:
    """Score function tests that stub out LLM judge."""

    def test_syntax_error_returns_zero_composite(self, monkeypatch):
        evolver = LocalCodeEvolver()
        result = evolver.score(SYNTAX_ERROR_CODE)
        assert result.composite == 0.0

    def test_real_code_ast_plus_keyword_without_llm(self, monkeypatch):
        evolver = LocalCodeEvolver()
        # Stub out LLM judge to return neutral 0.5
        monkeypatch.setattr(evolver, "llm_judge", lambda code, pattern: (0.5, "stubbed"))
        result = evolver.score(REAL_CODE, tasks=[], pattern="react")
        # s1=high, s2=high, s3=0.5 → composite should be above 0.4
        assert result.composite > 0.4, f"Expected >0.4, got {result.composite}"
