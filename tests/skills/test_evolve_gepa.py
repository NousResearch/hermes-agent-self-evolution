"""GEPA skill-evolution pipeline: real dspy.GEPA API, no fallback, skill text as
the optimised instructions, pluggable feedback metric.

No live LLM calls: every LM is a stub. ``StubLM`` targets dspy 3.5's engine
API (``BaseLM(engine=...)``); if dspy changes it, these tests fail rather
than skip, because a skipped proof proves nothing.
"""
import copy
import json

import dspy
import pytest
from click.testing import CliRunner
from dspy.clients.base_lm import BaseLM
from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
from dspy.lm15 import Message, Response, TextPart, Usage
from dspy.utils.dummies import DummyLM

from evolution.core import fitness
from evolution.skills import evolve_skill as es
from evolution.skills.skill_module import SkillModule, extract_skill_text

MARKER = "ALWAYS-LIST-THE-FOUR-CODEWORDS"
BODY = (
    "# Demo Review Skill\n\n"
    "## When to Use\nUse this skill when asked to review a short change.\n\n"
    "## Procedure\n"
    "1. Read the change carefully and note what it is meant to do.\n"
    "2. Check inputs, error handling and edge cases.\n"
    "3. Report concrete findings with a short explanation for each.\n\n"
    "## Pitfalls\n- Do not invent problems that are not in the change.\n"
    "- Keep the report short and specific.\n"
)
EVOLVED = BODY + f"\n## Output\n{MARKER}: alpha, bravo, charlie, delta.\n"
RUBRIC = "Mentions alpha bravo charlie delta"
GOOD = "alpha bravo charlie delta"
BAD = "nothing relevant here"


# ── stub LMs ─────────────────────────────────────────────────────────────────

class _RespondEngine(DummyEngine):
    """Answers from ``owner.respond(system_prompt, last_user_message) -> fields``.

    dspy 3.5 ChainOfThought: without native reasoning, its ``dspy.Reasoning``
    field goes over the wire as ``plan`` (``__dspy_reasoning_fallback``), so
    stub answers use ``plan`` rather than ``reasoning``.
    """

    def _complete_messages(self, messages):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        fields = self.owner.respond(system, messages[-1]["content"])
        text = self.owner._format_answer_fields(fields)
        return Response(id=None, model="stub", message=Message.assistant([TextPart(text)]),
                        finish_reason="stop",
                        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))


class StubLM(DummyLM):
    """Offline LM whose answer depends on the system prompt (the instructions)."""

    def __init__(self, respond):
        engine = _RespondEngine(self)
        BaseLM.__init__(self, "stub", "chat", 0.0, 1000, False,
                        engine=engine, async_engine=AsyncDummyEngine(engine))
        self.answers, self.follow_examples, self.reasoning = {}, False, False
        self.adapter, self._cache_responses = None, False
        self.respond = respond
        self.calls = []

    def copy(self, **kwargs):
        return BaseLM.copy(self, **kwargs)


def task_lm():
    def respond(system, user):
        return {"plan": "stub plan", "output": GOOD if MARKER in system else BAD}
    return StubLM(respond)


def reflection_lm():
    return DummyLM({"": {"new_instruction": EVOLVED}})   # JSONAdapter-formatted


def examples(n):
    return [dspy.Example(task_input=f"Review change number {i}.",
                         expected_behavior=RUBRIC).with_inputs("task_input")
            for i in range(n)]


KEYWORD = fitness.make_skill_metric("keyword")


# ── (c) skill text is the optimisable instructions ───────────────────────────

def test_skill_text_is_the_instructions_not_an_input():
    module = SkillModule(BODY)
    (name, predictor), = module.named_predictors()
    assert set(predictor.signature.input_fields) == {"task_input"}
    assert predictor.signature.instructions == BODY.strip()
    assert module.skill_text == BODY.strip()


def test_skill_text_reaches_the_lm_as_instructions():
    seen = {}

    def respond(system, user):
        seen["system"], seen["user"] = system, user
        return {"plan": "p", "output": "ok"}

    with dspy.context(lm=StubLM(respond)):
        assert SkillModule(BODY)(task_input="Review this.").output == "ok"
    assert "Check inputs, error handling and edge cases." in seen["system"]
    assert "Check inputs, error handling" not in seen["user"]


def test_empty_skill_is_refused():
    with pytest.raises(ValueError):
        SkillModule("   \n")


def test_rewritten_instructions_are_extracted():
    """Same update GEPA's build_program applies: deepcopy, then with_instructions."""
    module = SkillModule(BODY)
    evolved = copy.deepcopy(module)
    for _, predictor in evolved.named_predictors():
        predictor.signature = predictor.signature.with_instructions(EVOLVED)
    assert extract_skill_text(evolved) == EVOLVED.strip() != module.skill_text
    assert module.skill_text == BODY.strip()


def test_extract_requires_exactly_one_predictor():
    class Two(dspy.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = dspy.Predict("x -> y"), dspy.Predict("x -> y")
    with pytest.raises(ValueError):
        extract_skill_text(Two())


# ── (a) + (b) real GEPA with a reflection LM, output differs from input ──────

def test_gepa_evolves_the_skill_text_offline():
    baseline = SkillModule(BODY)
    optimised = es.run_gepa(baseline, examples(3), examples(2), metric=KEYWORD,
                            task_lm=task_lm(), reflection_lm=reflection_lm(),
                            budget={"max_metric_calls": 40})
    evolved = extract_skill_text(optimised)
    assert evolved != baseline.skill_text
    assert MARKER in evolved
    assert baseline.skill_text == BODY.strip()            # student left untouched
    with dspy.context(lm=task_lm()):
        assert optimised(task_input="Review change 9.").output == GOOD
        assert baseline(task_input="Review change 9.").output == BAD


def test_gepa_errors_propagate_without_fallback(monkeypatch):
    def broken_gepa(*args, **kwargs):
        raise TypeError("GEPA.__init__() got an unexpected keyword argument 'max_steps'")

    def forbidden(*args, **kwargs):
        raise AssertionError("MIPROv2 must never be used")

    monkeypatch.setattr(dspy, "GEPA", broken_gepa)
    monkeypatch.setattr(dspy, "MIPROv2", forbidden)
    with pytest.raises(TypeError, match="max_steps"):
        es.run_gepa(SkillModule(BODY), examples(2), examples(1), metric=KEYWORD,
                    task_lm=task_lm(), reflection_lm=reflection_lm(),
                    budget={"auto": "light"})


def test_no_fallback_optimiser_in_source():
    source = open(es.__file__, encoding="utf-8").read()
    assert "MIPROv2" not in source and "max_steps" not in source


def test_optimizer_gets_reflection_lm_and_real_budget(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(dspy, "GEPA", Recorder)
    lm = reflection_lm()
    es.build_optimizer(KEYWORD, reflection_lm=lm, budget={"max_metric_calls": 7})
    assert seen == {"metric": KEYWORD, "reflection_lm": lm, "max_metric_calls": 7}


def test_reflection_lm_is_required():
    with pytest.raises(ValueError, match="reflection"):
        es.build_optimizer(KEYWORD, reflection_lm=None, budget={"auto": "light"})


@pytest.mark.parametrize("kwargs", [
    {}, {"auto": "light", "max_metric_calls": 5}, {"auto": "extreme"},
    {"max_metric_calls": 0}, {"max_full_evals": True}, {"max_full_evals": -1},
])
def test_budget_must_be_exactly_one_valid_value(kwargs):
    with pytest.raises(ValueError):
        es.gepa_budget(**kwargs)


def test_budget_resolution():
    assert es.resolve_budget(10, None, None) == {"max_full_evals": 10}
    assert es.resolve_budget(10, "light", None) == {"auto": "light"}
    assert es.resolve_budget(10, None, 50) == {"max_metric_calls": 50}
    with pytest.raises(ValueError):
        es.resolve_budget(10, "light", 50)


# ── (d) pluggable metric with feedback ───────────────────────────────────────

def test_metric_has_gepa_signature_and_returns_feedback():
    gold = examples(1)[0]
    result = KEYWORD(gold, dspy.Prediction(output="alpha bravo"), None, "predictor.predict", None)
    assert 0 < result.score < 1
    assert "Missing" in result.feedback and "charlie" in result.feedback
    assert KEYWORD.score(gold, dspy.Prediction(output="")) == 0.0
    assert KEYWORD(gold, dspy.Prediction(output="  ")).feedback == "The response was empty."


def test_custom_scorer_is_pluggable():
    metric = fitness.SkillMetric(lambda ex, pred: (7.0, "too good"), name="custom")
    out = metric(examples(1)[0], dspy.Prediction(output="x"))
    assert out.score == 1.0 and out.feedback == "too good"      # clamped to [0, 1]
    with pytest.raises(TypeError):
        fitness.SkillMetric("not callable")


def test_judge_metric_scores_rubric_without_seeing_the_skill():
    prompts = []

    def respond(system, user):
        prompts.append(system + user)
        return {"plan": "p", "correctness": 1.0, "procedure_following": 0.5,
                "conciseness": 0.0, "feedback": "Name the failing edge case."}

    metric = fitness.make_skill_metric("judge", judge_lm=StubLM(respond))
    out = metric(examples(1)[0], dspy.Prediction(output="some answer"))
    assert out.score == pytest.approx(0.5 * 1.0 + 0.3 * 0.5)
    assert "Name the failing edge case." in out.feedback
    assert prompts and all(MARKER not in p and "## Procedure" not in p for p in prompts)


def test_unparseable_judge_scores_are_not_neutral():
    assert fitness._parse_score("high") is None
    assert fitness._parse_score(float("nan")) is None
    assert fitness._parse_score(True) is None
    assert fitness._parse_score("0.7") == 0.7
    assert fitness._parse_score(3) == 1.0


def test_unknown_metric_is_refused():
    with pytest.raises(ValueError):
        fitness.make_skill_metric("vibes")


# ── evolve() end to end, offline ─────────────────────────────────────────────

def _repo_and_dataset(tmp_path):
    repo = tmp_path / "hermes"
    skill_dir = repo / "skills" / "testing" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: demo\ndescription: Demo review skill\n---\n\n{BODY}")
    data = tmp_path / "golden"
    data.mkdir()
    rows = [{"task_input": f"Review change number {i}.", "expected_behavior": RUBRIC}
            for i in range(8)]
    (data / "golden.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return repo, data


def test_evolve_end_to_end_writes_a_changed_skill(tmp_path, monkeypatch):
    repo, data = _repo_and_dataset(tmp_path)
    built = []

    def fake_make_lm(model, **kwargs):
        built.append((model, kwargs))
        return reflection_lm() if model == "stub/reflection" else task_lm()

    monkeypatch.setattr(es, "make_lm", fake_make_lm)
    monkeypatch.chdir(tmp_path)
    es.evolve("demo", eval_source="golden", dataset_path=str(data),
              optimizer_model="stub/reflection", eval_model="stub/task",
              hermes_repo=str(repo), metric="keyword", max_metric_calls=40)

    assert ("stub/reflection", es.REFLECTION_LM_KWARGS) in built
    assert ("stub/task", {}) in built
    assert len(built) == 2                                   # no judge LM for --metric keyword
    run_dir, = (tmp_path / "output" / "demo").iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["optimizer"] == "dspy.GEPA"
    assert metrics["gepa_budget"] == {"max_metric_calls": 40}
    assert "iterations" in metrics and metrics["iterations"] is None   # old schema key kept
    assert metrics["optimizer_model"] == "stub/reflection"
    assert metrics["metric"] == "keyword-overlap" and metrics["skill_changed"] is True
    assert metrics["evolved_score"] > metrics["baseline_score"]
    evolved = (run_dir / "evolved_skill.md").read_text()
    baseline = (run_dir / "baseline_skill.md").read_text()
    assert evolved != baseline and MARKER in evolved and evolved.startswith("---\nname: demo")


def test_cli_rejects_two_budgets_before_doing_anything(tmp_path):
    repo, _ = _repo_and_dataset(tmp_path)
    result = CliRunner().invoke(es.main, ["--skill", "demo", "--hermes-repo", str(repo),
                                          "--auto", "light", "--max-metric-calls", "5",
                                          "--dry-run"])
    assert result.exit_code == 2 and "not both" in result.output


def test_cli_dry_run_reports_budget_and_models(tmp_path):
    repo, _ = _repo_and_dataset(tmp_path)
    result = CliRunner().invoke(es.main, ["--skill", "demo", "--hermes-repo", str(repo),
                                          "--auto", "light", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dspy.GEPA" in result.output and "'auto': 'light'" in result.output
