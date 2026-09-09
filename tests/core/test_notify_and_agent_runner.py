"""Tests for run reporting and agent-in-the-loop evaluation.

The notification tests pin the behaviour that replaced a notifier which
crashed on send (``_standalone_send() takes 2 positional arguments but 3 were
given``). Combined with an eval source that produced nothing, the deployed
steady state was: fail every skill, tell no one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.core.agent_runner import (
    AgentEvaluator,
    AgentTask,
    CallableGrader,
    HermesAgentBackend,
    RegexGrader,
    ScriptedBackend,
    SubstringGrader,
    count_metrics,
    tasks_from_examples,
)
from evolution.core.notify import (
    DeliveryResult,
    FileChannel,
    Notifier,
    RunSummary,
    WebhookChannel,
)


# ── notification ────────────────────────────────────────────────────────


class ExplodingChannel:
    name = "exploding"

    def send(self, subject, body):
        raise RuntimeError("_standalone_send() takes 2 positional arguments but 3 were given")


class TestFileChannel:
    def test_writes_a_latest_snapshot_and_appends_to_the_log(self, tmp_path):
        channel = FileChannel(tmp_path / "status")
        assert channel.send("subject one", "body one").delivered
        assert channel.send("subject two", "body two").delivered

        latest = (tmp_path / "status" / "latest.txt").read_text()
        log = (tmp_path / "status" / "runs.log").read_text()
        assert "subject two" in latest
        assert "subject one" in log and "subject two" in log


class TestNotifier:
    def test_a_channel_that_raises_is_recorded_not_swallowed(self, tmp_path):
        notifier = Notifier([ExplodingChannel(), FileChannel(tmp_path)])
        outcome = notifier.send("s", "b")

        assert outcome.delivered, "the file channel should still succeed"
        failed = [r for r in outcome.results if not r.delivered]
        assert len(failed) == 1
        assert "_standalone_send" in failed[0].detail

    def test_every_channel_failing_is_reported_as_undelivered(self):
        outcome = Notifier([ExplodingChannel()]).send("s", "b")
        assert outcome.delivered is False
        assert "FAILED" in outcome.render()

    def test_no_channels_renders_cleanly(self):
        assert Notifier([]).send("s", "b").render() == "no channels configured"

    def test_from_env_always_includes_a_local_record(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EVOLUTION_WEBHOOK_URL", raising=False)
        notifier = Notifier.from_env(tmp_path)
        assert any(isinstance(c, FileChannel) for c in notifier.channels)

    def test_from_env_adds_a_webhook_when_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVOLUTION_WEBHOOK_URL", "https://example.invalid/hook")
        monkeypatch.setenv("EVOLUTION_WEBHOOK_SECRET", "shh")
        notifier = Notifier.from_env(tmp_path)
        assert any(isinstance(c, WebhookChannel) for c in notifier.channels)

    def test_webhook_secret_can_come_from_a_file(self, tmp_path, monkeypatch):
        secret_file = tmp_path / "secret"
        secret_file.write_text("from-file\n")
        monkeypatch.setenv("EVOLUTION_WEBHOOK_URL", "https://example.invalid/hook")
        monkeypatch.delenv("EVOLUTION_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("EVOLUTION_WEBHOOK_SECRET_FILE", str(secret_file))

        webhook = next(c for c in Notifier.from_env(tmp_path).channels if isinstance(c, WebhookChannel))
        assert webhook.secret == "from-file"

    def test_unconfigured_webhook_fails_without_making_a_request(self):
        assert WebhookChannel(url="", secret="").send("s", "b").delivered is False


class TestRunSummary:
    def test_a_failure_makes_the_exit_code_nonzero(self):
        """A sweep that evaluated nothing must not look healthy to a scheduler."""
        summary = RunSummary(subject="s", failed=[("skill-a", "no data")])
        assert summary.ok is False
        assert summary.exit_code == 1

    def test_success_exits_zero(self):
        assert RunSummary(subject="s", succeeded=["a"]).exit_code == 0

    def test_skips_alone_are_not_failures(self):
        assert RunSummary(subject="s", skipped=[("a", "held")]).exit_code == 0

    def test_render_lists_every_outcome(self):
        text = RunSummary(
            subject="s",
            succeeded=["a"],
            failed=[("b", "boom")],
            skipped=[("c", "held")],
            notes=["extra"],
        ).render()
        assert "ok    a" in text and "FAIL  b" in text and "skip  c" in text and "extra" in text

    def test_delivery_never_decides_the_exit_code(self, tmp_path):
        """A delivered message about a failed run still exits non-zero."""
        summary = RunSummary(subject="s", failed=[("a", "x")])
        outcome = Notifier([FileChannel(tmp_path)]).send(summary.subject, summary.render())
        assert outcome.delivered is True
        assert summary.exit_code == 1

    def test_undelivered_success_still_exits_zero(self):
        summary = RunSummary(subject="s", succeeded=["a"])
        outcome = Notifier([ExplodingChannel()]).send(summary.subject, summary.render())
        assert outcome.delivered is False
        assert summary.exit_code == 0


# ── graders ─────────────────────────────────────────────────────────────


class TestGraders:
    def test_substring_all_scores_partial_credit(self):
        grader = SubstringGrader(["alpha", "beta", "gamma"])
        assert grader.grade("alpha and beta") == pytest.approx(2 / 3)

    def test_substring_any_is_binary(self):
        grader = SubstringGrader(["alpha", "beta"], mode="any")
        assert grader.grade("only beta") == 1.0
        assert grader.grade("neither") == 0.0

    def test_substring_is_case_insensitive_by_default(self):
        assert SubstringGrader(["Alpha"]).grade("alpha") == 1.0

    def test_empty_needles_score_zero(self):
        assert SubstringGrader([]).grade("anything") == 0.0

    def test_regex_grader(self):
        assert RegexGrader([r"\d{3}-\d{4}"]).grade("call 555-1234") == 1.0
        assert RegexGrader([r"\d{3}-\d{4}"]).grade("no number") == 0.0

    def test_callable_grader_clamps_and_survives_exceptions(self):
        assert CallableGrader(lambda out: 9.0).grade("x") == 1.0
        assert CallableGrader(lambda out: 1 / 0).grade("x") == 0.0


# ── metrics extraction ──────────────────────────────────────────────────


class TestCountMetrics:
    def test_counts_turns_tools_and_tokens_from_dicts(self):
        log = count_metrics(
            [
                {"role": "user", "content": "hi", "token_count": 5},
                {"role": "assistant", "tool_calls": [{}, {}], "token_count": 10},
                {"role": "tool", "tool_name": "read_file", "token_count": 3},
                {"role": "assistant", "token_count": 7},
            ]
        )
        assert log.api_turns == 2
        assert log.tool_calls == 2
        assert log.total_tokens == 25
        assert log.tools_used == ["read_file"]

    def test_object_shaped_messages_work_too(self):
        class Msg:
            role = "assistant"
            tool_calls = None
            tool_name = ""
            token_count = 4

        assert count_metrics([Msg()]).api_turns == 1

    def test_empty_transcript(self):
        assert count_metrics([]).api_turns == 0


# ── evaluator ───────────────────────────────────────────────────────────


def task(task_id="t1", needles=("expected",), **kw):
    return AgentTask(id=task_id, prompt="do it", grader=SubstringGrader(list(needles)), **kw)


class TestAgentEvaluator:
    def test_scores_a_correct_answer(self):
        backend = ScriptedBackend(lambda prompt, system: "the expected answer")
        run = AgentEvaluator(backend).evaluate("skill text", [task()])
        assert run.accuracy == 1.0
        assert run.errors == 0

    def test_the_skill_text_reaches_the_backend(self):
        seen = {}

        def responder(prompt, system):
            seen["system"] = system
            return "expected"

        AgentEvaluator(ScriptedBackend(responder)).evaluate("MY SKILL", [task()])
        assert seen["system"] == "MY SKILL"

    def test_reps_produce_multiple_observations(self):
        backend = ScriptedBackend(lambda p, s: "expected")
        run = AgentEvaluator(backend, reps=3).evaluate("s", [task()])
        assert len(run.results) == 3
        assert len({r.task_id for r in run.results}) == 3

    def test_a_backend_error_scores_zero_and_is_counted(self):
        def boom(prompt, system):
            raise RuntimeError("model down")

        run = AgentEvaluator(ScriptedBackend(boom)).evaluate("s", [task()])
        assert run.results[0].score == 0.0
        assert run.errors == 1

    def test_workspace_is_built_and_cleaned_up(self):
        created: list[Path] = []

        def build(path: Path):
            created.append(path)
            (path / "planted.txt").write_text("expected")

        def responder(prompt, system):
            return (created[-1] / "planted.txt").read_text()

        run = AgentEvaluator(ScriptedBackend(responder)).evaluate(
            "s", [task(build_workspace=build)]
        )
        assert run.accuracy == 1.0
        assert created and not created[0].exists(), "workspace should be removed"

    def test_a_failing_workspace_build_is_reported_not_raised(self):
        def build(path):
            raise OSError("disk full")

        run = AgentEvaluator(ScriptedBackend(lambda p, s: "x")).evaluate(
            "s", [task(build_workspace=build)]
        )
        assert run.errors == 1
        assert "workspace build failed" in run.results[0].error

    def test_run_aggregates_are_json_safe(self):
        run = AgentEvaluator(ScriptedBackend(lambda p, s: "expected")).evaluate("s", [task()])
        payload = run.as_dict()
        assert payload["tasks"] == 1 and payload["accuracy"] == 1.0


class TestHermesAgentBackend:
    def test_missing_repo_reports_unavailable_rather_than_raising(self, tmp_path):
        ok, detail = HermesAgentBackend(tmp_path / "nope").available()
        assert ok is False
        assert "not found" in detail

    def test_a_repo_without_run_agent_is_unavailable(self, tmp_path):
        (tmp_path / "agent").mkdir()
        ok, detail = HermesAgentBackend(tmp_path).available()
        assert ok is False

    def test_unavailable_backend_returns_an_errored_log_not_an_exception(self, tmp_path):
        log = HermesAgentBackend(tmp_path / "nope").run("p", "s", ["file"], None, 10)
        assert log.error

    def test_kwargs_are_filtered_to_the_constructor_signature(self):
        class OldAgent:
            def __init__(self, model=None):
                pass

        filtered = HermesAgentBackend._filter_kwargs(
            OldAgent, {"model": "m", "enabled_toolsets": ["file"], "headless": True}
        )
        assert filtered == {"model": "m"}

    def test_var_keyword_agents_accept_everything(self):
        class NewAgent:
            def __init__(self, **kwargs):
                pass

        filtered = HermesAgentBackend._filter_kwargs(NewAgent, {"model": "m", "anything": 1})
        assert filtered == {"model": "m", "anything": 1}


class TestTasksFromExamples:
    def test_rubric_content_words_become_the_grader(self):
        class Example:
            task_input = "summarize the incident"
            expected_behavior = "Should mention rollback and downtime"
            category = "ops"

        tasks = tasks_from_examples([Example()])
        assert len(tasks) == 1
        assert "rollback" in tasks[0].grader.needles
        # Rubric filler must not become ground truth.
        assert "should" not in tasks[0].grader.needles

    def test_examples_without_input_are_dropped(self):
        class Empty:
            task_input = ""
            expected_behavior = "x"

        assert tasks_from_examples([Empty()]) == []
