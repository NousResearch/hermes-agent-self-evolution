"""Tests for the continuous loop.

Nothing here spawns an optimizer, runs a benchmark, or touches a model. Every
side effect the loop has is behind an injectable callable, and each test
substitutes a recorder for it, so a full "detect problem -> optimize -> propose"
cycle runs in milliseconds with no API key and no hermes-agent checkout.

The fake hermes-agent repo is a directory with a ``batch_runner.py`` in it,
which is the marker the real repo actually has at its root.
"""

import io
import json

import pytest
from click.testing import CliRunner
from rich.console import Console

from evolution.core.gates import KNOWN_BENCHMARKS, GateResult, GateStatus
from evolution.monitor.loop import (
    API_KEY_ENV_VARS,
    DEFAULT_SCHEDULE,
    PHASE_DISPATCH,
    CycleStatus,
    DispatchStatus,
    LoopConfig,
    ProcessResult,
    build_command,
    cron_line,
    default_history_path,
    has_api_key,
    looks_like_hermes_repo,
    main,
    preflight,
    run_cycle,
)
from evolution.monitor.metrics import (
    BENCHMARK_SCORE,
    OPTIMIZATION_RUN,
    SECONDS_PER_DAY,
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    USER_CORRECTION,
    MetricPoint,
    MetricStore,
)
from evolution.monitor.triage import TargetType, TriageConfig

T0 = 1_700_000_000.0
DAY = SECONDS_PER_DAY
KEYED_ENV = {"OPENAI_API_KEY": "test-key-not-a-real-one"}


def point(metric, target, value, days_ago=1.0, samples=1, **metadata):
    return MetricPoint(
        metric=metric,
        target=target,
        value=value,
        timestamp=T0 - days_ago * DAY,
        samples=samples,
        source="test",
        metadata=dict(metadata),
    )


@pytest.fixture
def out():
    """A Console that captures instead of printing into the test log."""
    return Console(file=io.StringIO(), width=200)


@pytest.fixture
def hermes_repo(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "tools").mkdir(parents=True)
    (repo / "batch_runner.py").write_text("# stand-in for the real batch runner\n")
    return repo


@pytest.fixture
def store(tmp_path):
    return MetricStore(tmp_path / "history" / "metrics.jsonl", clock=lambda: T0)


class FakeDispatcher:
    """Stands in for the subprocess runner and remembers every invocation."""

    def __init__(self, returncode=0, output=""):
        self.returncode = returncode
        self.output = output
        self.calls = []

    def __call__(self, command, env, timeout):
        self.calls.append({"command": list(command), "env": dict(env), "timeout": timeout})
        return ProcessResult(returncode=self.returncode, output=self.output)


class FakeBranches:
    """Stands in for reading evolve/ refs out of the hermes-agent checkout.

    ``produces`` is the branch a successful dispatch is pretended to create, so
    a test can distinguish a phase that proposed something from one that ran
    cleanly and decided nothing was deployable.
    """

    def __init__(self, produces="evolve/target-20260731_000000"):
        self.produces = produces
        self._seen = False

    def __call__(self, repo):
        if not self.produces:
            return set()
        if self._seen:
            return {self.produces}
        self._seen = True
        return set()


class FakeBenchmarks:
    """Returns a score for known benchmarks and UNAVAILABLE for the rest."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def __call__(self, repo, name, baseline=None, fast=True):
        self.calls.append({"repo": repo, "name": name, "baseline": baseline, "fast": fast})
        score = self.scores.get(name)
        if score is None:
            return GateResult(name, GateStatus.UNAVAILABLE, f"benchmark '{name}' not found")
        return GateResult(
            name,
            GateStatus.PASSED,
            f"scored {score:.1%}",
            score=score,
            baseline=baseline,
        )


def exploding_dispatcher(command, env, timeout):  # pragma: no cover - must not run
    raise AssertionError(f"dry run must not dispatch: {command}")


def exploding_benchmarks(repo, name, baseline=None, fast=True):  # pragma: no cover
    raise AssertionError(f"dry run must not run benchmarks: {name}")


def cycle(store, hermes_repo=None, **kwargs):
    """run_cycle with the offline defaults every test wants."""
    config = LoopConfig(
        hermes_repo=hermes_repo,
        triage=kwargs.pop("triage", TriageConfig()),
        max_targets=kwargs.pop("max_targets", 1),
        benchmarks=kwargs.pop("benchmarks", ("tblite",)),
        cooldown_days=kwargs.pop("cooldown_days", 14.0),
        iterations=kwargs.pop("iterations", None),
        python=kwargs.pop("python", "/usr/bin/python3"),
    )
    kwargs.setdefault("benchmark_runner", FakeBenchmarks())
    kwargs.setdefault("dispatcher", FakeDispatcher())
    kwargs.setdefault("branch_lister", FakeBranches())
    kwargs.setdefault("module_available", lambda module: True)
    kwargs.setdefault("env", dict(KEYED_ENV))
    kwargs.setdefault("now", T0)
    return run_cycle(config, store, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Dispatch table
# ──────────────────────────────────────────────────────────────────────────


class TestDispatchTable:
    @pytest.mark.parametrize(
        "target_type,phase,module,flag",
        [
            (TargetType.SKILL, 1, "evolution.skills.evolve_skill", "--skill"),
            (TargetType.TOOL, 2, "evolution.tools.evolve_tool_descriptions", "--tool"),
            (TargetType.PROMPT, 3, "evolution.prompts.evolve_prompt_section", "--section"),
            (TargetType.CODE, 4, "evolution.code.evolve_tool_code", "--tool"),
        ],
    )
    def test_each_target_type_maps_to_its_phase(self, target_type, phase, module, flag):
        entry = PHASE_DISPATCH[target_type]
        assert entry.phase == phase
        assert entry.module == module
        assert entry.flag == flag

    def test_benchmarks_have_no_entry_point(self):
        assert TargetType.BENCHMARK not in PHASE_DISPATCH

    def test_build_command_names_the_module_and_target(self):
        command = build_command(
            PHASE_DISPATCH[TargetType.SKILL], "arxiv", python="/usr/bin/python3"
        )
        assert command == [
            "/usr/bin/python3",
            "-m",
            "evolution.skills.evolve_skill",
            "--skill",
            "arxiv",
        ]

    def test_build_command_omits_iterations_by_default(self):
        command = build_command(PHASE_DISPATCH[TargetType.TOOL], "read_file")
        assert "--iterations" not in command

    def test_build_command_passes_iterations_when_asked(self):
        command = build_command(
            PHASE_DISPATCH[TargetType.PROMPT], "MEMORY_GUIDANCE", iterations=5
        )
        assert command[-4:] == ["MEMORY_GUIDANCE", "--iterations", "5", "--write"]

    def test_the_writing_phases_are_dispatched_with_write(self):
        """Without it the loop can never produce the PR its report claims."""
        for target_type in (TargetType.TOOL, TargetType.PROMPT):
            assert build_command(PHASE_DISPATCH[target_type], "x")[-1] == "--write"

    def test_phases_with_no_write_path_do_not_get_the_flag(self):
        for target_type in (TargetType.SKILL, TargetType.CODE):
            assert "--write" not in build_command(PHASE_DISPATCH[target_type], "x")


# ──────────────────────────────────────────────────────────────────────────
# Preflight
# ──────────────────────────────────────────────────────────────────────────


class TestPreflight:
    def test_passes_when_everything_is_present(self, hermes_repo):
        ok, reason, entry = preflight(
            TargetType.SKILL,
            hermes_repo,
            env=KEYED_ENV,
            module_available=lambda m: True,
        )
        assert ok and reason == ""
        assert entry.phase == 1

    def test_missing_phase_module_is_a_skip(self, hermes_repo):
        ok, reason, _ = preflight(
            TargetType.TOOL, hermes_repo, env=KEYED_ENV, module_available=lambda m: False
        )
        assert not ok
        assert "not installed" in reason

    def test_missing_repo_is_a_skip(self):
        ok, reason, _ = preflight(
            TargetType.SKILL, None, env=KEYED_ENV, module_available=lambda m: True
        )
        assert not ok
        assert "no hermes-agent checkout" in reason

    def test_a_directory_that_is_not_hermes_is_a_skip(self, tmp_path):
        ok, reason, _ = preflight(
            TargetType.SKILL,
            tmp_path / "empty",
            env=KEYED_ENV,
            module_available=lambda m: True,
        )
        assert not ok
        assert "no hermes-agent checkout" in reason

    def test_missing_api_key_is_a_skip(self, hermes_repo):
        ok, reason, _ = preflight(
            TargetType.SKILL, hermes_repo, env={}, module_available=lambda m: True
        )
        assert not ok
        assert "no API key" in reason

    def test_a_benchmark_target_has_nowhere_to_go(self, hermes_repo):
        ok, reason, entry = preflight(
            TargetType.BENCHMARK,
            hermes_repo,
            env=KEYED_ENV,
            module_available=lambda m: True,
        )
        assert not ok
        assert entry is None
        assert "no phase entry point" in reason

    def test_repo_detection_needs_a_real_marker(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert not looks_like_hermes_repo(empty)
        assert not looks_like_hermes_repo(None)
        assert not looks_like_hermes_repo(tmp_path / "missing")
        (empty / "agent").mkdir()
        assert looks_like_hermes_repo(empty)

    @pytest.mark.parametrize("name", API_KEY_ENV_VARS)
    def test_any_recognised_key_counts(self, name):
        assert has_api_key({name: "value"})

    def test_a_blank_key_does_not_count(self):
        assert not has_api_key({"OPENAI_API_KEY": "   "})


# ──────────────────────────────────────────────────────────────────────────
# A full cycle
# ──────────────────────────────────────────────────────────────────────────


class TestCycle:
    def test_once_runs_detect_triage_dispatch_and_record(self, store, hermes_repo, out):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert report.status is CycleStatus.PROPOSED
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0]["command"][2:] == [
            "evolution.skills.evolve_skill",
            "--skill",
            "arxiv",
        ]
        assert report.dispatches[0].phase == 1
        assert report.dispatches[0].status is DispatchStatus.PROPOSED

    def test_a_tool_target_goes_to_phase_two(self, store, hermes_repo, out):
        store.extend([point(TOOL_SELECTION_ACCURACY, "search_files", 0.4, samples=80)])
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert report.dispatches[0].phase == 2
        assert dispatcher.calls[0]["command"][2:] == [
            "evolution.tools.evolve_tool_descriptions",
            "--tool",
            "search_files",
            "--write",
        ]

    def test_a_prompt_target_goes_to_phase_three(self, store, hermes_repo, out):
        store.extend(
            [
                point(
                    USER_CORRECTION,
                    "MEMORY_GUIDANCE",
                    1.0,
                    days_ago=n + 1,
                    target_type="prompt",
                )
                for n in range(6)
            ]
        )
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert report.dispatches[0].phase == 3
        assert dispatcher.calls[0]["command"][2:] == [
            "evolution.prompts.evolve_prompt_section",
            "--section",
            "MEMORY_GUIDANCE",
            "--write",
        ]

    def test_a_code_target_goes_to_phase_four(self, store, hermes_repo, out):
        store.extend([point("tool_crash_free_rate", "file_tools", 0.4, samples=60)])
        dispatcher = FakeDispatcher()

        report = cycle(
            store,
            hermes_repo,
            dispatcher=dispatcher,
            triage=TriageConfig(
                extra_metric_types={"tool_crash_free_rate": TargetType.CODE}
            ),
            out=out,
        )

        assert report.dispatches[0].phase == 4
        assert dispatcher.calls[0]["command"][2:] == [
            "evolution.code.evolve_tool_code",
            "--tool",
            "file_tools",
        ]

    def test_the_child_process_is_told_where_hermes_lives(self, store, hermes_repo, out):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        dispatcher = FakeDispatcher()

        cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert dispatcher.calls[0]["env"]["HERMES_AGENT_REPO"] == str(hermes_repo)

    def test_the_child_environment_is_an_allowlist_not_a_copy(
        self, store, hermes_repo, out
    ):
        """A phase gets what a phase needs; the shell's secrets stay home.

        The monitor once handed dispatched phases a full copy of its own
        environment, which forwarded every token in the operator's shell to
        a subprocess that ultimately drives contributor-influenced code.
        """
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        dispatcher = FakeDispatcher()
        parent = {
            **KEYED_ENV,
            "PATH": "/usr/bin",
            "LC_CTYPE": "UTF-8",
            "AWS_SECRET_ACCESS_KEY": "not-for-phases",
            "GITHUB_TOKEN": "ghp_nope",
            "SSH_AUTH_SOCK": "/tmp/agent.sock",
        }

        cycle(store, hermes_repo, dispatcher=dispatcher, env=parent, out=out)

        child = dispatcher.calls[0]["env"]
        assert child["OPENAI_API_KEY"] == KEYED_ENV["OPENAI_API_KEY"]
        assert child["PATH"] == "/usr/bin"
        assert child["LC_CTYPE"] == "UTF-8"
        assert child["HERMES_AGENT_REPO"] == str(hermes_repo)
        for secret in ("AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "SSH_AUTH_SOCK"):
            assert secret not in child

    def test_a_proposal_is_written_back_into_history(self, store, hermes_repo, out):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])

        cycle(store, hermes_repo, out=out)

        recorded = store.query(metric=OPTIMIZATION_RUN, target="arxiv")
        assert len(recorded) == 1
        assert recorded[0].value == pytest.approx(1.0)
        assert recorded[0].metadata["status"] == "proposed"
        assert recorded[0].metadata["phase"] == 1

    def test_a_failed_optimization_is_not_recorded_as_a_success(
        self, store, hermes_repo, out
    ):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])

        report = cycle(
            store,
            hermes_repo,
            dispatcher=FakeDispatcher(returncode=1, output="boom"),
            out=out,
        )

        assert report.status is CycleStatus.FAILED
        recorded = store.query(metric=OPTIMIZATION_RUN)[0]
        assert recorded.value == pytest.approx(0.0)
        assert recorded.metadata["status"] == "failed"

    def test_a_missing_api_key_skips_rather_than_pretending(
        self, store, hermes_repo, out
    ):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, env={}, out=out)

        assert report.status is CycleStatus.SKIPPED
        assert dispatcher.calls == []
        assert "no API key" in report.dispatches[0].reason
        recorded = store.query(metric=OPTIMIZATION_RUN)[0]
        assert recorded.value == pytest.approx(0.0)
        assert recorded.metadata["status"] == "skipped"

    def test_a_phase_that_is_not_installed_skips_with_its_name(
        self, store, hermes_repo, out
    ):
        store.extend([point(TOOL_SELECTION_ACCURACY, "search_files", 0.4, samples=80)])

        report = cycle(
            store, hermes_repo, module_available=lambda module: False, out=out
        )

        assert report.status is CycleStatus.SKIPPED
        assert "evolution.tools.evolve_tool_descriptions" in report.dispatches[0].reason

    def test_a_missing_checkout_skips_the_whole_cycle(self, store, out):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        dispatcher = FakeDispatcher()

        report = cycle(store, None, dispatcher=dispatcher, out=out)

        assert report.status is CycleStatus.SKIPPED
        assert dispatcher.calls == []
        assert any("scheduled checks skipped" in note for note in report.notes)

    def test_a_target_proposed_last_week_is_left_alone(self, store, hermes_repo, out):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40),
                point(
                    OPTIMIZATION_RUN,
                    "arxiv",
                    1.0,
                    days_ago=3,
                    status="proposed",
                    phase=1,
                ),
            ]
        )
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert dispatcher.calls == []
        assert report.status is CycleStatus.SKIPPED
        assert "cooling down" in report.dispatches[0].reason

    def test_the_cooldown_expires(self, store, hermes_repo, out):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40),
                point(OPTIMIZATION_RUN, "arxiv", 1.0, days_ago=30, status="proposed"),
            ]
        )
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert len(dispatcher.calls) == 1
        assert report.status is CycleStatus.PROPOSED

    def test_max_targets_caps_the_work(self, store, hermes_repo, out):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "alpha", 0.2, samples=100),
                point(SKILL_SUCCESS_RATE, "beta", 0.3, samples=90),
                point(SKILL_SUCCESS_RATE, "gamma", 0.4, samples=80),
            ]
        )
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, max_targets=2, out=out)

        assert len(dispatcher.calls) == 2
        assert [d.target for d in report.dispatches] == ["alpha", "beta"]

    def test_an_empty_history_dispatches_nothing(self, store, hermes_repo, out):
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert report.status is CycleStatus.NO_TARGETS
        assert dispatcher.calls == []
        assert report.dispatches == []

    def test_an_advisory_only_ranking_dispatches_nothing(self, store, hermes_repo, out):
        store.extend([point(BENCHMARK_SCORE, "tblite", 0.4, samples=20)])
        dispatcher = FakeDispatcher()

        report = cycle(store, hermes_repo, dispatcher=dispatcher, out=out)

        assert report.status is CycleStatus.NO_TARGETS
        assert dispatcher.calls == []
        assert any("advisory only" in note for note in report.notes)
        assert report.ranked and report.ranked[0].target == "tblite"

    def test_the_report_serialises_to_json(self, store, hermes_repo, out):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        report = cycle(store, hermes_repo, out=out)
        blob = json.loads(json.dumps(report.to_dict()))
        assert blob["status"] == "proposed"
        assert blob["dispatches"][0]["target"] == "arxiv"


class TestRankedTable:
    """The table is what a human reads before approving a run, so the
    uncertainty behind each rank has to be visible in it."""

    def _series(self, target, values, samples=60):
        return [
            point(SKILL_SUCCESS_RATE, target, value, days_ago=30 - 5 * i, samples=samples)
            for i, value in enumerate(values)
        ]

    def test_a_clean_decline_shows_its_p_value_and_r_squared(
        self, store, hermes_repo, out
    ):
        store.extend(self._series("eroding", [0.91, 0.86, 0.78, 0.71, 0.62, 0.55]))
        cycle(store, hermes_repo, out=out)
        rendered = out.file.getvalue()
        assert "R²" in rendered
        assert "p=0.000" in rendered
        assert "(ns)" not in rendered

    def test_a_noisy_decline_is_marked_not_significant(self, store, hermes_repo, out):
        store.extend(
            self._series("noisy", [0.90, 0.35, 0.85, 0.30, 0.88, 0.33, 0.60])
        )
        cycle(store, hermes_repo, out=out)
        rendered = out.file.getvalue()
        assert "0.582" in rendered
        assert "(ns)" in rendered

    def test_an_explanation_is_printed_as_prose_not_markup(self, store, hermes_repo):
        # Rich reads a lowercase bracketed phrase as a style tag and drops the
        # whole bracket when the style is unknown, so both suffixes an
        # explanation can carry have to be escaped. A console wide enough that
        # nothing wraps, since a wrapped cell hides the phrase behind borders.
        wide = Console(file=io.StringIO(), width=400)
        store.extend([point(BENCHMARK_SCORE, "tblite", 0.4, samples=20)])
        store.extend(self._series("noisy", [0.90, 0.35, 0.85, 0.30, 0.88, 0.33, 0.60]))
        cycle(store, hermes_repo, out=wide)
        rendered = wide.file.getvalue()
        assert "advisory: no phase entry point for this target type" in rendered
        assert "trend p=0.582, R²=0.06" in rendered

    def test_a_target_with_no_fittable_trend_claims_nothing(
        self, store, hermes_repo, out
    ):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        cycle(store, hermes_repo, out=out)
        rendered = out.file.getvalue()
        assert "arxiv" in rendered
        assert "p=" not in rendered
        assert "(ns)" not in rendered


class TestScheduledChecks:
    def test_a_scored_benchmark_lands_in_history(self, store, hermes_repo, out):
        benchmarks = FakeBenchmarks({"tblite": 0.62})

        report = cycle(store, hermes_repo, benchmark_runner=benchmarks, out=out)

        recorded = store.query(metric=BENCHMARK_SCORE, target="tblite")
        assert len(recorded) == 1
        assert recorded[0].value == pytest.approx(0.62)
        assert recorded[0].timestamp == pytest.approx(T0)
        assert report.checks[0].recorded is True

    def test_a_benchmark_score_carries_its_task_count(self, store, hermes_repo, out):
        cycle(store, hermes_repo, benchmark_runner=FakeBenchmarks({"tblite": 0.62}), out=out)

        recorded = store.query(metric=BENCHMARK_SCORE, target="tblite")[0]
        assert recorded.samples == KNOWN_BENCHMARKS["tblite"].fast_task_count
        assert recorded.metadata["tasks"] == recorded.samples

    def test_an_absent_benchmark_records_nothing_at_all(self, store, hermes_repo, out):
        report = cycle(store, hermes_repo, benchmark_runner=FakeBenchmarks(), out=out)

        assert store.query(metric=BENCHMARK_SCORE) == []
        assert report.checks[0].status == GateStatus.UNAVAILABLE.value
        assert report.checks[0].recorded is False

    def test_the_previous_score_becomes_the_baseline(self, store, hermes_repo, out):
        store.extend([point(BENCHMARK_SCORE, "tblite", 0.70, days_ago=7)])
        benchmarks = FakeBenchmarks({"tblite": 0.68})

        cycle(store, hermes_repo, benchmark_runner=benchmarks, out=out)

        assert benchmarks.calls[0]["baseline"] == pytest.approx(0.70)
        assert benchmarks.calls[0]["fast"] is True

    def test_checks_do_not_run_without_a_checkout(self, store, out):
        benchmarks = FakeBenchmarks({"tblite": 0.62})

        cycle(store, None, benchmark_runner=benchmarks, out=out)

        assert benchmarks.calls == []


class TestDryRun:
    def test_nothing_runs_and_nothing_is_written(self, store, hermes_repo, out):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])
        before = store.path.read_bytes()

        report = cycle(
            store,
            hermes_repo,
            dry_run=True,
            dispatcher=exploding_dispatcher,
            benchmark_runner=exploding_benchmarks,
            out=out,
        )

        assert report.status is CycleStatus.DRY_RUN
        assert store.path.read_bytes() == before
        assert store.query(metric=OPTIMIZATION_RUN) == []

    def test_the_plan_still_shows_the_command_that_would_run(
        self, store, hermes_repo, out
    ):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])

        report = cycle(
            store,
            hermes_repo,
            dry_run=True,
            dispatcher=exploding_dispatcher,
            benchmark_runner=exploding_benchmarks,
            out=out,
        )

        dispatch = report.dispatches[0]
        assert dispatch.status is DispatchStatus.DRY_RUN
        assert dispatch.command[2:] == [
            "evolution.skills.evolve_skill",
            "--skill",
            "arxiv",
        ]
        assert "evolve_skill" in out.file.getvalue()

    def test_a_dry_run_still_reports_why_it_would_skip(self, store, hermes_repo, out):
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)])

        report = cycle(
            store,
            hermes_repo,
            dry_run=True,
            env={},
            dispatcher=exploding_dispatcher,
            benchmark_runner=exploding_benchmarks,
            out=out,
        )

        assert report.dispatches[0].status is DispatchStatus.SKIPPED
        assert "no API key" in report.dispatches[0].reason
        assert store.query(metric=OPTIMIZATION_RUN) == []

    def test_an_empty_history_dry_run_is_still_a_dry_run(self, store, hermes_repo, out):
        report = cycle(
            store,
            hermes_repo,
            dry_run=True,
            dispatcher=exploding_dispatcher,
            benchmark_runner=exploding_benchmarks,
            out=out,
        )
        assert report.status is CycleStatus.DRY_RUN


# ──────────────────────────────────────────────────────────────────────────
# Cron
# ──────────────────────────────────────────────────────────────────────────


class TestCronLine:
    def test_the_default_line_is_one_line_with_five_schedule_fields(self, tmp_path):
        line = cron_line(history_path=tmp_path / "metrics.jsonl", python="/usr/bin/python3")
        assert "\n" not in line
        assert line.split()[:5] == DEFAULT_SCHEDULE.split()
        assert len(DEFAULT_SCHEDULE.split()) == 5

    def test_it_runs_exactly_one_cycle(self, tmp_path):
        line = cron_line(history_path=tmp_path / "metrics.jsonl")
        assert "-m evolution.monitor.loop" in line
        assert "--once" in line
        assert str(tmp_path / "metrics.jsonl") in line

    def test_output_is_redirected_next_to_the_history(self, tmp_path):
        line = cron_line(history_path=tmp_path / "monitor" / "metrics.jsonl")
        assert line.endswith("2>&1")
        assert str(tmp_path / "monitor" / "loop.log") in line

    def test_a_custom_schedule_is_used_verbatim(self, tmp_path):
        line = cron_line(schedule="30 2 * * 0", history_path=tmp_path / "m.jsonl")
        assert line.startswith("30 2 * * 0 ")

    def test_an_invalid_schedule_is_refused(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            cron_line(schedule="every monday", history_path=tmp_path / "m.jsonl")
        assert "5 fields" in str(excinfo.value)

    def test_the_repo_and_thresholds_are_carried_through(self, tmp_path, hermes_repo):
        line = cron_line(
            history_path=tmp_path / "m.jsonl",
            hermes_repo=hermes_repo,
            threshold=0.25,
            max_targets=3,
        )
        assert f"--hermes-repo {hermes_repo}" in line
        assert "--threshold 0.25" in line
        assert "--max-targets 3" in line

    def test_paths_with_spaces_are_quoted(self, tmp_path):
        history = tmp_path / "my history" / "metrics.jsonl"
        line = cron_line(history_path=history, cwd=tmp_path / "my work")
        assert "'" in line
        assert "my history" in line

    def test_the_default_history_lives_under_the_output_dir(self, tmp_path):
        assert default_history_path(tmp_path) == tmp_path / "monitor" / "metrics.jsonl"
        assert default_history_path().parts[-2:] == ("monitor", "metrics.jsonl")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_emit_cron_prints_a_line_and_installs_nothing(self, tmp_path):
        result = CliRunner().invoke(
            main,
            ["--emit-cron", "--history-path", str(tmp_path / "m.jsonl")],
            env={"COLUMNS": "200"},
        )
        assert result.exit_code == 0
        assert "crontab -e" in result.output
        assert "--once" in result.output
        assert "Nothing was installed" in result.output

    def test_emit_cron_rejects_a_bad_schedule(self, tmp_path):
        result = CliRunner().invoke(
            main,
            [
                "--emit-cron",
                "--schedule",
                "weekly",
                "--history-path",
                str(tmp_path / "m.jsonl"),
            ],
            env={"COLUMNS": "200"},
        )
        assert result.exit_code == 2

    def test_no_flags_reports_status_without_writing(self, tmp_path):
        history = tmp_path / "m.jsonl"
        MetricStore(history, clock=lambda: T0).extend(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)]
        )
        before = history.read_bytes()

        result = CliRunner().invoke(
            main, ["--history-path", str(history)], env={"COLUMNS": "200"}
        )

        assert result.exit_code == 0
        assert "Triage" in result.output
        assert history.read_bytes() == before

    def test_status_on_an_empty_history_is_not_an_error(self, tmp_path):
        result = CliRunner().invoke(
            main,
            ["--history-path", str(tmp_path / "nothing-here.jsonl")],
            env={"COLUMNS": "200"},
        )
        assert result.exit_code == 0
        assert not (tmp_path / "nothing-here.jsonl").exists()

    def test_once_with_dry_run_writes_nothing(self, tmp_path, hermes_repo):
        history = tmp_path / "m.jsonl"
        MetricStore(history, clock=lambda: T0).extend(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=40)]
        )
        before = history.read_bytes()

        result = CliRunner().invoke(
            main,
            [
                "--once",
                "--dry-run",
                "--history-path",
                str(history),
                "--hermes-repo",
                str(hermes_repo),
            ],
            env={"COLUMNS": "200"},
        )

        assert result.exit_code == 0
        assert history.read_bytes() == before

    def test_the_help_text_documents_the_required_flags(self):
        result = CliRunner().invoke(main, ["--help"], env={"COLUMNS": "200"})
        for flag in (
            "--once",
            "--dry-run",
            "--hermes-repo",
            "--threshold",
            "--max-targets",
            "--emit-cron",
            "--history-path",
        ):
            assert flag in result.output


class TestProposedMeansABranchExists:
    """Exit code 0 is not a proposal.

    A phase exits 0 just as happily when the cross-tool guard rejected the
    candidate, a gate blocked it, or the rewrite came back identical. Reading
    that as PROPOSED told the operator a human had a PR waiting when no branch
    existed anywhere, and wrote a 1.0 into the metric history to match.
    """

    def _tool_store(self, store):
        store.extend([point(TOOL_SELECTION_ACCURACY, "search_files", 0.4, samples=80)])
        return store

    def test_a_branch_appearing_is_a_proposal(self, store, hermes_repo, out):
        self._tool_store(store)
        report = cycle(
            store, hermes_repo, out=out,
            branch_lister=FakeBranches("evolve/search_files-20260731_010203"),
        )
        dispatch = report.dispatches[0]
        assert dispatch.status is DispatchStatus.PROPOSED
        assert "evolve/search_files-20260731_010203" in dispatch.reason

    def test_a_clean_run_that_produced_nothing_is_not_a_proposal(self, store, hermes_repo, out):
        self._tool_store(store)
        report = cycle(
            store, hermes_repo, out=out,
            dispatcher=FakeDispatcher(returncode=0),
            branch_lister=FakeBranches(produces=""),
        )
        dispatch = report.dispatches[0]
        assert dispatch.returncode == 0
        assert dispatch.status is DispatchStatus.NO_CHANGE
        assert "no branch" in dispatch.reason
        assert report.proposed == []

    def test_a_failure_is_still_a_failure(self, store, hermes_repo, out):
        self._tool_store(store)
        report = cycle(
            store, hermes_repo, out=out,
            dispatcher=FakeDispatcher(returncode=1),
            branch_lister=FakeBranches(produces=""),
        )
        assert report.dispatches[0].status is DispatchStatus.FAILED

    def test_a_pre_existing_branch_is_not_mistaken_for_a_new_one(self, store, hermes_repo, out):
        self._tool_store(store)
        stale = {"evolve/something-old"}
        report = cycle(
            store, hermes_repo, out=out, branch_lister=lambda repo: set(stale)
        )
        assert report.dispatches[0].status is DispatchStatus.NO_CHANGE

    def test_a_non_git_checkout_does_not_crash_the_cycle(self, store, tmp_path, out):
        self._tool_store(store)
        from evolution.monitor.loop import _evolve_branches
        assert _evolve_branches(tmp_path) == set()
        assert _evolve_branches(None) == set()


class TestNoChangeIsNotRenderedAsAFailure:
    """The status has been distinct from FAILED since the branch comparison.

    The console line was not: a run that did everything right and found nothing
    deployable printed "failed ... exit 0", which is red and self-contradictory.
    """

    def _tool_store(self, store):
        store.extend(
            [
                point(TOOL_SELECTION_ACCURACY, "search_files", 0.4, days_ago=d, samples=80)
                for d in (30, 20, 10, 1)
            ]
        )
        return store

    def _no_change(self, store, hermes_repo, out):
        self._tool_store(store)
        return cycle(
            store, hermes_repo, out=out,
            dispatcher=FakeDispatcher(returncode=0),
            branch_lister=FakeBranches(produces=""),
        )

    def test_it_is_not_called_a_failure(self, store, hermes_repo, out):
        self._no_change(store, hermes_repo, out)
        assert "failed" not in out.file.getvalue()

    def test_it_says_no_change(self, store, hermes_repo, out):
        self._no_change(store, hermes_repo, out)
        assert "no change" in out.file.getvalue()

    def test_it_never_prints_a_failure_with_exit_zero(self, store, hermes_repo, out):
        self._no_change(store, hermes_repo, out)
        assert "exit 0" not in out.file.getvalue()

    def test_the_cycle_status_is_no_change_not_no_targets(self, store, hermes_repo, out):
        """A dispatched-and-declined cycle is not the same as an empty one."""
        report = self._no_change(store, hermes_repo, out)
        assert report.status is CycleStatus.NO_CHANGE
        assert "Nothing to do" not in out.file.getvalue()
        assert "found nothing deployable" in out.file.getvalue()

    def test_the_status_is_still_no_change(self, store, hermes_repo, out):
        report = self._no_change(store, hermes_repo, out)
        assert report.dispatches[0].status is DispatchStatus.NO_CHANGE
        assert report.proposed == []

    def test_a_real_failure_is_still_rendered_as_one(self, store, hermes_repo, out):
        self._tool_store(store)
        cycle(
            store, hermes_repo, out=out,
            dispatcher=FakeDispatcher(returncode=1),
            branch_lister=FakeBranches(produces=""),
        )
        rendered = out.file.getvalue()
        assert "failed" in rendered
        assert "exit 1" in rendered

    def test_a_proposal_is_still_rendered_as_one(self, store, hermes_repo, out):
        self._tool_store(store)
        cycle(
            store, hermes_repo, out=out,
            branch_lister=FakeBranches("evolve/search_files-20260731_010203"),
        )
        assert "proposed" in out.file.getvalue()
