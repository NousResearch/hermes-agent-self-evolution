from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from evolution.code.evolve_tool_code import _split_argv, main
from evolution.code.verified_search import (
    CommandCheck,
    ExternalCLIProposer,
    GateTask,
    PatchCandidate,
    PublicSearchContext,
    VerifiedPatchSearch,
    compare_adaptive_frozen,
)


BASE = "def solve(x):\n    return 0\n"
LEVEL1 = "def solve(x):\n    return 1\n"
LEVEL1_LARGE = "def solve(x):\n    # equivalent visible behaviour\n    return 1\n"
LEVEL2 = "def solve(x):\n    return 2\n"


def _patch(before: str, after: str, path: str = "solver.py") -> str:
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


@pytest.fixture
def repair_fixture(tmp_path: Path):
    repo = tmp_path / "stable"
    repo.mkdir()
    (repo / "solver.py").write_text(BASE, encoding="utf-8")
    (repo / "visible_test.py").write_text(
        "from solver import solve\n"
        "assert isinstance(solve(3), int)\n"
        "assert solve(3) >= 0\n"
        "print('VISIBLE_OK')\n",
        encoding="utf-8",
    )
    (repo / "full_test.py").write_text(
        "from solver import solve\n"
        "assert isinstance(solve(5), int)\n"
        "assert 0 <= solve(5) <= 2\n",
        encoding="utf-8",
    )
    hidden = tmp_path / "sealed_check.py"
    hidden.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "ns = {}\n"
        "exec((Path(sys.argv[1]) / 'solver.py').read_text(encoding='utf-8'), ns)\n"
        "print(f\"SCORE={ns['solve'](5) / 2.0}\")\n",
        encoding="utf-8",
    )
    task = GateTask(
        task_id="progressive-solver-repair",
        visible_checks=(
            CommandCheck("visible", ("{python}", "visible_test.py")),
        ),
        sealed_checks=(
            CommandCheck(
                "sealed-score",
                ("{python}", str(hidden), "{repo}"),
                score_pattern=r"SCORE=([0-9.]+)",
                minimum=0.4,
            ),
        ),
        full_suite_check=CommandCheck(
            "full-suite", ("{python}", "full_test.py")
        ),
        allowed_paths=("solver.py",),
        max_changed_files=1,
        max_changed_lines=4,
    )
    return repo, task, tmp_path / "runs", hidden


class ProgressiveProposer:
    def __init__(self) -> None:
        self.contexts: list[PublicSearchContext] = []

    def propose(self, context: PublicSearchContext, budget: int):
        self.contexts.append(context)
        if "promote-one" in context.promoted_operators:
            return [
                PatchCandidate(
                    _patch(LEVEL1, LEVEL2),
                    operator="promote-two",
                    parent_id=context.parent_id,
                )
            ][:budget]
        return [
            PatchCandidate(_patch(BASE, LEVEL1), operator="promote-one"),
            PatchCandidate(_patch(BASE, LEVEL1_LARGE), operator="bloated-one"),
        ][:budget]


def _manifest(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_public_context_never_contains_sealed_commands(repair_fixture):
    repo, task, run_dir, hidden = repair_fixture
    proposer = ProgressiveProposer()
    report = VerifiedPatchSearch(repo, task, run_dir / "public-boundary").run(
        proposer, cycles=1, budget=2
    )
    serialized = json.dumps(proposer.contexts[0].as_dict(), sort_keys=True)
    assert str(hidden) not in serialized
    assert "sealed-score" not in serialized
    assert "full_test.py" not in serialized
    assert report.accepted_ids


def test_smallest_visible_equivalent_patch_is_the_only_one_admitted(repair_fixture):
    repo, task, run_dir, _ = repair_fixture
    report = VerifiedPatchSearch(repo, task, run_dir / "dedupe").run(
        ProgressiveProposer(), cycles=1, budget=2
    )
    expected = PatchCandidate(_patch(BASE, LEVEL1), operator="promote-one")
    assert report.accepted_ids == (expected.candidate_id,)
    events = _manifest(report.manifest_path)
    duplicates = [event for event in events if event.get("reason") == "visible_behavior_duplicate"]
    assert len(duplicates) == 1
    assert duplicates[0]["duplicate_of"] == expected.candidate_id
    accepted = next(event for event in events if event.get("reason") == "accepted")
    assert accepted["trace"] == [
        "copy_stable_tree",
        f"apply_candidate:{expected.candidate_id}",
        "run_visible_checks",
        "run_sealed_checks",
        "run_full_suite",
        "discard_workspace",
    ]
    assert accepted["trace_digest"]


def test_operator_promotion_feeds_the_next_generation_and_replays(repair_fixture):
    repo, task, run_dir, _ = repair_fixture
    proposer = ProgressiveProposer()
    report = VerifiedPatchSearch(repo, task, run_dir / "recursive").run(
        proposer, cycles=2, budget=2
    )
    assert report.initial_sealed_score == 0.0
    assert report.final_sealed_score == 1.0
    assert report.hidden_gain == 1.0
    assert report.promoted_operators == ("promote-one", "promote-two")
    assert len(report.accepted_ids) == 2
    assert proposer.contexts[1].promoted_operators == ("promote-one",)
    assert proposer.contexts[1].parent_id == report.accepted_ids[0]
    assert report.replay_verified
    assert (repo / "solver.py").read_text(encoding="utf-8") == BASE


def test_equal_budget_frozen_control_does_not_receive_promoted_operator(repair_fixture):
    repo, task, run_dir, _ = repair_fixture
    report = compare_adaptive_frozen(
        repo,
        task,
        run_dir / "counterfactual",
        ProgressiveProposer,
        cycles=2,
        budget=2,
    )
    assert report.equal_budget
    assert report.adaptive.proposal_budget == report.frozen.proposal_budget == 4
    assert report.adaptive.final_sealed_score == 1.0
    assert report.frozen.final_sealed_score == 0.0
    assert report.recursive_gain == 1.0
    assert report.frozen.accepted_ids == ()


def test_path_escape_is_rejected_without_touching_stable_repo(repair_fixture):
    repo, task, run_dir, _ = repair_fixture

    class EscapeProposer:
        def propose(self, context, budget):
            patch = _patch("x = 1\n", "x = 2\n", "../escape.py")
            return [PatchCandidate(patch, operator="escape")]

    report = VerifiedPatchSearch(repo, task, run_dir / "escape").run(
        EscapeProposer(), cycles=1, budget=1
    )
    events = _manifest(report.manifest_path)
    candidate = next(event for event in events if event.get("event") == "candidate")
    assert candidate["reason"].startswith("patch_rejected:ValueError")
    assert not (repo.parent / "escape.py").exists()
    assert (repo / "solver.py").read_text(encoding="utf-8") == BASE


def test_diff_header_cannot_hide_a_different_body_path(repair_fixture):
    repo, task, run_dir, _ = repair_fixture
    malicious = (
        "diff --git a/solver.py b/solver.py\n"
        "--- a/other.py\n"
        "+++ b/other.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )

    class MismatchProposer:
        def propose(self, context, budget):
            return [PatchCandidate(malicious, operator="header-confusion")]

    report = VerifiedPatchSearch(repo, task, run_dir / "header-confusion").run(
        MismatchProposer(), cycles=1, budget=1
    )
    events = _manifest(report.manifest_path)
    candidate = next(event for event in events if event.get("event") == "candidate")
    assert "body paths" in candidate["reason"]
    assert not (repo / "other.py").exists()


def test_visible_only_overfit_is_rejected_by_sealed_gate(repair_fixture):
    repo, task, run_dir, _ = repair_fixture
    overfit = "def solve(x):\n    return 1 if x == 3 else 0\n"

    class OverfitProposer:
        def propose(self, context, budget):
            return [PatchCandidate(_patch(BASE, overfit), operator="visible-overfit")]

    report = VerifiedPatchSearch(repo, task, run_dir / "overfit").run(
        OverfitProposer(), cycles=2, budget=1
    )
    events = _manifest(report.manifest_path)
    candidates = [event for event in events if event.get("event") == "candidate"]
    candidate = candidates[0]
    assert candidate["visible"][0]["passed"]
    assert candidate["reason"] == "sealed_gate_failed"
    assert candidates[1]["reason"] == "rejected_cached"
    assert report.accepted_ids == ()


def test_full_suite_is_a_hard_gate_after_sealed_improvement(repair_fixture):
    repo, task, run_dir, _ = repair_fixture
    regressive = "def solve(x):\n    return 3\n"

    class RegressiveProposer:
        def propose(self, context, budget):
            return [PatchCandidate(_patch(BASE, regressive), operator="regression")]

    report = VerifiedPatchSearch(repo, task, run_dir / "regression").run(
        RegressiveProposer(), cycles=1, budget=1
    )
    events = _manifest(report.manifest_path)
    candidate = next(event for event in events if event.get("event") == "candidate")
    assert candidate["sealed"][0]["score"] == 1.5
    assert not candidate["full_suite"]["passed"]
    assert candidate["reason"] == "full_suite_failed"
    assert report.accepted_ids == ()


def test_external_cli_proposer_receives_public_json_only(repair_fixture, tmp_path):
    repo, task, _, hidden = repair_fixture
    capture = tmp_path / "context.json"
    cwd_capture = tmp_path / "cwd.txt"
    script = tmp_path / "proposer.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"Path({str(capture)!r}).write_text(sys.stdin.read(), encoding='utf-8')\n"
        f"Path({str(cwd_capture)!r}).write_text(str(Path.cwd()), encoding='utf-8')\n"
        "print(json.dumps({'patch': " + repr(_patch(BASE, LEVEL1)) + ", "
        "'operator': 'external-repair'}))\n",
        encoding="utf-8",
    )
    context = PublicSearchContext(
        task=task.public_dict(),
        cycle=1,
        budget=1,
        seed=7,
        parent_id=None,
        promoted_operators=(),
        public_residues={},
        sealed_rejections=0,
    )
    proposals = ExternalCLIProposer((sys.executable, str(script))).propose(context, 1)
    captured = capture.read_text(encoding="utf-8")
    assert len(proposals) == 1
    assert proposals[0].operator == "external-repair"
    assert str(hidden) not in captured
    assert "sealed-score" not in captured
    assert Path(cwd_capture.read_text(encoding="utf-8")) != repo


def test_cli_command_parsing_and_end_to_end_run(repair_fixture, tmp_path):
    repo, _, run_dir, _ = repair_fixture
    hidden = tmp_path / "sealed binary.py"
    hidden.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "ns = {}\n"
        "exec((Path(sys.argv[1]) / 'solver.py').read_text(encoding='utf-8'), ns)\n"
        "raise SystemExit(0 if ns['solve'](5) >= 1 else 1)\n",
        encoding="utf-8",
    )
    proposer = tmp_path / "proposal source.py"
    proposer.write_text(
        "import json, sys\n"
        "json.loads(sys.stdin.read())\n"
        "print(json.dumps({'patch': " + repr(_patch(BASE, LEVEL1)) + ", "
        "'operator': 'cli-repair'}))\n",
        encoding="utf-8",
    )
    command = f'"{sys.executable}" "{proposer}"'
    assert _split_argv(command) == (sys.executable, str(proposer))
    result = CliRunner().invoke(
        main,
        [
            "--repo",
            str(repo),
            "--task-id",
            "cli-repair",
            "--proposer-command",
            command,
            "--visible",
            "visible::{python} visible_test.py",
            "--sealed",
            f'sealed::{{python}} "{hidden}" {{repo}}',
            "--full-suite",
            "suite::{python} full_test.py",
            "--allow",
            "solver.py",
            "--run-dir",
            str(run_dir / "cli"),
            "--cycles",
            "1",
            "--budget",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["accepted_ids"], result.output
    assert report["replay_verified"]


def test_run_directory_inside_stable_repo_is_rejected(repair_fixture):
    repo, task, _, _ = repair_fixture
    with pytest.raises(ValueError, match="outside"):
        VerifiedPatchSearch(repo, task, repo / ".phase4-run")
