"""Tests for Phase 4 candidate-vs-baseline freeze comparator."""

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "phase4-code-evolution" / "pytest-freeze-comparator"


def _tool_source(*, body: str = "return value", registry: str | None = None, cli_arg: str = "--dry-run") -> str:
    default_registry = """
registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": "string"}}, "required": ["value"]},
    handler=safe_tool,
)
"""
    registry_block = default_registry if registry is None else registry
    return f"""
from argparse import ArgumentParser
from tools.registry import registry


def safe_tool(value: str, *, strict: bool = True) -> str:
    if not value:
        raise ValueError("value is required")
    {body}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("{cli_arg}", action="store_true")
    return parser


class SafeToolHelper:
    def normalize(self, value: str) -> str:
        return value.strip()

    def validate(self, value: str, *, allow_empty: bool = False) -> bool:
        return allow_empty or bool(value)


{registry_block}
""".lstrip()


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_freeze_comparator_passes_body_only_changes_and_private_helpers(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source(body="return value"))
    candidate = _write(
        tmp_path / "candidate" / "safe_tool.py",
        _tool_source(body="return value.strip()") + "\n\ndef _candidate_helper(value: str) -> str:\n    return value\n",
    )

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["phase"] == "4"
    assert report["mode"] == "freeze-comparator"
    assert report["candidate_only"] is True
    assert report["apply_ready"] is False
    assert report["hermes_source_mutation_performed"] is False
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["comparisons"]["function_signatures"]["changed"] == []
    assert report["comparisons"]["registry_register_calls"]["changed"] == []
    assert report["comparisons"]["public_cli_args"]["changed"] == []


def test_freeze_comparator_rejects_public_function_signature_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate = _write(
        tmp_path / "candidate" / "safe_tool.py",
        _tool_source().replace("def safe_tool(value: str, *, strict: bool = True)", "def safe_tool(value: str, retries: int, *, strict: bool = True)"),
    )

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "function_signature_changed:safe_tool" in report["failed_checks"]
    changed = report["comparisons"]["function_signatures"]["changed"]
    assert changed[0]["name"] == "safe_tool"
    assert "retries" in changed[0]["candidate"]


def test_freeze_comparator_rejects_registry_register_schema_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate_registry = """
registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": "string"}, "mode": {"type": "string"}}, "required": ["value", "mode"]},
    handler=safe_tool,
)
"""
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source(registry=candidate_registry))

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "registry_register_changed:safe_tool" in report["failed_checks"]
    changed = report["comparisons"]["registry_register_calls"]["changed"]
    assert changed[0]["name"] == "safe_tool"
    assert "mode" in changed[0]["candidate"]["schema_parameters"]


def test_freeze_comparator_rejects_removed_registry_register_call(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source(registry=""))

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "registry_register_removed:safe_tool" in report["failed_checks"]


def test_freeze_comparator_rejects_public_cli_arg_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source(cli_arg="--dry-run"))
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source(cli_arg="--apply"))

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "public_cli_arg_removed:--dry-run" in report["failed_checks"]
    assert "public_cli_arg_added:--apply" in report["failed_checks"]


def test_freeze_comparator_rejects_symlink_baseline_or_candidate_inputs(tmp_path):
    from evolution.code.freeze_comparator import FreezeComparatorError, compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source())
    baseline_link = tmp_path / "baseline-link.py"
    candidate_link = tmp_path / "candidate-link.py"
    baseline_link.symlink_to(baseline)
    candidate_link.symlink_to(candidate)

    try:
        compare_candidate_to_baseline(baseline_file=baseline_link, candidate_file=candidate)
    except FreezeComparatorError as exc:
        assert "baseline-file must not be a symlink" in str(exc)
    else:
        raise AssertionError("expected symlink baseline-file rejection")

    try:
        compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate_link)
    except FreezeComparatorError as exc:
        assert "candidate-file must not be a symlink" in str(exc)
    else:
        raise AssertionError("expected symlink candidate-file rejection")


def test_freeze_comparator_rejects_registry_schema_detail_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate_registry = """
registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": "integer"}}, "required": ["value"]},
    handler=safe_tool,
)
"""
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source(registry=candidate_registry))

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "registry_register_changed:safe_tool" in report["failed_checks"]
    changed = report["comparisons"]["registry_register_calls"]["changed"]
    assert changed[0]["baseline"]["schema"]["parameters"]["value"]["type"] == "string"
    assert changed[0]["candidate"]["schema"]["parameters"]["value"]["type"] == "integer"


def test_freeze_comparator_rejects_public_cli_arg_keyword_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline_source = _tool_source().replace(
        'parser.add_argument("--dry-run", action="store_true")',
        'parser.add_argument("--limit", type=int, default=3, metavar="N")',
    )
    candidate_source = _tool_source().replace(
        'parser.add_argument("--dry-run", action="store_true")',
        'parser.add_argument("--limit", type=str, default=3, metavar="N")',
    )
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", baseline_source)
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", candidate_source)

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "public_cli_arg_changed:--limit" in report["failed_checks"]
    changed = report["comparisons"]["public_cli_args"]["changed"]
    assert changed[0]["baseline"]["options"]["type"] == "int"
    assert changed[0]["candidate"]["options"]["type"] == "str"


def test_freeze_comparator_rejects_public_class_add_remove(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate = _write(
        tmp_path / "candidate" / "safe_tool.py",
        _tool_source() + "\n\nclass NewPublicClass:\n    pass\n",
    )

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "class_name_added:NewPublicClass" in report["failed_checks"]


def test_freeze_comparator_rejects_public_decorator_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline_source = _tool_source().replace(
        "    def normalize(self, value: str) -> str:\n",
        "    @staticmethod\n    def normalize(self, value: str) -> str:\n",
    )
    candidate_source = _tool_source()
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", baseline_source)
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", candidate_source)

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "decorator_changed:SafeToolHelper.normalize" in report["failed_checks"]
    changed = report["comparisons"]["decorators"]["changed"]
    assert changed[0]["baseline"] == ["staticmethod"]
    assert changed[0]["candidate"] == []


def test_freeze_comparator_rejects_registry_expression_schema_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline_registry = """
registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": str.__name__}}, "required": ["value"]},
    handler=safe_tool,
)
"""
    candidate_registry = baseline_registry.replace("str.__name__", "int.__name__")
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source(registry=baseline_registry))
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source(registry=candidate_registry))

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "registry_register_changed:safe_tool" in report["failed_checks"]
    changed = report["comparisons"]["registry_register_calls"]["changed"]
    assert changed[0]["baseline"]["schema"]["parameters"]["value"]["type"] == "str.__name__"
    assert changed[0]["candidate"]["schema"]["parameters"]["value"]["type"] == "int.__name__"


def test_freeze_comparator_rejects_registry_kwargs_expansion_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline_registry = """
registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": "string"}}, "required": ["value"]},
    handler=safe_tool,
    **{"timeout": 1},
)
"""
    candidate_registry = baseline_registry.replace('**{"timeout": 1}', '**{"timeout": 2}')
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source(registry=baseline_registry))
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source(registry=candidate_registry))

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "registry_register_changed:safe_tool" in report["failed_checks"]
    changed = report["comparisons"]["registry_register_calls"]["changed"]
    assert changed[0]["baseline"]["kwargs_expansions"] == [{"timeout": 1}]
    assert changed[0]["candidate"]["kwargs_expansions"] == [{"timeout": 2}]


def test_freeze_comparator_rejects_public_cli_callable_keyword_change(tmp_path):
    from evolution.code.freeze_comparator import compare_candidate_to_baseline

    baseline_source = _tool_source().replace(
        'parser.add_argument("--dry-run", action="store_true")',
        'parser.add_argument("--value", type=resolve_type("int"))',
    )
    candidate_source = _tool_source().replace(
        'parser.add_argument("--dry-run", action="store_true")',
        'parser.add_argument("--value", type=resolve_type("str"))',
    )
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", baseline_source)
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", candidate_source)

    report = compare_candidate_to_baseline(baseline_file=baseline, candidate_file=candidate)

    assert report["passed"] is False
    assert "public_cli_arg_changed:--value" in report["failed_checks"]
    changed = report["comparisons"]["public_cli_args"]["changed"]
    assert changed[0]["baseline"]["options"]["type"] == "resolve_type('int')"
    assert changed[0]["candidate"]["options"]["type"] == "resolve_type('str')"


def test_freeze_comparator_cli_writes_report_under_phase4_output_root(tmp_path):
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source(body="return value.strip()"))
    output = OUTPUT_ROOT / "cli-pass" / "freeze_comparison_report.json"
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            "python",
            "-m",
            "evolution.code.freeze_comparator",
            "--baseline-file",
            str(baseline),
            "--candidate-file",
            str(candidate),
            "--output-json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["artifacts"]["output_json"] == str(output)


def test_freeze_comparator_cli_fails_visible_for_freeze_regression(tmp_path):
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate = _write(
        tmp_path / "candidate" / "safe_tool.py",
        _tool_source().replace("def safe_tool(value: str, *, strict: bool = True)", "def safe_tool(value: str, retries: int, *, strict: bool = True)"),
    )

    completed = subprocess.run(
        [
            "python",
            "-m",
            "evolution.code.freeze_comparator",
            "--baseline-file",
            str(baseline),
            "--candidate-file",
            str(candidate),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "function_signature_changed:safe_tool" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_freeze_comparator_cli_rejects_output_outside_phase4_root(tmp_path):
    baseline = _write(tmp_path / "baseline" / "safe_tool.py", _tool_source())
    candidate = _write(tmp_path / "candidate" / "safe_tool.py", _tool_source())

    completed = subprocess.run(
        [
            "python",
            "-m",
            "evolution.code.freeze_comparator",
            "--baseline-file",
            str(baseline),
            "--candidate-file",
            str(candidate),
            "--output-json",
            str(tmp_path / "outside.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "output-json must stay under output/phase4-code-evolution/" in completed.stderr
    assert "Traceback" not in completed.stderr
