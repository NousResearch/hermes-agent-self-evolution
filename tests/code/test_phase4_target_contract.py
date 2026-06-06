"""Tests for Phase 4 code-evolution target contracts."""

from pathlib import Path

import pytest
import yaml


def _write_tool(path: Path, body: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        body
        or """
from tools.registry import registry


def safe_tool(value: str) -> str:
    if not value:
        raise ValueError("value is required")
    return value


registry.register(name="safe_tool", toolset="example", schema={"parameters": {}}, handler=safe_tool)
""".lstrip()
    )
    return path


def _write_spec(path: Path, repo: Path, files: list[str] | None = None, **overrides) -> Path:
    payload = {
        "phase": "4",
        "mode": "code-evolution-target",
        "target_id": "safe-tool-edge-case-001",
        "hermes_base": {
            "repo": str(repo),
            "base_ref": "test-base",
            "require_clean_worktree": True,
        },
        "allowed_mutation": {
            "files": files or ["tools/safe_tool.py"],
            "deny_globs": ["**/config*.py", "skills/**", "plugins/**", "memory/**"],
        },
        "freeze": {
            "function_signatures": True,
            "registry_register_calls": True,
            "public_cli_args": True,
        },
        "reproduction": {
            "failing_case_description": "empty value should stay rejected",
            "reproducer_command": "python -m pytest tests/tools/test_safe_tool.py -q",
        },
        "fitness": {
            "required_commands": [
                "python -m compileall -q tools/safe_tool.py",
                "python -m pytest tests/tools/test_safe_tool.py -q",
            ]
        },
        "benchmarks": {
            "full_benchmark_required_before_acceptance": True,
            "run_benchmarks_now": False,
        },
        "approvals": {
            "darwinian_install_approved": False,
            "darwinian_execution_approved": False,
            "hermes_source_mutation_approved": False,
            "budget_approved_usd": 0,
        },
    }
    payload.update(overrides)
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def test_target_contract_accepts_one_allowlisted_tool_file(tmp_path):
    from evolution.code.target_contract import load_target_spec

    hermes_repo = tmp_path / "hermes-agent"
    target_file = _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "target.yaml", hermes_repo)

    spec = load_target_spec(spec_path)

    assert spec.phase == "4"
    assert spec.mode == "code-evolution-target"
    assert spec.target_id == "safe-tool-edge-case-001"
    assert spec.hermes_repo == hermes_repo.resolve()
    assert spec.target_files == (target_file.resolve(),)
    assert spec.relative_target_files == ("tools/safe_tool.py",)
    assert spec.approvals["darwinian_install_approved"] is False
    assert spec.approvals["darwinian_execution_approved"] is False
    assert spec.approvals["hermes_source_mutation_approved"] is False
    assert spec.approvals["budget_approved_usd"] == 0
    assert spec.benchmarks["run_benchmarks_now"] is False


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (["../outside.py"], "target file must not contain traversal segments"),
        (["tools/safe_tool.py", "tools/other.py"], "exactly one target file"),
        (["skills/local_skill.py"], "denied target path"),
        (["tools/../skills/local_skill.py"], "target file must not contain traversal segments"),
        (["config.py"], "denied target path"),
        (["secret.py"], "denied target path"),
        (["token.txt"], "denied target path"),
        (["credentials.json"], "denied target path"),
    ],
)
def test_target_contract_rejects_unsafe_target_paths(tmp_path, files, message):
    from evolution.code.target_contract import TargetContractError, load_target_spec

    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    _write_tool(hermes_repo / "tools" / "other.py")
    _write_tool(hermes_repo / "skills" / "local_skill.py")
    _write_tool(hermes_repo / "config.py")
    (hermes_repo / "secret.py").write_text("VALUE = 'not-a-real-secret'\n")
    (hermes_repo / "token.txt").write_text("placeholder token fixture\n")
    (hermes_repo / "credentials.json").write_text("{}\n")
    spec_path = _write_spec(tmp_path / "target.yaml", hermes_repo, files=files)

    with pytest.raises(TargetContractError, match=message):
        load_target_spec(spec_path)


def test_target_contract_rejects_multiple_targets_even_when_flagged_allowed(tmp_path):
    from evolution.code.target_contract import TargetContractError, load_target_spec

    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    _write_tool(hermes_repo / "tools" / "other.py")
    spec_path = _write_spec(
        tmp_path / "target.yaml",
        hermes_repo,
        allowed_mutation={
            "files": ["tools/safe_tool.py", "tools/other.py"],
            "allow_multiple_targets": True,
            "deny_globs": ["**/config*.py", "skills/**", "plugins/**", "memory/**"],
        },
    )

    with pytest.raises(TargetContractError, match="exactly one target file"):
        load_target_spec(spec_path)


def test_target_contract_rejects_symlink_target(tmp_path):
    from evolution.code.target_contract import TargetContractError, load_target_spec

    hermes_repo = tmp_path / "hermes-agent"
    real_target = _write_tool(tmp_path / "real_tool.py")
    link_target = hermes_repo / "tools" / "safe_tool.py"
    link_target.parent.mkdir(parents=True, exist_ok=True)
    link_target.symlink_to(real_target)
    spec_path = _write_spec(tmp_path / "target.yaml", hermes_repo)

    with pytest.raises(TargetContractError, match="target file must not be a symlink"):
        load_target_spec(spec_path)


def test_target_contract_requires_clean_worktree_gate(tmp_path):
    from evolution.code.target_contract import TargetContractError, load_target_spec

    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")

    for hermes_base in [
        {"repo": str(hermes_repo), "base_ref": "test-base", "require_clean_worktree": False},
        {"repo": str(hermes_repo), "base_ref": "test-base"},
    ]:
        spec_path = _write_spec(tmp_path / "target.yaml", hermes_repo, hermes_base=hermes_base)
        with pytest.raises(TargetContractError, match="hermes_base.require_clean_worktree must be true"):
            load_target_spec(spec_path)


def test_target_contract_rejects_source_mutation_approvals_and_sensitive_text(tmp_path):
    from evolution.code.target_contract import TargetContractError, load_target_spec

    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(
        tmp_path / "target.yaml",
        hermes_repo,
        approvals={
            "darwinian_install_approved": False,
            "darwinian_execution_approved": False,
            "hermes_source_mutation_approved": True,
            "budget_approved_usd": 0,
        },
    )

    with pytest.raises(TargetContractError, match="hermes_source_mutation_approved must be false"):
        load_target_spec(spec_path)

    sensitive_value = "sk-" + "x" * 28
    spec_path.write_text(spec_path.read_text() + f"\nnotes: {sensitive_value}\n")
    with pytest.raises(TargetContractError, match="sensitive credential-like text"):
        load_target_spec(spec_path)
