"""Hermes integration adapter and compatibility matrix checks."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class CompatibilityCheck:
    """Single compatibility check row for a required Hermes component."""

    component: str
    relative_path: str
    required: bool
    exists: bool
    notes: str = ""


class HermesIntegrationAdapter:
    """Adapter that validates required hermes-agent integration surfaces."""

    REQUIRED_COMPONENTS: dict[str, tuple[str, bool]] = {
        "batch_runner": ("batch_runner.py", True),
        "trajectory": ("agent/trajectory.py", True),
        "session_db": ("hermes_state.py", True),
        "prompt_builder": ("agent/prompt_builder.py", True),
        "tool_registry": ("tools/registry.py", True),
        "tests": ("tests", True),
    }

    def __init__(self, hermes_repo: Path):
        self.hermes_repo = hermes_repo

    def detect_version(self) -> str:
        """Read the Hermes project version if available."""
        pyproject = self.hermes_repo / "pyproject.toml"
        if not pyproject.exists():
            return "unknown"

        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            return "unknown"

        project = data.get("project", {})
        version = project.get("version")
        return str(version) if version else "unknown"

    def compatibility_matrix(self) -> list[CompatibilityCheck]:
        """Build a compatibility matrix for all required components."""
        checks: list[CompatibilityCheck] = []
        for component, (relative_path, required) in self.REQUIRED_COMPONENTS.items():
            candidate = self.hermes_repo / relative_path
            checks.append(
                CompatibilityCheck(
                    component=component,
                    relative_path=relative_path,
                    required=required,
                    exists=candidate.exists(),
                ),
            )
        return checks

    def missing_required_components(self) -> list[CompatibilityCheck]:
        """Return required components that are missing from hermes-agent."""
        return [
            check
            for check in self.compatibility_matrix()
            if check.required and not check.exists
        ]

    def assert_compatible(self) -> None:
        """Raise with a readable message if required surfaces are missing."""
        missing = self.missing_required_components()
        if not missing:
            return

        formatted = ", ".join(
            f"{item.component} ({item.relative_path})"
            for item in missing
        )
        raise RuntimeError(
            "Hermes compatibility check failed. Missing required components: "
            f"{formatted}",
        )
