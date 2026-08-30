"""Load the hermes-agent tool catalogue and account for every description byte.

Phase 2 evolves the natural-language text inside tool schemas. Before anything
can be evolved it has to be found, grouped the way the agent actually sees it,
and measured. That is this module's whole job.

Three things here are worth knowing:

1. **Toolsets come from the registration call, not the filename.** hermes-agent
   registers tools with ``registry.register(name=..., toolset="file", ...)`` at
   module import time. Importing those modules is out of the question - they pull
   in the whole agent - so the ``register`` calls are read out of the AST. When a
   tool has no discoverable registration the toolset is inferred from the module
   stem and flagged as inferred, never silently presented as fact.

2. **Budgets are reported, not enforced here.** Every description ships on every
   API call, so PLAN.md caps them at 500 chars for a tool and 200 for a
   parameter. The real ``read_file`` description is 539 chars and the real
   ``write_file.cross_profile`` parameter is 302, both already over budget before
   evolution starts. A loader that raised on those would be useless against the
   actual repo, so over-budget items are surfaced as findings and the optimizer
   decides what to do about them.

3. **Writes re-discover.** Source spans are only valid against the exact text
   they were parsed from, so :func:`write_bundle` re-parses after every single
   edit instead of trusting the spans held on a catalogue that was loaded
   minutes ago. Everything still routes through ``artifact_io`` so
   ``verify_structure_unchanged`` keeps parameter names, types, enums, defaults
   and required lists frozen.
"""

from __future__ import annotations

import ast
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from evolution.core.artifact_io import (
    ToolDescriptor,
    apply_param_description,
    apply_tool_description,
    discover_tool_schemas,
    schema_skeleton,
    verify_structure_unchanged,
)
from evolution.core.config import EvolutionConfig

__all__ = [
    "DEFAULT_MAX_TOOL_DESC",
    "DEFAULT_MAX_PARAM_DESC",
    "BudgetFinding",
    "ToolEntry",
    "ToolCatalog",
    "ToolDescriptions",
    "ToolsetIndex",
    "DescriptionChange",
    "WriteReport",
    "UnknownTool",
    "load_catalog",
    "discover_toolsets",
    "infer_toolset",
    "bundle_to_dict",
    "bundle_from_dict",
    "diff_bundles",
    "write_bundle",
]

# Mirrors EvolutionConfig, restated so the catalogue is usable without a config.
DEFAULT_MAX_TOOL_DESC = 500
DEFAULT_MAX_PARAM_DESC = 200


class UnknownTool(KeyError):
    """Raised when a caller names a tool the catalogue does not contain."""


# ──────────────────────────────────────────────────────────────────────────
# Toolset discovery
# ──────────────────────────────────────────────────────────────────────────

# Positional order of registry.register(). hermes-agent always calls it with
# keywords, but reading positionally costs nothing and survives a refactor.
_REGISTER_POSITIONAL = ("name", "toolset", "schema", "handler")


@dataclass(frozen=True)
class ToolsetIndex:
    """Which toolset each tool was registered under.

    Indexed two ways because a registration may be readable by tool name, by
    the schema constant it points at, or both.
    """

    by_tool: dict[str, str] = field(default_factory=dict)
    by_constant: dict[str, str] = field(default_factory=dict)

    def lookup(self, tool_name: str, constant: str) -> Optional[str]:
        """The module for a tool, found by tool name or by schema constant name."""
        return self.by_tool.get(tool_name) or self.by_constant.get(constant)

    def __bool__(self) -> bool:
        return bool(self.by_tool or self.by_constant)


def _register_calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "register":
            yield node


def _register_args(call: ast.Call) -> dict[str, ast.AST]:
    args: dict[str, ast.AST] = {}
    for position, node in enumerate(call.args):
        if position < len(_REGISTER_POSITIONAL):
            args[_REGISTER_POSITIONAL[position]] = node
    for keyword in call.keywords:
        if keyword.arg:
            args[keyword.arg] = keyword.value
    return args


def _literal_str(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def discover_toolsets(hermes_repo: Path) -> ToolsetIndex:
    """Read ``toolset=`` off every ``registry.register`` call under ``tools/``.

    Purely static. A registration whose name or toolset is computed rather than
    written as a literal is skipped, which leaves the caller to infer.
    """
    tools_dir = Path(hermes_repo) / "tools"
    by_tool: dict[str, str] = {}
    by_constant: dict[str, str] = {}
    if not tools_dir.is_dir():
        return ToolsetIndex(by_tool, by_constant)

    for path in sorted(tools_dir.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "register" not in source:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue

        for call in _register_calls(tree):
            args = _register_args(call)
            toolset = _literal_str(args.get("toolset"))
            if not toolset:
                continue
            tool_name = _literal_str(args.get("name"))
            if tool_name:
                by_tool.setdefault(tool_name, toolset)
            schema = args.get("schema")
            if isinstance(schema, ast.Name):
                by_constant.setdefault(schema.id, toolset)

    return ToolsetIndex(by_tool, by_constant)


def infer_toolset(module: str) -> str:
    """Best-effort toolset name for a module with no readable registration."""
    for suffix in ("_tools", "_tool"):
        if module.endswith(suffix):
            trimmed = module[: -len(suffix)]
            if trimmed:
                return trimmed
    return module


# ──────────────────────────────────────────────────────────────────────────
# Catalogue entries
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BudgetFinding:
    """One description that is larger than its char budget."""

    tool_name: str
    kind: str  # "tool" or "param"
    param: Optional[str]
    size: int
    budget: int

    @property
    def overage(self) -> int:
        """Characters over budget, negative when within it."""
        return self.size - self.budget

    @property
    def target(self) -> str:
        """The tool, or ``tool.param`` when the finding is on a parameter."""
        return self.tool_name if self.param is None else f"{self.tool_name}.{self.param}"

    def describe(self) -> str:
        """The target, its size against its budget, and the overage."""
        return (
            f"{self.target}: {self.size}/{self.budget} chars "
            f"({self.overage:+d} over budget)"
        )

    def to_dict(self) -> dict:
        """Serialise one budget finding."""
        return {
            "tool_name": self.tool_name,
            "kind": self.kind,
            "param": self.param,
            "size": self.size,
            "budget": self.budget,
            "overage": self.overage,
        }


@dataclass
class ToolDescriptions:
    """The evolvable text of one tool: its description and its parameters'."""

    tool_name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)

    def copy(self) -> "ToolDescriptions":
        """A copy whose parameter dict can be mutated without touching this one."""
        return ToolDescriptions(self.tool_name, self.description, dict(self.params))

    @property
    def total_chars(self) -> int:
        """Description plus every parameter description."""
        return len(self.description) + sum(len(v) for v in self.params.values())

    def to_dict(self) -> dict:
        """Serialise the editable text for this tool."""
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "ToolDescriptions":
        """Rebuild from a serialised bundle entry."""
        return cls(
            tool_name=blob["tool_name"],
            description=blob.get("description", ""),
            params=dict(blob.get("params", {})),
        )


@dataclass
class ToolEntry:
    """One tool schema, plus everything Phase 2 needs to reason about it."""

    descriptor: ToolDescriptor
    toolset: str
    toolset_source: str = "inferred"  # "registry" or "inferred"
    param_types: dict[str, str] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    max_tool_desc: int = DEFAULT_MAX_TOOL_DESC
    max_param_desc: int = DEFAULT_MAX_PARAM_DESC
    # Allowed values per parameter, for parameters that constrain them. Frozen
    # like the rest of the schema, and the only ground truth a factual-accuracy
    # check has for "the description offers a mode the tool does not have".
    param_enums: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Pass-throughs so callers do not reach into the descriptor for basics.
    @property
    def tool_name(self) -> str:
        """Registered name of the tool."""
        return self.descriptor.tool_name

    @property
    def constant(self) -> str:
        """Name of the module-level schema constant."""
        return self.descriptor.constant

    @property
    def path(self) -> Path:
        """File the schema is defined in."""
        return self.descriptor.path

    @property
    def module(self) -> str:
        """Importable module name for that file."""
        return self.descriptor.module

    @property
    def description(self) -> str:
        """The tool description text."""
        return self.descriptor.description

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names in declaration order."""
        return tuple(self.descriptor.params)

    @property
    def param_descriptions(self) -> dict[str, str]:
        """Each parameter description, keyed by parameter name."""
        return {name: p.description for name, p in self.descriptor.params.items()}

    # Sizes.
    @property
    def description_size(self) -> int:
        """Length of the tool description in characters."""
        return len(self.description)

    @property
    def param_sizes(self) -> dict[str, int]:
        """Length of each parameter description, keyed by name."""
        return {name: len(text) for name, text in self.param_descriptions.items()}

    @property
    def total_chars(self) -> int:
        """Description plus every parameter description."""
        return self.description_size + sum(self.param_sizes.values())

    @property
    def over_budget(self) -> bool:
        """True when anything on this tool exceeds its budget."""
        return bool(self.budget_findings())

    def budget_findings(self) -> list[BudgetFinding]:
        """Every description on this tool that busts its budget."""
        findings: list[BudgetFinding] = []
        if self.description_size > self.max_tool_desc:
            findings.append(
                BudgetFinding(
                    tool_name=self.tool_name,
                    kind="tool",
                    param=None,
                    size=self.description_size,
                    budget=self.max_tool_desc,
                )
            )
        for name, size in self.param_sizes.items():
            if size > self.max_param_desc:
                findings.append(
                    BudgetFinding(
                        tool_name=self.tool_name,
                        kind="param",
                        param=name,
                        size=size,
                        budget=self.max_param_desc,
                    )
                )
        return findings

    def descriptions(self) -> ToolDescriptions:
        """This entry's editable text: the tool description and its parameters."""
        return ToolDescriptions(
            tool_name=self.tool_name,
            description=self.description,
            params=self.param_descriptions,
        )

    def signature(self) -> str:
        """``read_file(path: string [required], offset: integer)``.

        The frozen half of the schema, rendered for a prompt. Parameter names
        and types never evolve, so showing them costs nothing and stops the
        optimizer from inventing arguments that do not exist.
        """
        bits = []
        for name in self.param_names:
            ptype = self.param_types.get(name) or "any"
            flag = " [required]" if name in self.required else ""
            bits.append(f"{name}: {ptype}{flag}")
        return f"{self.tool_name}({', '.join(bits)})"

    def to_dict(self) -> dict:
        """Serialise the catalogue entry and its measurements."""
        return {
            "tool_name": self.tool_name,
            "constant": self.constant,
            "module": self.module,
            "path": str(self.path),
            "toolset": self.toolset,
            "toolset_source": self.toolset_source,
            "description_size": self.description_size,
            "param_sizes": self.param_sizes,
            "required": list(self.required),
            "param_types": dict(self.param_types),
            "param_enums": {name: list(values) for name, values in self.param_enums.items()},
            "over_budget": [f.to_dict() for f in self.budget_findings()],
        }


@dataclass
class ToolCatalog:
    """Every literal tool schema in a hermes-agent checkout."""

    hermes_repo: Path
    entries: list[ToolEntry] = field(default_factory=list)
    max_tool_desc: int = DEFAULT_MAX_TOOL_DESC
    max_param_desc: int = DEFAULT_MAX_PARAM_DESC

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ToolEntry]:
        return iter(self.entries)

    def __contains__(self, tool_name: object) -> bool:
        return any(e.tool_name == tool_name for e in self.entries)

    @property
    def names(self) -> list[str]:
        """Every tool name, in catalogue order."""
        return [e.tool_name for e in self.entries]

    def get(self, tool_name: str) -> Optional[ToolEntry]:
        """The entry for *tool_name*, or None when it is not in the catalogue."""
        for entry in self.entries:
            if entry.tool_name == tool_name:
                return entry
        return None

    def require(self, tool_name: str) -> ToolEntry:
        """The entry for *tool_name*, or raise UnknownTool naming what does exist."""
        entry = self.get(tool_name)
        if entry is None:
            raise UnknownTool(
                f"no tool named {tool_name!r} in {self.hermes_repo} "
                f"(known: {', '.join(self.names) or 'none'})"
            )
        return entry

    def by_toolset(self) -> dict[str, list[ToolEntry]]:
        """Entries grouped by toolset. Keys sorted; entries keep catalogue order."""
        grouped: dict[str, list[ToolEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.toolset, []).append(entry)
        return {k: grouped[k] for k in sorted(grouped)}

    def by_module(self) -> dict[str, list[ToolEntry]]:
        """Entries grouped by defining module. Keys sorted; entries keep catalogue order."""
        grouped: dict[str, list[ToolEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.module, []).append(entry)
        return {k: grouped[k] for k in sorted(grouped)}

    def select(
        self,
        tools: Optional[Sequence[str]] = None,
        toolset: Optional[str] = None,
    ) -> "ToolCatalog":
        """A narrowed catalogue. Unknown tool names are an error, not a no-op."""
        wanted = list(tools or [])
        chosen = list(self.entries)

        if wanted:
            missing = [name for name in wanted if name not in self]
            if missing:
                raise UnknownTool(
                    f"unknown tool(s): {', '.join(sorted(missing))} "
                    f"(known: {', '.join(self.names) or 'none'})"
                )
            chosen = [e for e in chosen if e.tool_name in set(wanted)]

        if toolset:
            chosen = [e for e in chosen if e.toolset == toolset]

        return ToolCatalog(
            hermes_repo=self.hermes_repo,
            entries=chosen,
            max_tool_desc=self.max_tool_desc,
            max_param_desc=self.max_param_desc,
        )

    def budget_findings(self) -> list[BudgetFinding]:
        """Every over-budget description across the whole catalogue."""
        findings: list[BudgetFinding] = []
        for entry in self.entries:
            findings.extend(entry.budget_findings())
        return findings

    @property
    def total_description_chars(self) -> int:
        """Description characters across the whole catalogue."""
        return sum(e.total_chars for e in self.entries)

    def bundle(self) -> dict[str, ToolDescriptions]:
        """The evolvable text, keyed by tool name."""
        return {e.tool_name: e.descriptions() for e in self.entries}

    def to_dict(self) -> dict:
        """Serialise the catalogue, grouped by toolset."""
        return {
            "hermes_repo": str(self.hermes_repo),
            "tools": [e.to_dict() for e in self.entries],
            "toolsets": {k: [e.tool_name for e in v] for k, v in self.by_toolset().items()},
            "total_description_chars": self.total_description_chars,
            "over_budget": [f.to_dict() for f in self.budget_findings()],
        }


def _skeletons_for_module(path: Path) -> dict[str, dict]:
    """``{constant: skeleton entry}`` for one tools module, or empty on error."""
    try:
        return schema_skeleton(path.read_text(encoding="utf-8"))  # type: ignore[return-value]
    except (OSError, UnicodeDecodeError, SyntaxError):
        return {}


def load_catalog(
    hermes_repo: Path,
    config: Optional[EvolutionConfig] = None,
) -> ToolCatalog:
    """Build a :class:`ToolCatalog` from a hermes-agent checkout.

    Never raises on an over-budget description, a missing registration, or an
    unparseable module. A repo with no ``tools/`` directory yields an empty
    catalogue, which the caller reports rather than crashing on.
    """
    repo = Path(hermes_repo)
    max_tool = config.max_tool_desc_size if config else DEFAULT_MAX_TOOL_DESC
    max_param = config.max_param_desc_size if config else DEFAULT_MAX_PARAM_DESC

    toolsets = discover_toolsets(repo)
    skeleton_cache: dict[Path, dict[str, dict]] = {}
    entries: list[ToolEntry] = []

    for descriptor in discover_tool_schemas(repo):
        if descriptor.path not in skeleton_cache:
            skeleton_cache[descriptor.path] = _skeletons_for_module(descriptor.path)
        skeleton = skeleton_cache[descriptor.path].get(descriptor.constant, {})

        properties = skeleton.get("properties") or {}
        param_types = {
            name: str(shape.get("type", "any"))
            for name, shape in properties.items()
            if isinstance(shape, dict)
        }
        param_enums = {
            name: tuple(str(value) for value in shape["enum"])
            for name, shape in properties.items()
            if isinstance(shape, dict) and isinstance(shape.get("enum"), list)
        }
        raw_required = skeleton.get("required")
        required = tuple(raw_required) if isinstance(raw_required, list) else ()

        registered = toolsets.lookup(descriptor.tool_name, descriptor.constant)
        entries.append(
            ToolEntry(
                descriptor=descriptor,
                toolset=registered or infer_toolset(descriptor.module),
                toolset_source="registry" if registered else "inferred",
                param_types=param_types,
                required=required,
                max_tool_desc=max_tool,
                max_param_desc=max_param,
                param_enums=param_enums,
            )
        )

    return ToolCatalog(
        hermes_repo=repo,
        entries=entries,
        max_tool_desc=max_tool,
        max_param_desc=max_param,
    )


# ──────────────────────────────────────────────────────────────────────────
# Bundles
# ──────────────────────────────────────────────────────────────────────────


def bundle_to_dict(bundle: dict[str, ToolDescriptions]) -> dict:
    """Serialise a name-to-descriptions bundle for JSON."""
    return {name: descriptions.to_dict() for name, descriptions in bundle.items()}


def bundle_from_dict(blob: dict) -> dict[str, ToolDescriptions]:
    """Rebuild a bundle from its serialised form."""
    return {name: ToolDescriptions.from_dict(entry) for name, entry in blob.items()}


@dataclass(frozen=True)
class DescriptionChange:
    """One description that differs between two bundles."""

    tool_name: str
    kind: str  # "tool" or "param"
    param: Optional[str]
    before: str
    after: str

    @property
    def target(self) -> str:
        """The tool, or ``tool.param`` when the change is on a parameter."""
        return self.tool_name if self.param is None else f"{self.tool_name}.{self.param}"

    @property
    def delta_chars(self) -> int:
        """Characters gained or lost by this change."""
        return len(self.after) - len(self.before)

    def to_dict(self) -> dict:
        """Serialise one description change, before and after."""
        return {
            "tool_name": self.tool_name,
            "kind": self.kind,
            "param": self.param,
            "before": self.before,
            "after": self.after,
            "delta_chars": self.delta_chars,
        }


def diff_bundles(
    baseline: dict[str, ToolDescriptions],
    candidate: dict[str, ToolDescriptions],
) -> list[DescriptionChange]:
    """Every description that actually moved, in stable order.

    Tools present only in the candidate are ignored: this phase rewrites text
    in place and never invents a tool.
    """
    changes: list[DescriptionChange] = []
    for tool_name in sorted(baseline):
        before = baseline[tool_name]
        after = candidate.get(tool_name)
        if after is None:
            continue
        if after.description != before.description:
            changes.append(
                DescriptionChange(
                    tool_name=tool_name,
                    kind="tool",
                    param=None,
                    before=before.description,
                    after=after.description,
                )
            )
        for param in sorted(before.params):
            new_text = after.params.get(param)
            if new_text is not None and new_text != before.params[param]:
                changes.append(
                    DescriptionChange(
                        tool_name=tool_name,
                        kind="param",
                        param=param,
                        before=before.params[param],
                        after=new_text,
                    )
                )
    return changes


# ──────────────────────────────────────────────────────────────────────────
# Write-back
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class WriteReport:
    """What a write-back did, or would have done under ``dry_run``."""

    dry_run: bool = False
    changes: list[DescriptionChange] = field(default_factory=list)
    files_written: list[Path] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> int:
        """How many descriptions were changed."""
        return len(self.changes)

    def summary(self) -> str:
        """One line naming how many descriptions changed and in how many files."""
        verb = "would write" if self.dry_run else "wrote"
        return (
            f"{verb} {self.changed} description(s) across "
            f"{len(self.files_written)} file(s)"
        )

    def to_dict(self) -> dict:
        """Serialise the write result, skips and reasons included."""
        return {
            "dry_run": self.dry_run,
            "changes": [c.to_dict() for c in self.changes],
            "files_written": [str(p) for p in self.files_written],
            "skipped": [{"target": t, "reason": r} for t, r in self.skipped],
        }


def _descriptors_for_source(source: str, module: str) -> dict[str, ToolDescriptor]:
    """Re-discover descriptors against an in-memory source string.

    ``artifact_io`` discovers from a repo on disk, and its spans are only valid
    against the exact text they were parsed from. Staging the intermediate text
    in a throwaway directory is what lets a sequence of edits to one file stay
    span-exact without touching the real repo, and it is what makes ``dry_run``
    able to verify a full multi-edit rewrite it never commits.

    The re-parse per edit is the contract, not overhead to hoist: every span
    is only valid against the exact text it was parsed from. The directory
    around it costs a millisecond or two against a parse several times that,
    inside a run whose optimizer step is measured in minutes.
    """
    with tempfile.TemporaryDirectory(prefix="hase-stage-") as tmp:
        staged_tools = Path(tmp) / "tools"
        staged_tools.mkdir()
        (staged_tools / f"{module}.py").write_text(source, encoding="utf-8")
        return {d.tool_name: d for d in discover_tool_schemas(Path(tmp))}


def write_bundle(
    hermes_repo: Path,
    bundle: dict[str, ToolDescriptions],
    dry_run: bool = False,
    baseline: Optional[dict[str, ToolDescriptions]] = None,
) -> WriteReport:
    """Write evolved descriptions back into a hermes-agent checkout.

    Every edit goes through ``artifact_io``, so a rewrite that would move a
    parameter name, type, enum, default or required list raises
    ``artifact_io.StructureViolation`` and nothing reaches disk. The whole file is
    verified once more after the last edit before it is written.

    ``dry_run`` runs the identical rewrite and verification path and then throws
    the result away, so a dry run that reports success is real evidence the
    write would succeed.
    """
    repo = Path(hermes_repo)
    catalog = load_catalog(repo)
    baseline = baseline or catalog.bundle()
    report = WriteReport(dry_run=dry_run)

    # Group the requested edits by the file that owns them.
    per_file: dict[Path, list[str]] = {}
    for tool_name in bundle:
        entry = catalog.get(tool_name)
        if entry is None:
            report.skipped.append((tool_name, "not present in this hermes-agent checkout"))
            continue
        per_file.setdefault(entry.path, []).append(tool_name)

    for path, tool_names in sorted(per_file.items(), key=lambda kv: str(kv[0])):
        original = path.read_text(encoding="utf-8")
        source = original
        module = path.stem

        for tool_name in sorted(tool_names):
            desired = bundle[tool_name]
            before = baseline.get(tool_name)

            if before is None or desired.description != before.description:
                descriptor = _descriptors_for_source(source, module).get(tool_name)
                if descriptor is None:
                    report.skipped.append((tool_name, "schema vanished during rewrite"))
                    continue
                if descriptor.description != desired.description:
                    source = apply_tool_description(source, descriptor, desired.description)
                    report.changes.append(
                        DescriptionChange(
                            tool_name=tool_name,
                            kind="tool",
                            param=None,
                            before=descriptor.description,
                            after=desired.description,
                        )
                    )

            for param in sorted(desired.params):
                new_text = desired.params[param]
                if before is not None and before.params.get(param) == new_text:
                    continue
                descriptor = _descriptors_for_source(source, module).get(tool_name)
                if descriptor is None:
                    report.skipped.append(
                        (f"{tool_name}.{param}", "schema vanished during rewrite")
                    )
                    continue
                existing = descriptor.params.get(param)
                if existing is None:
                    report.skipped.append(
                        (f"{tool_name}.{param}", "no such parameter in the schema")
                    )
                    continue
                if existing.description == new_text:
                    continue
                source = apply_param_description(source, descriptor, param, new_text)
                report.changes.append(
                    DescriptionChange(
                        tool_name=tool_name,
                        kind="param",
                        param=param,
                        before=existing.description,
                        after=new_text,
                    )
                )

        if source == original:
            continue

        # Belt and braces: artifact_io verified each individual edit, this
        # checks the accumulated result of all of them.
        verify_structure_unchanged(original, source)

        if not dry_run:
            path.write_text(source, encoding="utf-8")
        report.files_written.append(path)

    return report
