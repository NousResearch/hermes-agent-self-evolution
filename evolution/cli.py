"""The ``hermes-evolve`` command - one entry point for every phase.

Each phase already ships a standalone ``python -m evolution.<phase>....``
command, and those keep working. This groups them under a single verb so the
pipeline is discoverable without reading PLAN.md first:

    hermes-evolve status                 what this repo can see and gate on
    hermes-evolve skill  --skill NAME    Phase 1, skill files
    hermes-evolve tools                  Phase 2, tool descriptions
    hermes-evolve prompt --all-sections  Phase 3, system prompt sections
    hermes-evolve code   --tool NAME     Phase 4, tool implementation code
    hermes-evolve monitor --once         Phase 5, the continuous loop

Subcommands are resolved lazily. Importing every phase up front would drag
dspy into an invocation as trivial as ``--help``, which costs seconds; this way
you only pay for the phase you actually run.
"""

from __future__ import annotations

import importlib
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()

# subcommand name -> (module path, attribute, one-line help)
_SUBCOMMANDS: dict[str, tuple[str, str, str]] = {
    "skill": (
        "evolution.skills.evolve_skill",
        "main",
        "Phase 1: evolve a SKILL.md file",
    ),
    "tools": (
        "evolution.tools.evolve_tool_descriptions",
        "main",
        "Phase 2: evolve tool descriptions",
    ),
    "prompt": (
        "evolution.prompts.evolve_prompt_section",
        "main",
        "Phase 3: evolve system prompt sections",
    ),
    "code": (
        "evolution.code.evolve_tool_code",
        "main",
        "Phase 4: evolve tool implementation code",
    ),
    "monitor": (
        "evolution.monitor.loop",
        "main",
        "Phase 5: monitor, triage, and propose the next optimization",
    ),
}


class _LazyGroup(click.Group):
    """A group that imports a subcommand's module only when it is invoked."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Advertise the lazily loaded phase subcommands alongside the eager ones."""
        return sorted(set(super().list_commands(ctx)) | set(_SUBCOMMANDS))

    def get_command(self, ctx: click.Context, name: str) -> Optional[click.Command]:
        """Resolve a subcommand, importing its module only when it is asked for.

        The laziness is what lets ``hermes-evolve --help`` work without a
        hermes-agent checkout or every phase's optional dependencies installed.
        """
        existing = super().get_command(ctx, name)
        if existing is not None:
            return existing

        entry = _SUBCOMMANDS.get(name)
        if entry is None:
            return None

        module_path, attribute, _ = entry
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise click.ClickException(
                f"'{name}' could not be loaded: {exc}. "
                f"Install the package with: pip install -e \".[dev]\""
            ) from exc
        return getattr(module, attribute)

    def format_commands(self, ctx: click.Context, formatter) -> None:
        """Render the command list from the lazy table, which Click cannot see itself."""
        rows = [(name, _SUBCOMMANDS[name][2]) for name in sorted(_SUBCOMMANDS)]
        rows.append(("status", "Report what is discoverable and gateable"))
        with formatter.section("Commands"):
            formatter.write_dl(sorted(rows))


@click.group(cls=_LazyGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="hermes-agent-self-evolution", prog_name="hermes-evolve")
def cli() -> None:
    """Evolutionary self-improvement for Hermes Agent.

    Every phase reads from a hermes-agent checkout and writes proposals for a
    human to review. Nothing is ever merged or deployed automatically.
    """


@cli.command("status")
@click.option("--hermes-repo", default=None, help="Path to the hermes-agent repo")
def status(hermes_repo: Optional[str]) -> None:
    """Report what this installation can see, optimize, and gate on.

    Run this first. It answers the three questions that otherwise cost a failed
    optimization run to discover: is a hermes-agent checkout reachable, how many
    targets does each phase actually find in it, and which validation gates can
    really run.
    """
    from evolution.core.artifact_io import discover_prompt_sections, discover_tool_schemas
    from evolution.core.config import resolve_hermes_agent_path
    from evolution.core.gates import KNOWN_BENCHMARKS, find_benchmark
    from evolution.tools.tool_catalog import DEFAULT_MAX_TOOL_DESC

    console.print("\n[bold cyan]Hermes Agent Self-Evolution[/bold cyan] - status\n")

    try:
        repo = resolve_hermes_agent_path(hermes_repo)
    except FileNotFoundError as exc:
        console.print(f"  [red]✗ No hermes-agent repo found[/red]\n    {exc}")
        console.print(
            "\n  Set one with: [bold]export HERMES_AGENT_REPO=~/.hermes/hermes-agent[/bold]"
            "\n  or pass [bold]--hermes-repo PATH[/bold]\n"
        )
        raise SystemExit(1)

    if not repo.exists():
        console.print(f"  [red]✗ Path does not exist:[/red] {repo}\n")
        raise SystemExit(1)

    console.print(f"  Repo: {repo}\n")

    skills_dir = repo / "skills"
    skill_count = len(list(skills_dir.rglob("SKILL.md"))) if skills_dir.is_dir() else 0
    tools = discover_tool_schemas(repo)
    sections = discover_prompt_sections(repo)

    table = Table(title="Optimization targets")
    table.add_column("Phase", style="bold")
    table.add_column("Target")
    table.add_column("Found", justify="right")
    table.add_column("Source")

    table.add_row("1", "Skill files", str(skill_count), "skills/**/SKILL.md")
    table.add_row("2", "Tool descriptions", str(len(tools)), "tools/*.py schemas")
    table.add_row("3", "Prompt sections", str(len(sections)), "agent/prompt_builder.py")
    table.add_row(
        "4",
        "Tool code modules",
        str(len(list((repo / "tools").glob("*.py"))) if (repo / "tools").is_dir() else 0),
        "tools/*.py",
    )
    console.print(table)

    over = [t for t in tools if len(t.description) > DEFAULT_MAX_TOOL_DESC]
    if over:
        console.print(
            f"\n  [yellow]⚠ {len(over)} tool description(s) already exceed the "
            f"{DEFAULT_MAX_TOOL_DESC}-char budget before any evolution:[/yellow]"
        )
        for t in sorted(over, key=lambda x: -len(x.description))[:5]:
            console.print(f"    {t.tool_name}: {len(t.description)} chars")

    gate_table = Table(title="Validation gates")
    gate_table.add_column("Gate", style="bold")
    gate_table.add_column("Status")
    gate_table.add_column("Detail")

    has_tests = (repo / "tests").is_dir()
    gate_table.add_row(
        "pytest",
        "[green]available[/green]" if has_tests else "[red]missing[/red]",
        str(repo / "tests") if has_tests else "no tests/ directory",
    )
    for name in KNOWN_BENCHMARKS:
        location = find_benchmark(repo, name)
        gate_table.add_row(
            name,
            "[green]available[/green]" if location else "[dim]unavailable[/dim]",
            str(location) if location else f"set HERMES_BENCH_{name.upper()} to enable",
        )
    console.print()
    console.print(gate_table)

    if not any(find_benchmark(repo, n) for n in KNOWN_BENCHMARKS):
        console.print(
            "\n  [dim]No benchmarks resolved. Runs stay permissive by default;"
            "\n  pass --strict-gates to make a missing benchmark a hard failure.[/dim]"
        )
    console.print()


def main() -> None:
    """Console-script entry point for ``hermes-evolve``."""
    cli()


if __name__ == "__main__":
    main()
