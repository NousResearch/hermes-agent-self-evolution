"""CLI: evolve a Python skill file using the local-first evolver.

Usage:
    python -m evolution.code.evolve_code --skill react_agent.py --topic "ReAct agent" --iterations 3
    python -m evolution.code.evolve_code --skill react_agent.py --llm http://127.0.0.1:11434/v1 --model qwen2.5-coder:14b
"""

import sys
from pathlib import Path

import click

from evolution.code.local_evolver import LocalCodeEvolver


@click.command()
@click.option("--skill",      required=True, help="Path to Python skill file to evolve.")
@click.option("--topic",      default="",    help="Natural-language topic (default: derived from filename).")
@click.option("--iterations", default=3,     show_default=True, help="Number of mutation rounds.")
@click.option("--llm",        default="http://127.0.0.1:11434/v1", show_default=True, help="OpenAI-compatible LLM base URL.")
@click.option("--model",      default="qwen2.5-coder:14b",          show_default=True, help="Model name.")
@click.option("--out",        default="",    help="Output path (default: overwrites input file).")
@click.option("--db",         default="",    help="SQLite path for variant history (optional).")
@click.option("--dry-run",    is_flag=True,  help="Score only, do not write output.")
def main(skill, topic, iterations, llm, model, out, db, dry_run):
    skill_path = Path(skill)
    if not skill_path.exists():
        click.echo(f"Error: {skill_path} not found", err=True)
        sys.exit(1)

    code = skill_path.read_text()
    if not topic:
        topic = skill_path.stem.replace("_", " ")

    evolver = LocalCodeEvolver(
        llm_base_url=llm,
        model=model,
        db_path=Path(db) if db else None,
    )

    click.echo(f"🧬 Local-first evolver  |  skill: {skill_path.name}  |  topic: {topic}  |  {iterations} iterations")

    best_code, best_score = evolver.evolve(
        skill_name=skill_path.stem,
        topic=topic,
        initial_code=code,
        iterations=iterations,
        verbose=True,
    )

    click.echo(f"\n✅ Best score: {best_score:.0%}")

    if dry_run:
        click.echo("(dry-run: not writing output)")
        return

    out_path = Path(out) if out else skill_path
    out_path.write_text(best_code)
    click.echo(f"Written → {out_path}")


if __name__ == "__main__":
    main()
