"""``python -m evolution.monitor`` — the propose-only triage sentinel CLI.

Scans a target repo's recent git stream for repair candidates, ranks them
(dependency-regressions first), and writes a triage queue + report. With
``--attempt-top K`` it runs the validated repair loop on the top K candidates and
annotates the queue with the oracle-gate verdict — still propose-only (it never
opens a PR; a human reviews the queue and triggers any deploy).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from evolution.monitor.queue import build_queue, render_report, write_queue
from evolution.monitor.sentinel import scan

console = Console()


@click.command()
@click.option("--repo", "repo_root", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Target git repo to scan (e.g. the Hermes checkout).")
@click.option("--since-days", default=90, type=click.IntRange(min=1),
              help="Scan the fix-stream from this many days back (default 90).")
@click.option("--max-per-tool", default=5, type=click.IntRange(min=1),
              help="Cap candidates per tool when scanning (default 5).")
@click.option("--top", default=20, type=click.IntRange(min=1),
              help="How many ranked rows to show in the report (default 20).")
@click.option("--attempt-top", default=0, type=click.IntRange(min=0),
              help="Run the validated repair loop on the top K candidates and "
                   "annotate the queue (0 = scan/queue only; never opens a PR).")
@click.option("--max-cost-usd", default=None, type=click.FloatRange(min=0.0),
              help="Cost ceiling for --attempt-top.")
@click.option("--proposer-model", default=None)
@click.option("--base-python", default=None)
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, path_type=Path))
def main(repo_root, since_days, max_per_tool, top, attempt_top, max_cost_usd,
         proposer_model, base_python, output_dir):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / "monitor" / ts
    output_dir = Path(output_dir)

    console.print(f"[bold]monitor[/bold] — scanning [cyan]{repo_root}[/cyan] "
                  f"(last {since_days}d)…")
    candidates = scan(repo_root, since_days=since_days, max_per_tool=max_per_tool)
    payload = build_queue(candidates, repo=str(repo_root), since_days=since_days)

    if attempt_top:
        from evolution.monitor.attempt import attempt_candidates  # noqa: PLC0415
        attempt_candidates(repo_root, candidates[:attempt_top], payload,
                           max_cost_usd=max_cost_usd, proposer_model=proposer_model,
                           base_python=base_python, console=console)

    qpath = write_queue(output_dir, payload)
    (output_dir / "triage_report.md").write_text(render_report(payload, top=top))
    console.print(render_report(payload, top=top))
    console.print(f"\n  queue: [dim]{qpath}[/dim]")


if __name__ == "__main__":
    main()
