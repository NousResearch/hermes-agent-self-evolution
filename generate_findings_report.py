"""Render a synthesis "findings" report PDF from a self-contained prose YAML.

Unlike ``generate_report.py`` (which renders a per-run validation report and
extracts its numbers from a run dir's ``gate_decision.json``), a findings report
synthesizes results across many runs and experiments, so its numbers are authored
directly in the prose. To keep a stakeholder-facing PDF from drifting away from the
underlying data, the prose may carry a ``provenance`` block keyed to a source JSON;
this renderer **verifies those numbers against the source before rendering** and
refuses to emit a PDF if any disagree.

It reuses the visual language of the phase reports by importing the shared,
run-data-free primitives from ``generate_report`` (``_styles``, ``_title_page``,
``_footer``, ``_highlight_table``) — so a change to either renderer's shared helpers
is caught by both renderers' tests. The body is fully prose-driven: each section is
a list of blocks (``paragraph`` / ``subhead`` / ``bullets`` / ``table``), so the
synthesis structure lives in YAML rather than in hardcoded Python.

Usage:
    python generate_findings_report.py \\
        --prose reports/asymmetry_prose.yaml \\
        --out reports/asymmetry_report.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from generate_report import (
    DEFAULT_LOGO,
    REPO_ROOT,
    _footer,
    _highlight_table,
    _styles,
    _title_page,
)

# Letter (8.5") minus the 1" left/right margins build_findings_report sets below.
USABLE_WIDTH = 6.5 * inch


class ProvenanceError(ValueError):
    """A prose number does not match the source data it claims to come from."""


def _resolve(obj: Any, dotted: str) -> Any:
    """Walk a dotted path into nested dicts/lists, e.g. ``deploy_reachable.wilson.0``."""
    cur = obj
    for part in dotted.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def check_provenance(prose: dict, *, base_dir: Path = REPO_ROOT) -> list[dict]:
    """Verify each ``provenance.checks`` entry against the source JSON.

    Returns a list of mismatches (empty == all good). Numeric checks honor an
    optional per-check ``tol``; everything else is compared for exact equality.
    A prose with no ``provenance`` block is allowed and returns no failures.
    """
    prov = prose.get("provenance")
    if not prov:
        return []
    src = json.loads((Path(base_dir) / prov["source"]).read_text())
    failures: list[dict] = []
    for chk in prov["checks"]:
        actual = _resolve(src, chk["path"])
        expect = chk["expect"]
        tol = chk.get("tol")
        if tol is not None:
            ok = abs(float(actual) - float(expect)) <= float(tol)
        else:
            ok = actual == expect
        if not ok:
            failures.append({"path": chk["path"], "expect": expect, "actual": actual})
    return failures


def _table_block(block: dict, styles) -> Any:
    t = block["table"]
    ncol = len(t["header"])
    widths = ([w * inch for w in t["col_widths"]] if "col_widths" in t
              else [USABLE_WIDTH / ncol] * ncol)
    return _highlight_table(t["header"], t["rows"], widths, styles,
                            highlight_row=t.get("highlight_row"))


def _render_sections(sections: list[dict], styles) -> list:
    """Turn the prose ``sections`` into ReportLab flowables.

    Each block is a one-key dict naming its kind. Inline emphasis uses ReportLab's
    mini-HTML (``<b>`` / ``<i>``), not Markdown — the prose authors it directly.
    """
    flow: list = []
    for sec in sections:
        flow.append(Paragraph(sec["header"], styles['SectionHead']))
        for block in sec.get("blocks", []):
            if "paragraph" in block:
                flow.append(Paragraph(block["paragraph"], styles['BodyJust']))
            elif "subhead" in block:
                flow.append(Paragraph(block["subhead"], styles['SubSection']))
            elif "bullets" in block:
                for item in block["bullets"]:
                    flow.append(Paragraph(f"•&nbsp;&nbsp;{item}", styles['Metric']))
            elif "table" in block:
                flow.append(_table_block(block, styles))
                flow.append(Spacer(1, 0.12 * inch))
            elif "spacer" in block:
                flow.append(Spacer(1, float(block["spacer"]) * inch))
            else:
                raise ValueError(f"unknown prose block: {sorted(block)}")
    return flow


def build_findings_report(
    *,
    prose_path: Path,
    output_path: Path,
    logo_path: Path = DEFAULT_LOGO,
    base_dir: Path = REPO_ROOT,
    verify: bool = True,
) -> Path:
    """Render the findings PDF. Fails closed: if ``verify`` and any provenance
    number disagrees with its source JSON, raises :class:`ProvenanceError` and
    writes nothing."""
    prose = yaml.safe_load(Path(prose_path).read_text())
    if verify:
        failures = check_provenance(prose, base_dir=base_dir)
        if failures:
            raise ProvenanceError(
                "prose numbers do not match source data (fix the prose or the "
                f"provenance block): {failures}"
            )
    styles = _styles()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch,
    )
    flow = _title_page(prose, styles, Path(logo_path))
    flow += _render_sections(prose["sections"], styles)
    flow += _footer(prose, styles)
    doc.build(flow)
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prose", type=Path, default=REPO_ROOT / "reports" / "asymmetry_prose.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "asymmetry_report.pdf")
    parser.add_argument("--logo", type=Path, default=DEFAULT_LOGO)
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the provenance check (not recommended)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    path = build_findings_report(
        prose_path=args.prose, output_path=args.out,
        logo_path=args.logo, verify=not args.no_verify,
    )
    print(f"Findings report generated: {path}")
