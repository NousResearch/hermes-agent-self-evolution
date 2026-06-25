"""Tests for the synthesis findings-report renderer (generate_findings_report).

No LM, no run dir. Covers: a minimal prose renders to a real PDF (which also
catches a breaking change to the shared generate_report helpers); the provenance
check passes on matching numbers and fails on drift; the renderer fails closed
(raises and writes nothing) when a number drifts; and the actual banked artifact's
prose still matches its committed source JSON, with the headline numbers present in
the body so the prose can't silently diverge from verified provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from generate_findings_report import (
    ProvenanceError,
    build_findings_report,
    check_provenance,
)
from generate_report import REPO_ROOT

MINIMAL_PROSE = {
    "meta": {"title": "T", "subtitle": "S", "repository": "r"},
    "sections": [
        {"header": "Section", "blocks": [
            {"paragraph": "hello <b>world</b>"},
            {"subhead": "a subhead"},
            {"bullets": ["first", "second"]},
            {"table": {"header": ["x", "y"], "rows": [["1", "2"]], "highlight_row": 0}},
            {"spacer": 0.1},
        ]},
    ],
}


def _write_yaml(path: Path, obj: dict) -> Path:
    path.write_text(yaml.safe_dump(obj))
    return path


class TestRender:
    def test_minimal_prose_renders_pdf(self, tmp_path: Path):
        prose = _write_yaml(tmp_path / "p.yaml", MINIMAL_PROSE)
        out = tmp_path / "out.pdf"
        # Nonexistent logo exercises the no-logo title-page branch.
        build_findings_report(prose_path=prose, output_path=out,
                              logo_path=tmp_path / "nope.png", verify=False)
        data = out.read_bytes()
        assert data[:5] == b"%PDF-"
        assert len(data) > 1500  # a real multi-flowable document, not an empty shell

    def test_unknown_block_rejected(self, tmp_path: Path):
        bad = {"meta": MINIMAL_PROSE["meta"],
               "sections": [{"header": "H", "blocks": [{"bogus": 1}]}]}
        prose = _write_yaml(tmp_path / "p.yaml", bad)
        with pytest.raises(ValueError, match="unknown prose block"):
            build_findings_report(prose_path=prose, output_path=tmp_path / "o.pdf",
                                  logo_path=tmp_path / "nope.png", verify=False)


class TestProvenance:
    def _src(self, tmp_path: Path) -> None:
        (tmp_path / "data.json").write_text(json.dumps({
            "deploy_reachable": {"k": 12, "wilson": [0.38657, 0.78119]},
            "verdict": "GREEN",
        }))

    def _prose(self, k_expect: int) -> dict:
        return {
            "meta": MINIMAL_PROSE["meta"],
            "sections": MINIMAL_PROSE["sections"],
            "provenance": {"source": "data.json", "checks": [
                {"path": "deploy_reachable.k", "expect": k_expect},
                {"path": "deploy_reachable.wilson.0", "expect": 0.387, "tol": 0.01},
                {"path": "verdict", "expect": "GREEN"},
            ]},
        }

    def test_matching_numbers_pass(self, tmp_path: Path):
        self._src(tmp_path)
        assert check_provenance(self._prose(12), base_dir=tmp_path) == []

    def test_drift_is_reported(self, tmp_path: Path):
        self._src(tmp_path)
        failures = check_provenance(self._prose(99), base_dir=tmp_path)
        assert len(failures) == 1 and failures[0]["path"] == "deploy_reachable.k"

    def test_render_fails_closed_on_drift(self, tmp_path: Path):
        self._src(tmp_path)
        prose = _write_yaml(tmp_path / "p.yaml", self._prose(99))
        out = tmp_path / "out.pdf"
        with pytest.raises(ProvenanceError):
            build_findings_report(prose_path=prose, output_path=out,
                                  logo_path=tmp_path / "nope.png", base_dir=tmp_path)
        assert not out.exists()  # nothing written when provenance fails

    def test_no_provenance_block_is_allowed(self):
        assert check_provenance({"sections": []}) == []

    def test_per_check_source_override(self, tmp_path: Path):
        # A synthesis report pins numbers across several JSONs: the top-level source is
        # the default, and any check may name its own.
        (tmp_path / "a.json").write_text(json.dumps({"k": 1}))
        (tmp_path / "b.json").write_text(json.dumps({"k": 2}))
        prose = {"provenance": {"source": "a.json", "checks": [
            {"path": "k", "expect": 1},
            {"source": "b.json", "path": "k", "expect": 2},
        ]}}
        assert check_provenance(prose, base_dir=tmp_path) == []
        bad = {"provenance": {"source": "a.json", "checks": [
            {"source": "b.json", "path": "k", "expect": 99},
        ]}}
        fails = check_provenance(bad, base_dir=tmp_path)
        assert len(fails) == 1 and fails[0]["source"] == "b.json"

    def test_check_without_any_source_raises(self, tmp_path: Path):
        prose = {"provenance": {"checks": [{"path": "k", "expect": 1}]}}
        with pytest.raises(ValueError, match="no source"):
            check_provenance(prose, base_dir=tmp_path)


class TestBankedArtifact:
    """Guards the real committed findings report against drift from its source."""

    def test_prose_matches_committed_source(self):
        prose = yaml.safe_load((REPO_ROOT / "reports" / "asymmetry_prose.yaml").read_text())
        assert check_provenance(prose, base_dir=REPO_ROOT) == []

    def test_phase4_prose_matches_sources(self):
        prose = yaml.safe_load((REPO_ROOT / "reports" / "phase4_prose.yaml").read_text())
        assert check_provenance(prose, base_dir=REPO_ROOT) == []

    def test_phase5_prose_matches_sources(self):
        prose = yaml.safe_load((REPO_ROOT / "reports" / "phase5_prose.yaml").read_text())
        assert check_provenance(prose, base_dir=REPO_ROOT) == []

    def test_ledger_re_derives_deploy_reachable(self):
        # The committed per-organism manifest must reproduce the headline 12/20, so a
        # reader can audit which bugs deployed rather than trusting the aggregate.
        rows = [json.loads(line) for line
                in (REPO_ROOT / "reports" / "asymmetry_campaign_ledger.jsonl")
                .read_text().splitlines() if line.strip()]
        orgs = [r for r in rows if r.get("status") == "organism"]
        deploy_reachable = sum(1 for r in orgs if sum(bool(s) for s in r["seeds"]) >= 2)
        assert len(orgs) == 20
        assert deploy_reachable == 12

    def test_headline_numbers_appear_in_body(self):
        prose = yaml.safe_load((REPO_ROOT / "reports" / "asymmetry_prose.yaml").read_text())
        body = []
        for sec in prose["sections"]:
            for block in sec.get("blocks", []):
                if "paragraph" in block:
                    body.append(block["paragraph"])
                elif "subhead" in block:
                    body.append(block["subhead"])
                elif "bullets" in block:
                    body.extend(block["bullets"])
                elif "table" in block:
                    body.extend(str(c) for row in block["table"]["rows"] for c in row)
        text = " ".join(body)
        # Numeric headline tokens bound to the JSON-verified provenance, so the body
        # can't drift. ("GREEN" is verified via the provenance check, not the body —
        # the doc deliberately recasts it as "clears the futility floor".)
        for token in ("12 / 20", "0.60", "0.387", "0.781"):
            assert token in text, f"missing headline token in PDF body: {token!r}"
