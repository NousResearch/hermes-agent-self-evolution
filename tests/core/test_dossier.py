"""Honest dossier rendering — diff + selection metadata, no attribution claims."""
from evolution.core.dossier import render_dossier, write_dossier


def _lineage(*, deployed_idx, best_idx, candidates, seed_text, live_baseline_text,
            selection=None, n=None):
    return {
        "schema_version": "1",
        "deployed_idx": deployed_idx,
        "best_idx": best_idx,
        "n_candidates": n if n is not None else len(candidates),
        "seed_text": seed_text,
        "live_baseline_text": live_baseline_text,
        "selection": selection or {},
        "suite_sha256": "sha",
        "candidates": candidates,
    }


def _cand(idx, text, parents, val_aggregate, *, disc=None, is_best=False, is_deployed=False):
    return {"idx": idx, "parents": parents, "val_aggregate": val_aggregate,
            "val_subscores": None, "discovery_eval_count": disc, "text": text,
            "is_best": is_best, "is_deployed": is_deployed}


def test_dossier_shows_deploy_diff_and_selection():
    lin = _lineage(
        deployed_idx=1, best_idx=1, seed_text="use skills.", live_baseline_text="use skills.",
        candidates=[
            _cand(0, "use skills.", None, 0.30),
            _cand(1, "use skills. patch stale ones.", [0], 0.70, disc=12, is_best=True, is_deployed=True),
        ],
        selection={"strategy": "val-best"},
    )
    out = render_dossier(lin)
    assert "Deploy diff" in out
    assert "patch stale ones." in out  # the added text appears in the diff
    assert "Δ +0.4000" in out  # val delta vs seed (0.70 - 0.30)
    assert "discovered after 12 metric calls" in out
    # Honesty guard: no per-hunk / per-task attribution language.
    low = out.lower()
    assert "attribut" not in low and "per-task" not in low and "evidence" not in low


def test_dossier_flags_deployed_not_best():
    lin = _lineage(
        deployed_idx=1, best_idx=2, seed_text="s", live_baseline_text="s",
        candidates=[
            _cand(0, "s", None, 0.2),
            _cand(1, "s v1", [0], 0.5, is_deployed=True),
            _cand(2, "s v2", [1], 0.7, is_best=True),
        ],
        selection={"strategy": "knee:smallest", "picked_idx": 1},
    )
    out = render_dossier(lin)
    assert "GEPA val-argmax was candidate 2" in out


def test_dossier_baseline_drift_bucket_only_when_live_differs_from_seed():
    drifted = _lineage(
        deployed_idx=0, best_idx=0, seed_text="seed text", live_baseline_text="DIFFERENT live",
        candidates=[_cand(0, "seed text", None, 0.5, is_best=True, is_deployed=True)],
    )
    assert "baseline drift" in render_dossier(drifted).lower()
    clean = _lineage(
        deployed_idx=0, best_idx=0, seed_text="same", live_baseline_text="same",
        candidates=[_cand(0, "same", None, 0.5, is_best=True, is_deployed=True)],
    )
    assert "baseline drift" not in render_dossier(clean).lower()


def test_dossier_counts_merge_steps():
    # deployed (3) ← merge of (1,2) ← seed (0)
    lin = _lineage(
        deployed_idx=3, best_idx=3, seed_text="s", live_baseline_text="s",
        candidates=[
            _cand(0, "s", None, 0.2),
            _cand(1, "s a", [0], 0.4),
            _cand(2, "s b", [0], 0.4),
            _cand(3, "s a b", [1, 2], 0.7, is_best=True, is_deployed=True),
        ],
    )
    out = render_dossier(lin)
    assert "merge step" in out.lower()


def test_write_dossier(tmp_path):
    lin = _lineage(
        deployed_idx=0, best_idx=0, seed_text="x", live_baseline_text="x",
        candidates=[_cand(0, "x", None, 0.5, is_best=True, is_deployed=True)],
    )
    path = write_dossier(tmp_path, lin)
    assert path == tmp_path / "dossier.md"
    assert "Evolution dossier" in path.read_text()
