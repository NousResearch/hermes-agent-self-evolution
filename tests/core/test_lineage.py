"""Lineage persistence from a GEPA detailed_results."""
import json
from types import SimpleNamespace

from evolution.core.lineage import LINEAGE_NAME, build_lineage, write_lineage


def _details(texts, parents, val_agg, val_sub=None, disc=None, best_idx=0):
    # candidates carry a .text the extractor reads.
    cands = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(
        candidates=cands, parents=parents, val_aggregate_scores=val_agg,
        val_subscores=val_sub, discovery_eval_counts=disc, best_idx=best_idx,
    )


def _extract(c):
    return c.text


def test_build_lineage_record_shape_and_deployed_flag():
    details = _details(
        texts=["seed", "v1", "v2"],
        parents=[None, [0], [1]],
        val_agg=[0.2, 0.5, 0.7],
        val_sub=[[0.0, 0.4], [0.5, 0.5], [0.8, 0.6]],
        disc=[0, 10, 25],
        best_idx=2,
    )
    lin = build_lineage(
        details, extract_text=_extract, deployed_idx=1,  # deployed != best (2)
        selection={"strategy": "knee:smallest", "picked_idx": 1},
        seed_text="seed", live_baseline_text="seed", suite_sha256="sha",
    )
    assert lin["deployed_idx"] == 1 and lin["best_idx"] == 2
    assert lin["n_candidates"] == 3
    recs = {r["idx"]: r for r in lin["candidates"]}
    assert recs[1]["is_deployed"] is True and recs[1]["is_best"] is False
    assert recs[2]["is_best"] is True and recs[2]["is_deployed"] is False
    assert recs[1]["parents"] == [0]
    assert recs[2]["val_subscores"] == [0.8, 0.6]
    assert recs[0]["discovery_eval_count"] == 0
    assert recs[1]["text"] == "v1"


def test_build_lineage_none_without_parents():
    # MIPROv2 fallback shape: no parents attribute.
    assert build_lineage(
        SimpleNamespace(candidates=[]), extract_text=_extract, deployed_idx=0,
        selection={}, seed_text="", live_baseline_text="",
    ) is None


def test_extractor_failure_degrades_to_none_text():
    details = _details(["a"], [None], [0.5], best_idx=0)
    def boom(_c):
        raise ValueError("bad candidate")
    lin = build_lineage(details, extract_text=boom, deployed_idx=0,
                        selection={}, seed_text="a", live_baseline_text="a")
    assert lin["candidates"][0]["text"] is None


def test_write_lineage_roundtrips(tmp_path):
    details = _details(["seed", "v1"], [None, [0]], [0.3, 0.6], best_idx=1)
    path = write_lineage(
        tmp_path, details, extract_text=_extract, deployed_idx=1,
        selection={"strategy": "val-best"}, seed_text="seed",
        live_baseline_text="seed", suite_sha256="x",
    )
    assert path == tmp_path / LINEAGE_NAME
    loaded = json.loads(path.read_text())
    assert loaded["deployed_idx"] == 1
    assert loaded["candidates"][1]["is_deployed"] is True


def test_write_lineage_skips_on_no_details(tmp_path):
    assert write_lineage(
        tmp_path, SimpleNamespace(), extract_text=_extract, deployed_idx=0,
        selection={}, seed_text="", live_baseline_text="",
    ) is None
    assert not (tmp_path / LINEAGE_NAME).exists()
