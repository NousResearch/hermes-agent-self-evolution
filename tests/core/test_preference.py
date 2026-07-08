"""Tests for the preference layer (the alpha-imprint critic signal).

Covers the deterministic core only — aggregation, relevance, recency decay, the
blend math, and the importers — none of which touch an LLM.
"""

import json

import pytest

from evolution.core.preference import (
    APPROVE,
    REJECT,
    PreferenceBook,
    PreferenceReference,
    PreferenceSignal,
    blend_preference,
    discover_preference_paths,
    lexical_alignment,
    load_preference_book,
    load_preference_file,
    normalize_valence,
    overlap,
)


class TestValenceAndOverlap:
    def test_valence_aliases(self):
        assert normalize_valence("up") == APPROVE
        assert normalize_valence("LIKED") == APPROVE
        assert normalize_valence("down") == REJECT
        assert normalize_valence("dislike") == REJECT
        assert normalize_valence("meh") is None
        assert normalize_valence(None) is None

    def test_overlap_jaccard_and_stopwords(self):
        # "the"/"a" are stopwords and do not count toward overlap.
        assert overlap("the concise summary", "a concise summary") == pytest.approx(1.0)
        assert overlap("apples and oranges", "spaceships and rockets") == 0.0
        assert overlap("", "anything") == 0.0

    def test_overlap_partial(self):
        # tokens: {concise, python, answer} vs {concise, rust, answer} -> 2/4
        assert overlap("concise python answer", "concise rust answer") == pytest.approx(0.5)


class TestSignal:
    def test_valence_normalized_and_weight_clamped(self):
        s = PreferenceSignal(response="hi", valence="up", weight=5.0)
        assert s.valence == APPROVE
        assert s.weight == 1.0

    def test_invalid_valence_raises(self):
        with pytest.raises(ValueError):
            PreferenceSignal(response="hi", valence="sideways")

    def test_key_dedupes_on_response_and_valence(self):
        a = PreferenceSignal(response="A tidy   answer", valence="up")
        b = PreferenceSignal(response="a tidy answer", valence="up")
        assert a.key() == b.key()


class TestBook:
    def test_add_dedup_keeps_most_recent(self):
        book = PreferenceBook()
        book.add(PreferenceSignal(response="same reply", valence="up", ts=100))
        book.add(PreferenceSignal(response="same reply", valence="up", ts=200))
        assert len(book) == 1
        assert book.signals[0].ts == 200

    def test_opposite_valence_is_a_distinct_signal(self):
        book = PreferenceBook()
        book.add(PreferenceSignal(response="same reply", valence="up"))
        book.add(PreferenceSignal(response="same reply", valence="down"))
        assert len(book) == 2

    def test_empty_book_reference_has_zero_weight(self):
        ref = PreferenceBook().reference_for("anything at all")
        assert ref.weight == 0.0
        assert ref.is_empty

    def test_reference_retrieves_relevant_examples(self):
        book = PreferenceBook([
            PreferenceSignal(response="use a concise bullet list for the release notes", valence="up"),
            PreferenceSignal(response="a long rambling wall of text about release notes", valence="down"),
            PreferenceSignal(response="totally unrelated cooking recipe", valence="up"),
        ])
        ref = book.reference_for("write the release notes")
        assert ref.weight > 0.0
        assert any("bullet list" in a for a in ref.approved)
        assert any("rambling" in r for r in ref.rejected)
        # The unrelated approval should not surface for this task.
        assert not any("recipe" in a for a in ref.approved)

    def test_min_overlap_filters_noise(self):
        book = PreferenceBook([PreferenceSignal(response="quantum chromodynamics lecture", valence="up")])
        ref = book.reference_for("how do I bake sourdough bread")
        assert ref.is_empty

    def test_recency_decay_lowers_weight(self):
        import time as _t

        now = _t.time()
        fresh = PreferenceBook([
            PreferenceSignal(response="concise release notes bullet list", valence="up", ts=now)
        ])
        stale = PreferenceBook([
            PreferenceSignal(
                response="concise release notes bullet list",
                valence="up",
                ts=now - 90 * 86_400,  # 3 half-lives at 30d
            )
        ])
        wf = fresh.reference_for("release notes", now=now).weight
        ws = stale.reference_for("release notes", now=now).weight
        assert wf > ws > 0.0

    def test_noisy_or_corroboration_raises_but_bounds_weight(self):
        # Several consistent signals should push weight up, never above 1.
        many = PreferenceBook([
            PreferenceSignal(response=f"concise release notes bullet list variant {i}", valence="up")
            for i in range(6)
        ])
        ref = many.reference_for("release notes", now=1_000_000_000)
        assert 0.0 < ref.weight <= 1.0

    def test_max_examples_caps_each_side(self):
        # Distinct trailing words so the signals do not de-duplicate (a lone
        # digit would be dropped as a 1-char token and collapse them).
        tags = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        book = PreferenceBook([
            PreferenceSignal(response=f"concise release notes reply {tag}", valence="up") for tag in tags
        ])
        ref = book.reference_for("release notes", max_examples=2)
        assert len(ref.approved) == 2


class TestBlendMath:
    def test_zero_weight_returns_base_unchanged(self):
        assert blend_preference(0.42, 0.0, weight=0.0, influence=0.35) == pytest.approx(0.42)

    def test_full_alignment_moves_score_up_within_cap(self):
        # base 0.5, perfect alignment, weight 1, influence 0.35 -> 0.5 + 0.35*0.5
        assert blend_preference(0.5, 1.0, weight=1.0, influence=0.35) == pytest.approx(0.675)

    def test_rejection_pulls_score_down_within_cap(self):
        # base 0.8, alignment 0 (reproduces rejected style), weight 1, influence 0.35
        assert blend_preference(0.8, 0.0, weight=1.0, influence=0.35) == pytest.approx(0.52)

    def test_neutral_alignment_does_not_move_score(self):
        # 0.5 is the neutral midpoint: a "no clear signal" verdict must not drag
        # a strong (or weak) rubric score toward the middle.
        assert blend_preference(0.9, 0.5, weight=1.0, influence=0.35) == pytest.approx(0.9)
        assert blend_preference(0.2, 0.5, weight=1.0, influence=0.35) == pytest.approx(0.2)

    def test_bounded_and_monotone(self):
        assert 0.0 <= blend_preference(1.0, 0.0, 1.0, 1.0) <= 1.0
        assert blend_preference(0.5, 1.0, 0.5, 0.35) > blend_preference(0.5, 1.0, 0.2, 0.35)


class TestLexicalAlignment:
    def test_neutral_without_reference(self):
        assert lexical_alignment("anything at all", PreferenceReference()) == 0.5

    def test_leans_toward_approved_and_away_from_rejected(self):
        ref = PreferenceReference(
            approved=["concise bullet list of release notes"],
            rejected=["long rambling paragraph about release notes"],
            weight=1.0,
        )
        approved_like = lexical_alignment("here is a concise bullet list of the release notes", ref)
        rejected_like = lexical_alignment("here is a long rambling paragraph about the release notes", ref)
        assert approved_like > 0.5
        assert rejected_like < 0.5
        assert 0.0 <= rejected_like and approved_like <= 1.0


class TestImporters:
    def test_imprint_format(self, tmp_path):
        p = tmp_path / "imprints.jsonl"
        p.write_text(
            json.dumps({"ts": 10, "valence": "up", "excerpt": "a crisp reply", "message_id": "m1"}) + "\n"
            + json.dumps({"ts": 11, "valence": "down", "excerpt": "a bloated reply", "message_id": "m2"}) + "\n",
            encoding="utf-8",
        )
        signals = load_preference_file(p)
        assert len(signals) == 2
        assert {s.valence for s in signals} == {APPROVE, REJECT}
        assert signals[0].source == "imprint"

    def test_generic_format_and_synonyms(self, tmp_path):
        p = tmp_path / "prefs.jsonl"
        p.write_text(
            json.dumps({"context": "q1", "response": "r1", "valence": "like"}) + "\n"
            + json.dumps({"context": "q2", "response": "r2", "valence": "negative"}) + "\n",
            encoding="utf-8",
        )
        signals = load_preference_file(p)
        assert [s.valence for s in signals] == [APPROVE, REJECT]
        assert signals[0].context == "q1"

    def test_malformed_and_unknown_rows_skipped(self, tmp_path):
        p = tmp_path / "imprints.jsonl"
        p.write_text(
            "not json\n"
            + json.dumps({"valence": "up", "excerpt": "kept"}) + "\n"
            + json.dumps({"valence": "banana", "excerpt": "dropped"}) + "\n"
            + json.dumps({"valence": "up", "excerpt": ""}) + "\n",
            encoding="utf-8",
        )
        signals = load_preference_file(p)
        assert len(signals) == 1
        assert signals[0].response == "kept"

    def test_load_book_empty_when_no_paths(self):
        assert load_preference_book(paths=[]).is_empty

    def test_discover_respects_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert discover_preference_paths() == []
        mem = tmp_path / "memories"
        mem.mkdir()
        (mem / "imprints.jsonl").write_text("{}\n", encoding="utf-8")
        assert discover_preference_paths() == [mem / "imprints.jsonl"]
