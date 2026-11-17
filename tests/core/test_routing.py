import json

from evolution.core.routing import (
    PairwiseDescriptionMutator,
    RoutingDataset,
    RoutingExample,
    SkillCatalog,
    SkillCatalogEntry,
    acceptance_gate,
    build_neighbor_map,
    evaluate_routing,
    group_skill_families,
    routing_examples_from_events,
    validate_descriptions,
)


def make_catalog():
    return SkillCatalog(
        (
            SkillCatalogEntry("review", "Review code for defects", "software"),
            SkillCatalogEntry("debug", "Debug failing code", "software"),
        )
    )


def test_routing_dataset_roundtrip(tmp_path):
    path = tmp_path / "routing.jsonl"
    dataset = RoutingDataset(
        [RoutingExample("find a bug", "review", source="sessiondb", confidence="inferred"),
         RoutingExample("say hello", None)]
    )

    dataset.save(path)
    loaded = RoutingDataset.load(path)

    assert [example.to_dict() for example in loaded.examples] == [
        example.to_dict() for example in dataset.examples
    ]
    assert json.loads(path.read_text().splitlines()[0])["expected_skill"] == "review"


def test_evaluate_routing_reports_rejection_and_neighbor_confusion():
    catalog = make_catalog()
    dataset = RoutingDataset(
        [
            RoutingExample("find a bug", "review"),
            RoutingExample("fix a crash", "debug"),
            RoutingExample("tell me a joke", None),
        ]
    )

    def selector(query, _catalog):
        if "bug" in query:
            return "debug"
        if "crash" in query:
            return "debug"
        return None

    metrics = evaluate_routing(
        dataset,
        catalog,
        selector,
        neighboring_skills={"review": {"debug"}, "debug": {"review"}},
    )

    assert metrics.accuracy_at_1 == 2 / 3
    assert metrics.neighboring_false_positive_rate == 1 / 2
    assert metrics.rejection_rate == 1.0
    assert metrics.catalog_token_cost > 0


def test_acceptance_gate_uses_discovery_execution_product_and_cost():
    catalog = make_catalog()
    dataset = RoutingDataset([RoutingExample("find a bug", "review")])
    baseline = evaluate_routing(dataset, catalog, lambda _q, _c: "review")
    candidate = evaluate_routing(dataset, catalog, lambda _q, _c: "review")

    accepted = acceptance_gate(
        baseline,
        candidate,
        baseline_execution_success=0.9,
        candidate_execution_success=0.8,
    )
    rejected = acceptance_gate(
        baseline,
        candidate,
        baseline_execution_success=0.9,
        candidate_execution_success=1.0,
        max_cost_increase=-1,
    )

    assert not accepted.accepted
    assert accepted.reason == "discovery-times-execution regressed"
    assert not rejected.accepted
    assert rejected.reason == "catalog token cost exceeded budget"


def test_families_and_pairwise_prompt():
    catalog = make_catalog()
    families = group_skill_families(catalog)
    prompt = PairwiseDescriptionMutator().build_prompt(
        catalog.entries[0], catalog.entries[1], ["review this diff"], ["fix this traceback"]
    )

    assert set(families) == {"software"}
    assert [entry.name for entry in families["software"].entries] == ["review", "debug"]
    assert "anti-trigger" in prompt
    assert "review this diff" in prompt


def test_candidate_catalog_preserves_metadata_and_validates_descriptions():
    catalog = make_catalog()
    candidate = catalog.with_descriptions({"review": "Review code", "debug": "x" * 501})

    assert candidate.entries[0].path == catalog.entries[0].path
    assert build_neighbor_map(candidate)["review"] == {"debug"}
    assert validate_descriptions(candidate) == ["debug: description is 501 chars (max 500)"]


def test_session_signals_require_explicit_labels():
    examples = routing_examples_from_events(
        [
            {"query": "use the review skill", "manual_command": "/review"},
            {"query": "just answer this", "should_load_skill": False},
            {"query": "unclear request", "loaded_skill": "review"},
        ]
    )

    assert [example.expected_skill for example in examples] == ["review", None]
    assert examples[0].confidence == "inferred"
    assert examples[1].notes == "explicit no-skill signal"
