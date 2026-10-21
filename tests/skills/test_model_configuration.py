"""Exercise model selection through the real CLI, DSPy and HTTP boundary."""

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest
from click.testing import CliRunner

from evolution.core.dataset_builder import EvalDataset, EvalExample

module = importlib.import_module("evolution.skills.evolve_skill")


@pytest.fixture
def provider(tmp_path, monkeypatch):
    requests = []
    # Each check must cross HTTP instead of reusing another test's saved response.
    monkeypatch.setattr(dspy, "cache", dspy.clients.cache.Cache(False, False, None))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            messages = json.dumps(body.get("messages", []))
            if "optimizer" in body["model"]:
                content = json.dumps({"new_instruction": "# Demo\nAnswer the task correctly, directly and concisely."})
            elif "correctness" in messages and "procedure_following" in messages:
                content = ('[[ ## reasoning ## ]]\nRubric checked.\n'
                           '[[ ## correctness ## ]]\n0.8\n'
                           '[[ ## procedure_following ## ]]\n0.6\n'
                           '[[ ## conciseness ## ]]\n1.0\n'
                           '[[ ## feedback ## ]]\nBe specific.\n[[ ## completed ## ]]')
            elif "test_cases" in messages:
                cases = [{"task_input": f"Task {i}", "expected_behavior": "Explain the result"}
                         for i in range(8)]
                content = ('[[ ## reasoning ## ]]\nDiverse cases.\n'
                           '[[ ## test_cases ## ]]\n' + json.dumps(cases) + '\n[[ ## completed ## ]]')
            else:
                content = '[[ ## reasoning ## ]]\nSolved.\n[[ ## output ## ]]\ndone\n[[ ## completed ## ]]'
            payload = json.dumps({
                "id": "local-completion", "object": "chat.completion", "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("EVOLUTION_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("EVOLUTION_API_KEY", "local-test-only")
    monkeypatch.setenv("EVOLUTION_OPTIMIZER_MODEL", "openai/optimizer-local")
    monkeypatch.setenv("EVOLUTION_EVAL_MODEL", "openai/evaluator-local")
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: Demo\n---\n\n# Demo\nAnswer the task accurately, directly and concisely.\n")
    example = EvalExample(task_input="Explain result", expected_behavior="Explain the result")
    dataset = tmp_path / "dataset"
    EvalDataset(train=[example], val=[example], holdout=[example]).save(dataset)
    args = ["--skill", "demo", "--hermes-repo", str(tmp_path),
            "--eval-source", "golden", "--dataset-path", str(dataset), "--iterations", "2"]
    try:
        yield requests, args, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_uses_selected_models_and_llm_judgment(provider):
    requests, args, root = provider
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
    metrics = json.loads(next((root / "output").rglob("metrics.json")).read_text())
    assert metrics["baseline_score"] == pytest.approx(0.78)
    assert metrics["eval_model"] == "openai/evaluator-local"
    assert any(r["model"] == "optimizer-local" for r in requests)
    assert any("correctness" in json.dumps(r) for r in requests)
    judging = [json.dumps(r) for r in requests if "correctness" in json.dumps(r)]
    assert any("correctly, directly" in r for r in judging)


def test_synthetic_generation_uses_selected_evaluation_model(provider, monkeypatch):
    requests, args, root = provider
    monkeypatch.setattr(dspy.GEPA, "compile", lambda self, student, **kw: student)
    index = args.index("--eval-source")
    args[index + 1] = "synthetic"
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
    generation = [r for r in requests if "test_cases" in json.dumps(r)]
    assert generation and all(r["model"] == "evaluator-local" for r in generation)
    assert len(EvalDataset.load(root / "datasets" / "skills" / "demo").all_examples) == 8


def test_missing_gepa_fallback_preserves_optimizer_model(provider, monkeypatch):
    requests, args, root = provider
    monkeypatch.delattr(dspy, "GEPA")
    models = []

    def compile(self, student, trainset, **kwargs):
        models.append((self.prompt_model.model, self.task_model.model))
        prediction = student(task_input=trainset[0].task_input)
        assert self.metric(trainset[0], prediction) == pytest.approx(0.78)
        return student

    monkeypatch.setattr(dspy.MIPROv2, "compile", compile)
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
    assert models == [("openai/optimizer-local", "openai/evaluator-local")]


def test_gepa_runtime_failure_does_not_switch_models(provider, monkeypatch):
    _, args, _ = provider

    def fail(*args, **kwargs):
        raise RuntimeError("provider authentication failed")

    monkeypatch.setattr(dspy.GEPA, "compile", fail)
    result = CliRunner().invoke(module.main, args)
    assert isinstance(result.exception, RuntimeError)
    assert "provider authentication failed" in str(result.exception)
    assert "falling back" not in result.output


def test_explicit_cli_models_override_environment(provider, monkeypatch):
    requests, args, root = provider
    # Keep the actual GEPA object and model construction; skip expensive search here.
    monkeypatch.setattr(dspy.GEPA, "compile", lambda self, student, **kw: student)
    result = CliRunner().invoke(module.main, args + ["--optimizer-model", "openai/optimizer-explicit",
                                                   "--eval-model", "openai/evaluator-explicit"])
    assert result.exit_code == 0, result.output
    metrics = json.loads(next((root / "output").rglob("metrics.json")).read_text())
    assert metrics["optimizer_model"] == "openai/optimizer-explicit"
    assert metrics["eval_model"] == "openai/evaluator-explicit"
    assert requests and all(r["model"] == "evaluator-explicit" for r in requests)


def test_oversized_baseline_does_not_outrank_valid_candidate(provider):
    requests, args, root = provider
    skill = root / "skills" / "demo" / "SKILL.md"
    skill.write_text("---\nname: demo\ndescription: Demo\n---\n\n" + "Keep answers accurate. " * 800)
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
    metrics_files = list((root / "output").rglob("metrics.json"))
    assert metrics_files, result.output
    metrics = json.loads(metrics_files[0].read_text())
    assert metrics["constraints_passed"]
    evolved = next((root / "output").rglob("evolved_skill.md")).read_text()
    assert len(evolved) < 15000
    reflection = [json.dumps(r) for r in requests if r["model"] == "optimizer-local"]
    assert any("15000" in r and "size" in r for r in reflection)


def test_rejected_final_candidate_returns_failure(provider, monkeypatch):
    _, args, root = provider
    (root / "skills" / "demo" / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\n\n" + "x" * 16000)
    monkeypatch.setattr(dspy.GEPA, "compile", lambda self, student, **kw: student)
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code != 0
    assert (root / "output" / "demo" / "evolved_FAILED.md").is_file()
    assert not list((root / "output").rglob("metrics.json"))


def test_reflection_reuses_trajectory_grade_but_not_another_rubric(provider, monkeypatch):
    requests, args, _ = provider

    def compile(self, student, trainset, valset):
        gold = valset[0]
        pred = student(task_input=gold.task_input)
        judgments = lambda: sum("procedure_following" in json.dumps(r["messages"]) for r in requests)
        score = self.metric_fn(gold, pred)
        count = judgments()
        feedback = self.metric_fn(gold, pred, pred_name="predictor.predict")
        assert feedback.score == score and feedback.feedback
        assert judgments() == count
        changed = dspy.Example(task_input=gold.task_input, expected_behavior="Another rubric")
        self.metric_fn(changed, pred)
        assert judgments() == count + 1
        return student

    monkeypatch.setattr(dspy.GEPA, "compile", compile)
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
