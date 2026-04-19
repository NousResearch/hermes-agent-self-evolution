"""Tests for codex-batched prompt/protocol builders."""

from evolution.core.codex_protocol import build_evaluation_prompt, build_mutation_prompt


class TestCodexProtocol:
    def test_build_mutation_prompt_mentions_required_json_key(self):
        prompt = build_mutation_prompt(
            skill_name="obsidian",
            baseline_skill="baseline text",
            iterations=2,
        )

        assert "candidate_skill_markdown" in prompt
        assert "obsidian" in prompt
        assert "Iterations=2" in prompt
        assert "minimal" in prompt.lower()
        assert "20%" in prompt
        assert "do not rewrite the whole skill" in prompt.lower()
        assert "preserve structure" in prompt.lower()

    def test_build_evaluation_prompt_mentions_required_json_keys(self):
        prompt = build_evaluation_prompt(
            skill_name="obsidian",
            baseline_skill="baseline text",
            candidate_skill="candidate text",
            holdout_examples=[{"task_input": "a", "expected_behavior": "b"}],
        )

        assert "baseline_score" in prompt
        assert "candidate_score" in prompt
        assert "improvement" in prompt
        assert "recommendation" in prompt
        assert "winner" in prompt
        assert "reason" in prompt
        assert "confidence" in prompt
        assert '"task_input": "a"' in prompt
