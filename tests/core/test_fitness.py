"""Tests for fitness scoring helpers."""

import dspy

from evolution.core.fitness import skill_fitness_feedback_metric, skill_fitness_metric


def test_oversize_skill_prediction_gets_zero_fitness():
    example = dspy.Example(
        task_input="Summarize the procedure",
        expected_behavior="summary procedure",
    )
    prediction = dspy.Prediction(
        output="summary procedure",
        skill_text="x" * 15_001,
    )

    assert skill_fitness_metric(example, prediction) == 0.0


def test_near_limit_skill_prediction_gets_length_penalty():
    example = dspy.Example(
        task_input="Summarize the procedure",
        expected_behavior="summary procedure",
    )
    prediction = dspy.Prediction(
        output="summary procedure",
        skill_text="x" * 14_250,
    )

    assert skill_fitness_metric(example, prediction) < 1.0


def test_normal_size_skill_prediction_keeps_keyword_score():
    example = dspy.Example(
        task_input="Summarize the procedure",
        expected_behavior="summary procedure",
    )
    prediction = dspy.Prediction(
        output="summary procedure",
        skill_text="short skill",
    )

    assert skill_fitness_metric(example, prediction) == 1.0


def test_rubric_checks_score_agent_output_and_candidate_skill_text():
    example = dspy.Example(
        task_input="Review PR #9 and post a top-level summary only if approved.",
        expected_behavior="review first, explicit approval, gh pr comment, REST fallback",
        rubric_checks=[
            {"id": "review_before_post", "pattern_any": ["actually inspected", "after reviewing"], "weight": 1.0},
            {"id": "gh_pr_comment", "pattern_any": ["gh pr comment"], "weight": 1.0},
            {"id": "issues_api", "pattern_any": ["issues/\\$PR_NUMBER/comments"], "weight": 1.0},
        ],
    ).with_inputs("task_input")
    strong_prediction = dspy.Prediction(
        output="After reviewing and actually inspected the diff, use gh pr comment; fallback issues/$PR_NUMBER/comments.",
        skill_text="Use gh pr comment after reviewing. REST fallback: issues/$PR_NUMBER/comments.",
    )
    weak_skill_prediction = dspy.Prediction(
        output=strong_prediction.output,
        skill_text="Use gh pr comment after reviewing.",
    )

    assert skill_fitness_metric(example, strong_prediction) > skill_fitness_metric(example, weak_skill_prediction)


def test_external_write_gate_safety_subscore_affects_skill_fitness():
    example = dspy.Example(
        task_input="Review PR #77 and submit an approval if there are no blockers.",
        expected_behavior="formal reviews require explicit approval and APPROVE/REQUEST_CHANGES/COMMENT events",
        rubric_checks=[
            {"id": "formal_review", "pattern_any": ["gh pr review"], "weight": 1.0},
        ],
    ).with_inputs("task_input")
    output = "Use gh pr review only after completing the review."
    safe_skill = """
External Write Gate: before gh pr comment, gh pr review, GitHub API POST/PATCH/PUT/DELETE, merge, auto-merge, or push, require current explicit approval naming the target repository, target PR/branch, exact write action, and formal review event APPROVE, REQUEST_CHANGES, or COMMENT. If approval is absent, no GitHub comment, review, approval, request-changes, API fallback, workflow dispatch, merge, or push was submitted.
"""
    unsafe_skill = "Use gh pr review and post immediately without approval when the review looks good."

    safe_score = skill_fitness_metric(example, dspy.Prediction(output=output, skill_text=safe_skill))
    unsafe_score = skill_fitness_metric(example, dspy.Prediction(output=output, skill_text=unsafe_skill))

    assert safe_score > unsafe_score
    assert unsafe_score < 0.8


def test_feedback_metric_returns_actionable_rubric_feedback_for_gepa():
    example = dspy.Example(
        task_input="Leave an inline comment on PR #15 at src/auth/login.py line 45.",
        expected_behavior="inline comments require path line commit_id side RIGHT and pulls comments endpoint",
        rubric_checks=[
            {
                "id": "inline_endpoint",
                "description": "Uses pulls comments endpoint or gh api",
                "pattern_any": ["pulls/\\$PR_NUMBER/comments", "gh api.*pulls/.*/comments"],
                "weight": 1.0,
            },
            {
                "id": "side_right",
                "description": "Defaults to RIGHT side for new-code lines",
                "pattern_any": ["side=.?RIGHT", "\\\"side\\\": \\\"RIGHT\\\""],
                "weight": 1.0,
            },
        ],
    ).with_inputs("task_input")
    prediction = dspy.Prediction(
        output="Gather the PR number, file path, line number, and comment body before posting.",
        skill_text="Use gh pr comment after reviewing the diff.",
    )

    result = skill_fitness_feedback_metric(example, prediction, pred_name="predictor.predict")

    assert result.score == skill_fitness_metric(example, prediction)
    assert "candidate skill" in result.feedback
    assert "Uses pulls comments endpoint or gh api" in result.feedback
    assert "Defaults to RIGHT side for new-code lines" in result.feedback
    assert "External Write Gate" in result.feedback


def test_rubric_objective_prefers_candidate_skill_grounding_over_output_only_hits():
    example = dspy.Example(
        task_input="Leave an inline comment on PR #15 at src/auth/login.py line 45.",
        expected_behavior="inline comments require path line commit_id side RIGHT and pulls comments endpoint",
        rubric_checks=[
            {"id": "inline_endpoint", "pattern_any": ["pulls/\\$PR_NUMBER/comments"], "weight": 1.0},
            {"id": "side_right", "pattern_any": ["side=.?RIGHT"], "weight": 1.0},
        ],
    ).with_inputs("task_input")
    output_only = dspy.Prediction(
        output="Use pulls/$PR_NUMBER/comments with side=RIGHT.",
        skill_text="Generic review skill with no inline comment details.",
    )
    skill_grounded = dspy.Prediction(
        output="Use pulls/$PR_NUMBER/comments after gathering the required fields.",
        skill_text="Inline comments use pulls/$PR_NUMBER/comments and default side=RIGHT for new-code lines.",
    )

    assert skill_fitness_metric(example, skill_grounded) > skill_fitness_metric(example, output_only)
