from evolution.core.dataset_builder import SyntheticDatasetBuilder


def test_normalize_case_keys_handles_alternate_keys():
    case = {
        "scenario": "Reviewing pre-push changes",
        "user_input": "review the code before pushing",
        "expected_output": [{"check": "scope of change"}],
    }
    normalized = SyntheticDatasetBuilder._normalize_case_keys(case)
    assert normalized["task_input"] == "review the code before pushing"
    assert "scope of change" in normalized["expected_behavior"]


def test_parse_test_cases_accepts_json_array():
    raw = '[{"task_input":"Do X","expected_behavior":"Do Y","difficulty":"easy","category":"c"}]'
    parsed = SyntheticDatasetBuilder._parse_test_cases(raw)
    assert len(parsed) == 1
    assert parsed[0]["task_input"] == "Do X"


def test_parse_test_cases_accepts_numbered_markdown():
    raw = """
1. **Pre-push review with SQL injection vulnerability**
   - **Input:** Review this PR for issues
   - **Expected Output:** Flag SQL injection risk and suggest parameterized query.
2. **Minor typo cleanup**
   - **Test Case:** Check docs-only typo changes
   - **Expected Behavior:** Confirm low risk and avoid blocking comments.
"""
    parsed = SyntheticDatasetBuilder._parse_test_cases(raw)
    assert len(parsed) == 2
    assert parsed[0]["task_input"] == "Review this PR for issues"
    assert "SQL injection risk" in parsed[0]["expected_behavior"]
