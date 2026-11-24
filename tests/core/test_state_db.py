"""Direct tests for the shared state.db access facts."""

from evolution.core.state_db import clean_user_text, is_task_like


class TestCleanUserText:
    def test_strips_workspace_v1_line(self):
        assert clean_user_text("[Workspace::v1: /tmp]\nrun the tests") == "run the tests"

    def test_strips_self_closing_workspace_context_but_keeps_task(self):
        text = "<workspace_context src=\"webui\"/> I want to continue tuning the speed loop."
        assert clean_user_text(text) == "I want to continue tuning the speed loop."

    def test_strips_transport_tag(self):
        assert clean_user_text("[patc883] why is my deploy failing?") == "why is my deploy failing?"

    def test_transport_tag_leaves_wrapper_marker_exposed(self):
        """After the tag is stripped the wrapper marker is at the head, skippable."""
        assert clean_user_text("[patc883] [ASYNC DELEGATION results below]") == "[ASYNC DELEGATION results below]"

    def test_bracketed_directive_not_treated_as_transport_tag(self):
        assert clean_user_text("[IMPORTANT: real content").startswith("[IMPORTANT:")

    def test_none_and_empty(self):
        assert clean_user_text(None) == ""
        assert clean_user_text("") == ""


class TestIsTaskLike:
    def test_genuine_task_passes(self):
        assert is_task_like("please review my pull request for the widget")

    def test_short_messages_rejected(self):
        assert not is_task_like("ok")
        assert not is_task_like("")

    def test_all_wrapper_families_rejected(self):
        wrappers = [
            "[CONTEXT COMPACTION — REFERENCE ONLY] earlier turns",
            "[ASYNC DELEGATION BATCH COMPLETE — deleg_x]",
            "[IMPORTANT: The user has invoked a skill]",
            "[OUT-OF-BAND USER MESSAGE — direct] payload",
            "[Continuing toward your standing goal of X]",
            "[System: Your previous response was truncated. Continue now]",
            "[System note: The following is recalled memory context]",
            "[Your active task list was preserved across context compression]",
            "[STILL IN PROGRESS — this is the active request]",
            "Available swarm workers: worker-1, worker-2",
        ]
        for w in wrappers:
            assert not is_task_like(w), w

    def test_marker_beyond_head_window_passes(self):
        """A marker quoted late in a long genuine task does not reject it."""
        task = "x" * 250 + " [IMPORTANT: quoted late] " + "y" * 50
        assert is_task_like(task)
