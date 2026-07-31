"""Tests for the evolution exit-code contract and error classification.

Covers the 'error 1' failure mode from esc-20260731-201857-1474: the EMFILE
(OSError errno 24) crashes that previously surfaced as a generic exit 1 and
were mislabeled "LLM provider issue" by cron-evolve.sh.
"""

import errno

from evolution.core.errors import (
    EXIT_EMFILE,
    EXIT_EVAL_ERROR,
    EMFILE_HINT,
    classify_error,
    is_emfile,
)


class TestIsEmfile:
    def test_oserror_errno_24(self):
        assert is_emfile(OSError(errno.EMFILE, "Too many open files"))

    def test_oserror_other_errno(self):
        assert not is_emfile(OSError(errno.EACCES, "Permission denied"))

    def test_wrapped_message_fallback(self):
        # dspy/litellm sometimes wrap the raw OS error in a generic Exception
        # carrying the errno text but no errno attribute
        assert is_emfile(Exception("Too many open files (errno 24)"))

    def test_unrelated_exception(self):
        assert not is_emfile(ValueError("bad value"))


class TestClassifyError:
    def test_emfile_maps_to_exit_2_with_hint(self):
        code, msg = classify_error(OSError(errno.EMFILE, "Too many open files"))
        assert code == EXIT_EMFILE
        assert "ulimit" in msg
        assert "not an LLM/API failure" in msg

    def test_emfile_hint_is_actionable(self):
        assert "ulimit -n 1024" in EMFILE_HINT

    def test_non_emfile_eval_error_maps_to_exit_3(self):
        code, msg = classify_error(ValueError("bad value"))
        assert code == EXIT_EVAL_ERROR
        assert "bad value" in msg

    def test_oserror_other_errno_is_eval_error_not_emfile(self):
        code, _ = classify_error(OSError(errno.EACCES, "Permission denied"))
        assert code == EXIT_EVAL_ERROR
