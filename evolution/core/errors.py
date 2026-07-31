"""Exit-code contract and runtime error classification for the evolution pipeline.

The cron wrapper (cron-evolve.sh) maps the evolve_skill process exit code to a
failure category. Every exit code used by the pipeline MUST be documented here
and mirrored in cron-evolve.sh's classifier.

Exit codes:
    0  success — including the constraint-failure path (artifact saved for
       review, deployment skipped; cron detects it via output text)
    1  generic error (skill not found, dataset build failure, usage error)
    2  FD exhaustion (OSError errno 24 EMFILE) — environment limit, NOT an
       LLM/API failure; raise `ulimit -n` and retry
    3  holdout-eval / LLM failure (dspy, network, or scoring error during the
       evaluation phase)

History: the 'error 1' escalations (esc-20260731-201857-1474) were EMFILE
crashes — bare `OSError: [Errno 24] Too many open files` tracebacks — that
Python reported as exit 1 and cron-evolve.sh mislabeled as "LLM provider
issue". This module exists so that failure mode gets its own exit code (2)
and a meaningful message.
"""

import errno

EXIT_GENERIC = 1
EXIT_EMFILE = 2
EXIT_EVAL_ERROR = 3

EMFILE_HINT = (
    "Too many open files (OSError errno 24 EMFILE). This is an environment "
    "file-descriptor limit, not an LLM/API failure. Raise the process limit "
    "(e.g. `ulimit -n 1024`) and retry."
)


def is_emfile(exc: BaseException) -> bool:
    """Return True if the exception is FD exhaustion (EMFILE, errno 24).

    Checks the errno directly, and falls back to message matching because
    dspy/litellm sometimes wrap the raw OS error in a generic Exception
    that carries the errno text but no errno attribute.
    """
    if isinstance(exc, OSError):
        return exc.errno == errno.EMFILE
    return "too many open files" in str(exc).lower()


def classify_error(exc: BaseException) -> tuple[int, str]:
    """Map a runtime exception to (exit_code, human_message).

    Distinguishes FD exhaustion from evaluation/LLM errors so the cron
    wrapper can report a meaningful cause instead of the generic
    'LLM provider issue' label that masked the original EMFILE crashes.
    """
    if is_emfile(exc):
        return EXIT_EMFILE, EMFILE_HINT
    msg = str(exc) or exc.__class__.__name__
    return EXIT_EVAL_ERROR, f"Evaluation failed: {msg}"
