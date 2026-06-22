"""Tests for the Phase 2 target tools constants."""

from evolution.tools.targets import (
    TARGET_TOOLS,
    LABEL_OPEN,
    LABEL_CLOSE,
    BLOCK_START,
    BLOCK_END,
)


def test_target_tools_count():
    """Structural checks that guard against truncation, corruption, or dupes."""
    assert len(TARGET_TOOLS) == 6
    assert all(t and isinstance(t, str) for t in TARGET_TOOLS)
    assert len(TARGET_TOOLS) == len(set(TARGET_TOOLS))


def test_label_markers_are_distinct():
    assert LABEL_OPEN != LABEL_CLOSE
    assert len(LABEL_OPEN) == 2
    assert len(LABEL_CLOSE) == 2


def test_label_markers_do_not_collide_with_sentinels():
    """The mutable block's label names and boundary markers must be distinct
    so that the parser can unambiguously find each segment."""
    assert BLOCK_START != BLOCK_END
    assert BLOCK_START != LABEL_OPEN
    assert BLOCK_END != LABEL_CLOSE
    # Neither sentinel may contain the label markers
    assert LABEL_OPEN not in BLOCK_START
    assert LABEL_CLOSE not in BLOCK_START
    assert LABEL_OPEN not in BLOCK_END
    assert LABEL_CLOSE not in BLOCK_END


def test_all_five_constants_importable():
    """Sanity: every exported name resolves and is a string."""
    for name in (LABEL_OPEN, LABEL_CLOSE, BLOCK_START, BLOCK_END):
        assert isinstance(name, str) and len(name) > 0
