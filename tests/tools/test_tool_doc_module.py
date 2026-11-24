"""Tests for the tool-description artifact module and its extraction contract."""

from unittest.mock import Mock

import dspy
import pytest

from evolution.tools.tool_doc_module import (
    SELECT_TEMPLATE,
    ToolDocModule,
    extract_description,
)

DESCRIPTION = (
    "Search file contents or find files by name. Faster than grep; "
    "supports regex and glob targets."
)


class TestToolDocModule:
    def test_template_holds_the_description_between_markers(self):
        mod = ToolDocModule("search_files", DESCRIPTION)

        assert mod.extract() == SELECT_TEMPLATE.format(
            name="search_files", description=DESCRIPTION
        )

    def test_description_roundtrips(self):
        mod = ToolDocModule("search_files", DESCRIPTION)

        assert extract_description(mod.extract()) == DESCRIPTION

    def test_multiline_description_roundtrips(self):
        description = "Line one.\n\nContent search: regex.\nFile search: glob."

        mod = ToolDocModule("search_files", description)

        assert extract_description(mod.extract()) == description

    def test_forward_wires_to_underlying_predictor(self):
        mod = ToolDocModule("search_files", DESCRIPTION)
        mod.predict = Mock(return_value=Mock(output="search_files"))

        result = mod.forward("find the config files")

        mod.predict.assert_called_once_with(task_input="find the config files")
        assert result.output == "search_files"
        assert isinstance(result, dspy.Prediction)

    @pytest.mark.parametrize(
        "mangled",
        [
            "no markers at all",
            "Description: something",
            "<<DESCRIPTION>>missing the closing marker",
            "closing only<</DESCRIPTION>>",
        ],
    )
    def test_extraction_rejects_mangled_markers(self, mangled):
        with pytest.raises(ValueError):
            extract_description(mangled)
