"""A fake hermes-agent checkout, shaped like the real one.

Every Phase 2 test runs against this instead of a real repo, and nothing here
touches a network or a language model. The shapes are copied from the actual
hermes-agent source: schema dicts as module-level constants with inline
parameter descriptions, a parenthesised implicit-concat description, and
``registry.register(name=..., toolset=...)`` calls at the bottom of the module.

Two of the fixtures are deliberately over budget, because the real repo is:
``terminal``'s description exceeds 500 chars and ``write_file.cross_profile``
exceeds 200. Code that only works on a tidy catalogue is code that does not work.
"""

import json

import pytest

# 500+ chars, mirroring the real read_file description being 539.
LONG_TOOL_DESC = (
    "Run a shell command in a persistent session. "
    + "Prefer the purpose-built file tools over shell equivalents wherever one exists. " * 6
)

# 200+ chars, mirroring the real write_file.cross_profile parameter being 302.
LONG_PARAM_DESC = (
    "Opt out of the cross-profile soft guard. Defaults to false. "
    + "Set true only after explicit user direction to edit another profile. " * 4
)


def _file_tools_source() -> str:
    return f'''"""File tools."""

from tools import registry

READ_FILE_SCHEMA = {{
    "name": "read_file",
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal.",
    "parameters": {{
        "type": "object",
        "properties": {{
            "path": {{"type": "string", "description": "Path to the file to read"}},
            "offset": {{"type": "integer", "description": "Line number to start from", "default": 1}},
            "limit": {{"type": "integer", "description": "Maximum number of lines to read", "default": 500}}
        }},
        "required": ["path"]
    }}
}}

SEARCH_FILES_SCHEMA = {{
    "name": "search_files",
    "description": (
        "Search file contents or filenames with a regular expression.\\n"
        "\\n"
        "Use this instead of grep or find in terminal."
    ),
    "parameters": {{
        "type": "object",
        "properties": {{
            "pattern": {{"type": "string", "description": "Regular expression to search for"}},
            "target": {{"type": "string", "enum": ["content", "files"], "description": "Search inside files or match names", "default": "content"}},
            "path": {{"type": "string", "description": "Directory to search in"}}
        }},
        "required": ["pattern"]
    }}
}}

WRITE_FILE_SCHEMA = {{
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content.",
    "parameters": {{
        "type": "object",
        "properties": {{
            "path": {{"type": "string", "description": "Path to the file to write"}},
            "content": {{"type": "string", "description": "Complete content to write"}},
            "cross_profile": {{
                "type": "boolean",
                "description": {json.dumps(LONG_PARAM_DESC)},
                "default": False,
            }},
        }},
        "required": ["path", "content"]
    }}
}}


def _handle(args, **kwargs):
    return None


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle, emoji="R")
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle, emoji="S")
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle, emoji="W")
'''


def _shell_tools_source() -> str:
    return f'''"""Shell tools."""

from tools import registry

TERMINAL_SCHEMA = {{
    "name": "terminal",
    "description": {json.dumps(LONG_TOOL_DESC)},
    "parameters": {{
        "type": "object",
        "properties": {{
            "command": {{"type": "string", "description": "Shell command to run"}},
            "timeout": {{"type": "integer", "description": "Seconds before giving up", "default": 120}}
        }},
        "required": ["command"]
    }}
}}


def _handle_terminal(args, **kwargs):
    return None


registry.register(name="terminal", toolset="terminal", schema=TERMINAL_SCHEMA, handler=_handle_terminal)
'''


# No registry.register call: exercises the inferred-toolset path.
VISION_TOOLS_SOURCE = '''"""Vision tools."""

VISION_SCHEMA = {
    "name": "vision_analyze",
    "description": "Describe the contents of an image file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the image"}
        },
        "required": ["path"]
    }
}
'''


@pytest.fixture
def hermes_repo(tmp_path):
    """A tmp_path directory that discover_tool_schemas can read."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "file_tools.py").write_text(_file_tools_source())
    (tools / "shell_tools.py").write_text(_shell_tools_source())
    (tools / "vision_tools.py").write_text(VISION_TOOLS_SOURCE)
    (tools / "registry.py").write_text('"""Registry."""\n\n\ndef register(**kwargs):\n    return None\n')
    return tmp_path


@pytest.fixture
def empty_repo(tmp_path):
    """A checkout with no tools/ directory at all."""
    (tmp_path / "agent").mkdir()
    return tmp_path
