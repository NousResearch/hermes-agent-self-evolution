"""Tool with non-ASCII characters in the description.

Verifies the byte-offset splice path handles UTF-8 multi-byte chars correctly.
"""

NONASCII_SCHEMA = {
    "name": "nonascii_tool",
    "description": "Use this — not that — for em-dash heavy descriptions.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
