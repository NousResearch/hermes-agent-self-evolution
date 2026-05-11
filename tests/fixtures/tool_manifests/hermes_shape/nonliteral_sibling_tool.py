ALLOWED_MODES = ["read", "write", "patch"]

NONLITERAL_SIBLING_SCHEMA = {
    "name": "nonliteral_sibling",
    "description": "Description is literal but a sibling is a Name reference.",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ALLOWED_MODES},
        },
        "required": ["mode"],
    },
}
