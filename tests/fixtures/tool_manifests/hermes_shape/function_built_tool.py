def _build_schema():
    return {
        "name": "function_built_tool",
        "description": "This is unreachable via pure AST.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


FUNCTION_BUILT_SCHEMA = _build_schema()
