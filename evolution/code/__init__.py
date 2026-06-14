"""Code evolution: iterative test-feedback repair of a single tool, gated.

Unlike artifact-text evolution (tool descriptions, skills), where a capable
agent's behavior is decoupled from the artifact, code is scored by a
deterministic test with no agent between the change and the verdict — so an
iterative repair loop has a real gradient on authentic bugs. The distinctive
asset here is the gate (held-out split, surface freeze, file scope, regression
floor), not the commodity repair loop.
"""
