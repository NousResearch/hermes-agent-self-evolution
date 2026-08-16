"""Built-in objective verifiers.

Each module here self-registers its verifier (via @register_verifier)
when imported. load_builtins() imports them all; the registry in
evolution.core.verifier calls it lazily so that running a verifier
module directly (python -m evolution.verifiers.arxiv_verifier) does not
import the module twice.
"""

import importlib

_BUILTIN_MODULES = (
    "evolution.verifiers.arxiv_verifier",
)


def load_builtins() -> None:
    """Import every built-in verifier module, registering its verifier."""
    for module_name in _BUILTIN_MODULES:
        importlib.import_module(module_name)
