"""
Hermes PHI — Protected Health Information safety plugin for Hermes Agent.

Provides a drop-in plugin that wraps the llm-common Safety Kernel to redact
or block PHI before text leaves the process boundary for remote LLM calls.
"""

from hermes_phi.plugin import HermesPHIPlugin, PHISurface, PHIVerdict

__version__ = "0.1.0"
__all__ = ["HermesPHIPlugin", "PHISurface", "PHIVerdict"]

