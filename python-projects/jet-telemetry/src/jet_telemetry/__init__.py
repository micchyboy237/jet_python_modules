"""
Jet Telemetry: Minimalist shared instrumentation library.
Provides standardized setup and 4 core decorators for Arize Phoenix observability.
"""

from .decorators import chain, llm, tool, trace
from .setup import initialize_telemetry

__all__ = ["initialize_telemetry", "llm", "tool", "chain", "trace"]
