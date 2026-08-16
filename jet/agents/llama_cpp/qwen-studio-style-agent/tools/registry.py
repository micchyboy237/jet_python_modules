import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Adapter between agent tools and jet.adapters.llama_cpp.llm_utils.

    llm_utils expects dict[str, Callable]; this class provides schema
    management + safe execution wrapping for non-agentic contexts.
    """

    def __init__(self):
        self._schemas: list[dict[str, Any]] = []
        self._functions: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any], schema: dict[str, Any]):
        self._schemas.append(schema)
        self._functions[name] = func
        logger.info(f"Registered tool: {name}")

    def get_schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def as_llm_utils_registry(self) -> dict[str, Callable[..., Any]]:
        """Return the dict format expected by llm_utils.chat(tool_registry=...)."""
        return dict(self._functions)

    def execute(self, name: str, arguments: str) -> str:
        """Standalone execution for non-agentic contexts (e.g., testing)."""
        if name not in self._functions:
            return f"Error: Unknown tool '{name}'. Available: {list(self._functions.keys())}"
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON arguments: {e}"
        try:
            result = self._functions[name](**args)
            return str(result) if result is not None else "(no output)"
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return f"Tool execution error: {type(e).__name__}: {str(e)[:300]}"
