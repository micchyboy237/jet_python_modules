import json
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._functions: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable, schema: dict):
        self._tools[name] = schema
        self._functions[name] = func
        logger.info(f"Registered tool: {name}")

    def get_schemas(self) -> List[dict]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: str) -> str:
        if name not in self._functions:
            return (
                f"Error: Unknown tool '{name}'. Available: {list(self._tools.keys())}"
            )

        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON arguments: {e}"

        try:
            result = self._functions[name](**args)
            return str(result) if result is not None else "(tool returned no output)"
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return f"Tool execution error: {type(e).__name__}: {str(e)[:300]}"
