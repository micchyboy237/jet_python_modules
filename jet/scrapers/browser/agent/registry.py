"""
Playwright Tool Registry
Maps tool names to their implementations and manages execution logic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from jet.scrapers.browser.agent.toolbox import PlaywrightToolbox
from jet.scrapers.browser.agent.tools import (
    TOOLS_RETURNING_SNAPSHOTS,
    get_tool_registry,
)
from pydantic import BaseModel
from rich.console import Console

console = Console()


class PlaywrightToolRegistry:
    def __init__(self, toolbox: PlaywrightToolbox):
        self.toolbox = toolbox
        self.tool_map = get_tool_registry()

    def get_tool_schema(self) -> List[Dict]:
        schemas = []
        for name, info in self.tool_map.items():
            model = info["model"]
            schema = (
                {"type": "object", "properties": {}}
                if model == BaseModel
                else model.model_json_schema()
            )
            schemas.append(
                {"name": name, "description": info["desc"], "parameters": schema}
            )
        return schemas

    async def execute_tool(
        self, name: str, arguments: Dict, step_dir: Optional[Path] = None
    ) -> Any:
        if name not in self.tool_map:
            raise ValueError(f"Tool {name} not found")

        model_class = self.tool_map[name]["model"]
        validated_args = (
            {}
            if model_class == BaseModel
            else model_class(**arguments).model_dump(exclude_none=True)
        )

        if step_dir and name in ["browser_snapshot", "browser_take_screenshot"]:
            validated_args["step_dir"] = step_dir

        method = getattr(self.toolbox, name, None)
        if not method:
            raise ValueError(f"Method {name} not implemented")

        return await method(**validated_args)

    def should_refresh_snapshot(self, tool_name: str) -> bool:
        """Determine if a snapshot should be refreshed after executing a tool."""
        return tool_name in TOOLS_RETURNING_SNAPSHOTS
