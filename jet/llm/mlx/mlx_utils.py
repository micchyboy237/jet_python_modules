from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin
from collections.abc import Sequence
import inspect
import json
import re
from jet.servers.mcp.mcp_classes import ToolInfo
from jet.servers.mcp.mcp_utils import validate_tool_arguments, execute_tool
import asyncio


def parse_tool_call(llm_response: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Parses one or more tool calls from the LLM response.

    Args:
        llm_response: The LLM response containing tool call(s) in <tool_call> tags.

    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: A single tool call dict or a list of tool call dicts.
    """
    tool_open = "<tool_call>"
    tool_close = "</tool_call>"
    tool_calls = []
    start_idx = 0

    while True:
        start_tool = llm_response.find(tool_open, start_idx)
        if start_tool == -1:
            break
        start_tool += len(tool_open)
        end_tool = llm_response.find(tool_close, start_tool)
        if end_tool == -1:
            break
        tool_call = json.loads(llm_response[start_tool:end_tool].strip())
        # Rename 'name' to 'tool' to match ToolRequest model
        if "name" in tool_call:
            tool_call["tool"] = tool_call.pop("name")
        tool_calls.append(tool_call)
        start_idx = end_tool + len(tool_close)

    return tool_calls if len(tool_calls) > 1 else tool_calls[0] if tool_calls else {}


def create_mlx_tools(tools: List[ToolInfo]) -> List[Callable]:
    """
    Creates a list of sync callable functions from ToolInfo list for MLX model tool usage with type hints.

    Args:
        tools: List of ToolInfo containing name, description, schema, and output schema.

    Returns:
        List[Callable]: List of sync functions with type hints for MLX model compatibility.
    """
    def create_single_mlx_tool(tool_info: ToolInfo) -> Callable:
        def tool_function(**kwargs) -> Any:
            # Placeholder for MLX: actual execution happens in query_llm
            return {
                "name": tool_info["name"],
                "arguments": kwargs,
                "description": tool_info["description"],
                "schema": tool_info["schema"]
            }

        # Set function metadata for MLX model compatibility
        tool_function.__name__ = tool_info["name"]
        tool_function.__doc__ = tool_info["description"]

        # Create a signature with type hints from the schema
        parameters = tool_info["schema"].get(
            "arguments", {}).get("properties", {})
        sig_params = []
        for name, param_schema in parameters.items():
            param_type = str  # Default to str
            if param_schema.get("type") == "number":
                param_type = float
            elif param_schema.get("type") == "integer":
                param_type = int
            elif param_schema.get("type") == "boolean":
                param_type = bool
            elif param_schema.get("type") == "array":
                param_type = list
            sig_params.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=param_type
                )
            )
        tool_function.__signature__ = inspect.Signature(parameters=sig_params)

        return tool_function

    return [create_single_mlx_tool(tool_info) for tool_info in tools]


async def create_tool_function(tool_info: ToolInfo) -> Callable:
    """
    Creates an async callable function from ToolInfo for tool execution.

    Args:
        tool_info: ToolInfo containing name, description, schema, and output schema.

    Returns:
        Callable: An async function that executes the tool with validated arguments.
    """
    async def tool_function(**kwargs) -> Any:
        validate_tool_arguments(kwargs, tool_info["schema"], tool_info["name"])
        return await execute_tool(tool_info["name"], kwargs)

    # Set function metadata
    tool_function.__name__ = tool_info["name"]
    tool_function.__doc__ = tool_info["description"]

    # Create a signature with type hints from the schema
    parameters = tool_info["schema"].get("arguments", {}).get("properties", {})
    sig_params = []
    for name, param_schema in parameters.items():
        param_type = str
        if param_schema.get("type") == "number":
            param_type = float
        elif param_schema.get("type") == "integer":
            param_type = int
        elif param_schema.get("type") == "boolean":
            param_type = bool
        elif param_schema.get("type") == "array":
            param_type = list
        sig_params.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=param_type
            )
        )
    tool_function.__signature__ = inspect.Signature(parameters=sig_params)

    return tool_function
