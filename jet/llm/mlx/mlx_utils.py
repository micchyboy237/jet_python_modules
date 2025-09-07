import inspect
import json
import re
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union, get_args, get_origin
from collections.abc import Sequence

from pydantic.json_schema import JsonSchemaValue
from jet.llm.mlx.mlx_types import ToolArguments, ToolCall, ToolCallResult
from jet.logger import logger
from mlx_lm.utils import get_model_path, load_tokenizer, TokenizerWrapper
from jet.models.model_types import LLMModelType, Message
from jet.models.utils import resolve_model_value
from jet.servers.mcp.mcp_classes import ToolInfo
from jet.servers.mcp.mcp_utils import validate_tool_arguments, execute_tool
import asyncio


def parse_tool_call(llm_response: str) -> Dict[str, Any]:
    """
    Parses a single tool call from the LLM response.
    Args:
        llm_response: The LLM response containing a tool call in <tool_call> tags.
    Returns:
        Dict[str, Any]: A tool call dictionary, or None if not found or invalid.
    """
    from jet.logger import logger
    tool_open = "<tool_call>"
    tool_close = "</tool_call>"
    start_tool = llm_response.find(tool_open)
    if start_tool == -1:
        logger.warning("No <tool_call> tag found in LLM response.")
        return None
    start_tool += len(tool_open)
    end_tool = llm_response.find(tool_close, start_tool)
    if end_tool == -1:
        logger.warning("No </tool_call> tag found in LLM response.")
        return None
    try:
        tool_call = json.loads(llm_response[start_tool:end_tool].strip())
        return tool_call
    except json.JSONDecodeError:
        logger.error(
            f"Invalid JSON in tool call: {llm_response[start_tool:end_tool]}")
        return None


def parse_tool_calls(llm_response: str) -> List[Dict[str, Any]]:
    """
    Parses one or more tool calls from the LLM response.
    Args:
        llm_response: The LLM response containing tool call(s) in <tool_call> tags.
    Returns:
        List[Dict[str, Any]]: A list of tool call dictionaries.
    """
    from jet.logger import logger
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
        try:
            tool_call = json.loads(llm_response[start_tool:end_tool].strip())
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            logger.error(
                f"Invalid JSON in tool call: {llm_response[start_tool:end_tool]}")
        start_idx = end_tool + len(tool_close)
    return tool_calls


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


def get_tokenizer(model: LLMModelType) -> TokenizerWrapper:
    model_id = resolve_model_value(model)
    model_path = get_model_path(model_id)[0]
    tokenizer = load_tokenizer(model_path)
    return tokenizer


def get_tokenizer_info(model: LLMModelType) -> Dict:
    tokenizer = get_tokenizer(model)
    return {
        "has_thinking": tokenizer.has_thinking,
        "think_start": tokenizer.think_start,
        "think_end": tokenizer.think_end,
        "has_tool_calling": tokenizer.has_tool_calling,
        "tool_call_start": tokenizer.tool_call_start,
        "tool_call_end": tokenizer.tool_call_end,
        "eos_token_ids": list(tokenizer.eos_token_ids),
    }


def get_chat_template(model: LLMModelType) -> str:
    tokenizer = get_tokenizer(model)
    chat_template = tokenizer.chat_template or ""
    return chat_template


def has_tools(model: LLMModelType) -> bool:
    chat_template = get_chat_template(model)
    chat_template_str = str(chat_template)
    return "tools" in chat_template_str or "tool_use" in chat_template_str


def execute_tool_calls(
    tool_calls: Optional[List[Union[ToolArguments, ToolCall]]], tools: Optional[List[Callable]]
) -> Optional[List[ToolCallResult]]:
    """Execute tool calls from the response using provided tools."""
    if not tool_calls or not tools:
        return None

    results: List[ToolCallResult] = []

    for call in tool_calls:
        if "function" in call:
            tool_call = call["function"]
        else:
            tool_call = call
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})
        tool_result = None
        for tool in tools:
            if tool.__name__ == tool_name:
                try:
                    tool_result = tool(**tool_args)
                    logger.success(
                        f"Tool {tool_name} executed with result: {tool_result}"
                    )
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    tool_result = None
                break  # Only execute the first matching tool
        results.append({"tool_call": ToolCall(
            function=tool_call, type="function"), "tool_result": tool_result})

    return results


def process_response_format(
    input_data: Union[List[Message], str],
    response_format: Union[Literal["text", "json"], JsonSchemaValue],
) -> Union[List[Message], str]:
    """Process response format for messages or prompts, adding JSON instruction if needed."""
    # Improved JSON instruction, matching the style from MLXFunctionCaller
    if isinstance(response_format, dict):
        json_instruction = (
            "Return the response as a JSON object containing only the data fields defined in the following schema, "
            "without including the schema itself or any additional metadata:\n"
            f"{json.dumps(response_format, ensure_ascii=False)}\n"
            "For example, if the schema defines fields 'name' and 'age', return only {\"name\": \"value\", \"age\": number}."
        )
    else:
        json_instruction = (
            "Return the response as a JSON object. Do not include any extra metadata or schema definitions—"
            "just the data fields relevant to the user's request."
        )

    if isinstance(input_data, list):
        # Handle List[Message] for chat methods
        modified_messages = input_data.copy()  # Avoid modifying the input list
        if isinstance(response_format, (str, dict)) and (response_format == "json" or isinstance(response_format, dict)):
            # Check for existing system message
            system_msg_index = next((i for i, msg in enumerate(
                modified_messages) if msg.get("role") == "system"), None)
            if system_msg_index is not None:
                # System message exists; check if it mentions JSON
                content = modified_messages[system_msg_index].get("content", "")
                # Concatenate with two newlines
                modified_messages[system_msg_index]["content"] = content + \
                    f"\n\n{json_instruction}"
            else:
                # No system message; add new one
                modified_messages.insert(
                    0, {"role": "system", "content": json_instruction})
        return modified_messages
    elif isinstance(input_data, str):
        # Handle string prompt for generate methods
        if isinstance(response_format, (str, dict)) and (response_format == "json" or isinstance(response_format, dict)):
            return f"{json_instruction}\n\n{input_data}"
        return input_data
    else:
        raise ValueError("input_data must be a string or list of messages")
