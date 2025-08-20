import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from pydantic import ValidationError, validate_call
from jet.logger import CustomLogger
from jet.servers.mcp.mcp_classes import ToolRequest

MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")


async def discover_tools() -> List[Dict]:
    server_params = StdioServerParameters(
        command="python", args=[MCP_SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [{"name": t.name, "description": t.description, "schema": t.inputSchema, "outputSchema": t.outputSchema} for tool_type, tool_list in tools if tool_type == "tools" for t in tool_list]


async def execute_tool(tool_name: str, arguments: Dict[str, Any], logger: CustomLogger) -> str:
    server_params = StdioServerParameters(
        command="python", args=[MCP_SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            for attempt in range(3):
                try:
                    await session.initialize()
                    logger.debug(
                        f"Calling tool: {tool_name} with arguments: {arguments}")
                    wrapped_arguments = {"arguments": arguments}
                    result = await session.call_tool(tool_name, wrapped_arguments)
                    return str(result[0]["content"] if isinstance(result, list) and result and "content" in result[0] else result)
                except Exception as e:
                    if attempt == 2:
                        logger.error(
                            f"Tool execution failed after 3 attempts: {str(e)}")
                        return f"Error executing {tool_name}: {str(e)}"
                    await asyncio.sleep(0.5)


async def process_tool_requests(tool_requests: List[ToolRequest], tool_info: List[Dict], messages: List[Dict], logger: CustomLogger) -> Dict[str, Any]:
    tool_outputs = {}
    for tool_request in tool_requests:
        matching_tool = next(
            (tool for tool in tool_info if tool["name"] == tool_request.tool), None)
        if not matching_tool:
            messages.append(
                {"role": "user", "content": f"Error: Tool '{tool_request.tool}' not found"})
            continue
        resolved_args = {}
        for key, value in tool_request.arguments.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                placeholder = value[2:-2]
                try:
                    tool_name, field = placeholder.split(".")
                    if tool_name in tool_outputs and field in tool_outputs[tool_name]:
                        resolved_args[key] = tool_outputs[tool_name][field]
                    else:
                        messages.append(
                            {"role": "user", "content": f"Error: Invalid placeholder {value} for tool '{tool_request.tool}'"})
                        continue
                except ValueError:
                    messages.append(
                        {"role": "user", "content": f"Error: Malformed placeholder {value} for tool '{tool_request.tool}'"})
                    continue
            else:
                resolved_args[key] = value
        try:
            @validate_call(config=dict(validate_json_schema=True))
            def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
                pass
            validate_schema(resolved_args, matching_tool["schema"])
        except ValidationError as e:
            messages.append(
                {"role": "user", "content": f"Invalid tool arguments for '{tool_request.tool}': {str(e)}"})
            continue
        for attempt in range(2):
            tool_result = await execute_tool(tool_request.tool, resolved_args, logger)
            try:
                result_dict = json.loads(tool_result) if isinstance(
                    tool_result, str) and tool_result.strip().startswith("{") else {"content": tool_result}
                if tool_request.tool == "navigate_to_url" and "text_content" in result_dict:
                    if not result_dict["text_content"] or len(result_dict["text_content"].strip().split()) < 10:
                        messages.append(
                            {"role": "user", "content": f"Error: Insufficient text content from '{tool_request.tool}'"})
                        break
                elif tool_request.tool == "summarize_text" and "word_count" in result_dict:
                    if result_dict["word_count"] < 5:
                        messages.append(
                            {"role": "user", "content": f"Error: Summary too short from '{tool_request.tool}' (word_count: {result_dict['word_count']})"})
                        break
                tool_outputs[tool_request.tool] = result_dict
                messages.append(
                    {"role": "user", "content": f"Tool result: {tool_result}"})
                break
            except json.JSONDecodeError:
                tool_outputs[tool_request.tool] = {"content": tool_result}
                messages.append(
                    {"role": "user", "content": f"Tool result: {tool_result}"})
                break
            except Exception as e:
                if attempt == 1:
                    messages.append(
                        {"role": "user", "content": f"Error executing '{tool_request.tool}': {str(e)}"})
                    break
                await asyncio.sleep(0.5)
    return tool_outputs
