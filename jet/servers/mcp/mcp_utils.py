import asyncio
import mcp.types

from typing import List, Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from jet.servers.mcp.mcp_classes import ToolInfo
from jet.servers.mcp.config import MCP_SERVER_PATH
from jet.logger import logger

from pydantic import ValidationError, validate_call


def validate_tool_arguments(arguments: Dict[str, Any], schema: Dict[str, Any], tool_name: str) -> None:
    try:
        # Validate arguments against the tool's input schema
        @validate_call(config={"validate_json_schema": True})
        def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
            pass
        validate_schema(arguments, schema)
    except ValidationError as e:
        raise RuntimeError(
            f"Error executing tool '{tool_name}':\nInvalid tool arguments: {str(e)}"
        )


async def discover_tools(mcp_server_path: str = MCP_SERVER_PATH) -> List[ToolInfo]:
    server_params = StdioServerParameters(
        command="python", args=[mcp_server_path])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [{"name": t.name, "description": t.description, "schema": t.inputSchema, "outputSchema": t.outputSchema} for tool_type, tool_list in tools if tool_type == "tools" for t in tool_list]


async def execute_tool(tool_name: str, arguments: Dict[str, Any], mcp_server_path: str = MCP_SERVER_PATH) -> mcp.types.CallToolResult:
    server_params = StdioServerParameters(
        command="python", args=[mcp_server_path])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            for attempt in range(3):
                logger.info(
                    f"Attempt {attempt + 1} to call tool: {tool_name} with arguments: {arguments}")
                await session.initialize()
                logger.debug(
                    f"Calling tool: {tool_name} with arguments: {arguments}")
                # Wrap arguments in a nested 'arguments' field
                wrapped_arguments = {"arguments": arguments}
                result: mcp.types.CallToolResult = await session.call_tool(tool_name, wrapped_arguments)
                if not result.isError:
                    return result
                else:
                    if attempt == 2:
                        max_attempt_reached_err = f"Tool execution failed after {attempt + 1} attempts for tool '{tool_name}' with arguments: {arguments}"
                        logger.error(max_attempt_reached_err)
                        raise RuntimeError(max_attempt_reached_err)
                    await asyncio.sleep(0.5)
