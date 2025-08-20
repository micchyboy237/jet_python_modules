import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from pydantic import BaseModel, ValidationError, validate_call
from jet.logger import CustomLogger
from jet.servers.mcp.server.utils import parse_tool_requests
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

LOGS_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/playwright_mcp/server/logs"
logger = CustomLogger(f"{LOGS_DIR}/mcp_agent.log", overwrite=True)
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")
MODEL_PATH = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"


async def discover_tools() -> List[Dict]:
    server_params = StdioServerParameters(
        command="python", args=[MCP_SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [{"name": t.name, "description": t.description, "schema": t.inputSchema, "outputSchema": t.outputSchema} for tool_type, tool_list in tools if tool_type == "tools" for t in tool_list]


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
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


async def query_llm(prompt: str, tool_info: List[Dict], previous_messages: List[Dict] = []) -> tuple[str, List[Dict]]:
    """
    Query the LLM with a prompt and tool information, handling chained tool requests.

    Args:
        prompt: User input prompt.
        tool_info: List of available tools with their schemas.
        previous_messages: Previous conversation messages.

    Returns:
        Tuple of the final LLM response and updated messages.
    """
    tool_descriptions = "\n\n".join(
        [f"Tool: {t['name']}\nDescription: {t['description']}\nInput Schema: {json.dumps(t['schema'], indent=2)}\nOutput Schema: {json.dumps(t['outputSchema'], indent=2)}" for t in tool_info])
    system_prompt = f"You are an AI assistant with MCP tools:\n{tool_descriptions}\nUse JSON for tool requests: {{'tool': 'name', 'arguments': {{'arg': 'value'}}}}."
    messages = [m for m in previous_messages if m["role"]
                != "system"] + [{"role": "user", "content": prompt}]
    formatted_messages = [
        {"role": "system", "content": system_prompt}] + messages

    try:
        model, tokenizer = load(MODEL_PATH)
        sampler = make_sampler(temp=0.7)
        llm_response = generate(
            model,
            tokenizer,
            prompt=tokenizer.apply_chat_template(
                formatted_messages, tokenize=False, enable_thinking=False),
            max_tokens=4000,
            sampler=sampler,
            verbose=True,
        )
        # Parse multiple tool requests using parse_tool_requests
        tool_requests = parse_tool_requests(llm_response, logger)
        if tool_requests:
            messages.append({"role": "assistant", "content": llm_response})
            for tool_request in tool_requests:
                # Find matching tool in tool_info
                matching_tool = next(
                    (tool for tool in tool_info if tool["name"] == tool_request.tool), None)
                if not matching_tool:
                    messages.append({
                        "role": "user",
                        "content": f"Error: Tool '{tool_request.tool}' not found"
                    })
                    continue
                try:
                    @validate_call(config=dict(validate_json_schema=True))
                    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
                        pass
                    validate_schema(tool_request.arguments,
                                    matching_tool["schema"])
                    tool_result = await execute_tool(tool_request.tool, tool_request.arguments)
                    messages.append(
                        {"role": "user", "content": f"Tool result: {tool_result}"})
                except ValidationError as e:
                    messages.append({
                        "role": "user",
                        "content": f"Invalid tool arguments for '{tool_request.tool}': {str(e)}"
                    })
            # Perform follow-up LLM query with all tool results
            follow_up_messages = [
                {"role": "system", "content": system_prompt}] + messages
            llm_response = generate(
                model,
                tokenizer,
                prompt=tokenizer.apply_chat_template(
                    follow_up_messages, tokenize=False, enable_thinking=False),
                max_tokens=4000,
                sampler=sampler,
                verbose=True,
            )
        return llm_response, messages
    except Exception as e:
        logger.error(f"MLX inference failed: {str(e)}")
        return f"Error querying LLM: {str(e)}", messages


async def chat_session():
    tools = await discover_tools()
    logger.debug(
        f"Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            logger.debug("Ending chat session.")
            break
        response, messages = await query_llm(user_input, tools, messages)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(chat_session())
