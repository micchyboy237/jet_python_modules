import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from pydantic import BaseModel, ValidationError, validate_call
from jet.logger import CustomLogger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelType, Message
from jet.servers.mcp.utils import parse_tool_requests
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

LOGS_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/playwright_mcp/server/logs"
CHAT_LOGS = f"{LOGS_DIR}/chats"
logger = CustomLogger(f"{LOGS_DIR}/mcp_agent.log", overwrite=True)
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")
MODEL_PATH: LLMModelType = "qwen3-1.7b-4bit"


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


async def query_llm(prompt: str, tool_info: List[Dict], previous_messages: List[Dict] = []) -> tuple[str, List[Message]]:
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
    system_prompt = (
        f"You are an AI assistant with MCP tools:\n{tool_descriptions}\n"
        "Use JSON for tool requests: {'tool': 'name', 'arguments': {'arg': 'value'}}.\n"
        "For chained tools, use placeholders like {{tool_name.output_field}} (e.g., {{navigate_to_url.text_content}})."
    )
    messages: List[Message] = [m for m in previous_messages if m["role"]
                               != "system"] + [{"role": "user", "content": prompt}]
    formatted_messages: List[Message] = [
        {"role": "system", "content": system_prompt}] + messages

    try:
        model = MLXModelRegistry.load_model(MODEL_PATH)
        llm_response = model.chat(
            formatted_messages,
            max_tokens=4000,
            temperature=0.7,
            log_dir=CHAT_LOGS,
            verbose=True,
        )
        response_text = llm_response["content"]
        # Parse multiple tool requests
        tool_requests = parse_tool_requests(response_text, logger)
        if tool_requests:
            messages.append({"role": "assistant", "content": response_text})
            tool_outputs = {}  # Store outputs: {tool_name: output_dict}
            for tool_request in tool_requests:
                # Find matching tool
                matching_tool = next(
                    (tool for tool in tool_info if tool["name"] == tool_request.tool), None)
                if not matching_tool:
                    messages.append({
                        "role": "user",
                        "content": f"Error: Tool '{tool_request.tool}' not found"
                    })
                    continue

                # Resolve placeholders in arguments
                resolved_args = {}
                for key, value in tool_request.arguments.items():
                    if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                        placeholder = value[2:-2]
                        try:
                            tool_name, field = placeholder.split(".")
                            if tool_name in tool_outputs and field in tool_outputs[tool_name]:
                                resolved_args[key] = tool_outputs[tool_name][field]
                            else:
                                messages.append({
                                    "role": "user",
                                    "content": f"Error: Invalid placeholder {value} for tool '{tool_request.tool}'"
                                })
                                continue
                        except ValueError:
                            messages.append({
                                "role": "user",
                                "content": f"Error: Malformed placeholder {value} for tool '{tool_request.tool}'"
                            })
                            continue
                    else:
                        resolved_args[key] = value

                # Validate resolved arguments
                try:
                    @validate_call(config=dict(validate_json_schema=True))
                    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
                        pass
                    validate_schema(resolved_args, matching_tool["schema"])
                except ValidationError as e:
                    messages.append({
                        "role": "user",
                        "content": f"Invalid tool arguments for '{tool_request.tool}': {str(e)}"
                    })
                    continue

                # Execute tool with retry
                for attempt in range(2):
                    tool_result = await execute_tool(tool_request.tool, resolved_args)
                    try:
                        # Parse tool result as JSON
                        result_dict = json.loads(tool_result) if isinstance(
                            tool_result, str) and tool_result.strip().startswith("{") else {"content": tool_result}
                        # Validate output
                        if tool_request.tool == "navigate_to_url" and "text_content" in result_dict:
                            if not result_dict["text_content"] or len(result_dict["text_content"].strip().split()) < 10:
                                messages.append({
                                    "role": "user",
                                    "content": f"Error: Insufficient text content from '{tool_request.tool}'"
                                })
                                break
                        elif tool_request.tool == "summarize_text" and "word_count" in result_dict:
                            if result_dict["word_count"] < 5:
                                messages.append({
                                    "role": "user",
                                    "content": f"Error: Summary too short from '{tool_request.tool}' (word_count: {result_dict['word_count']})"
                                })
                                break
                        tool_outputs[tool_request.tool] = result_dict
                        messages.append(
                            {"role": "user", "content": f"Tool result: {tool_result}"})
                        break
                    except json.JSONDecodeError:
                        tool_outputs[tool_request.tool] = {
                            "content": tool_result}
                        messages.append(
                            {"role": "user", "content": f"Tool result: {tool_result}"})
                        break
                    except Exception as e:
                        if attempt == 1:
                            messages.append({
                                "role": "user",
                                "content": f"Error executing '{tool_request.tool}': {str(e)}"
                            })
                            break
                        await asyncio.sleep(0.5)

            # # Perform follow-up LLM query with all tool results
            # follow_up_messages: List[Message] = [
            #     {"role": "system", "content": system_prompt}] + messages
            # llm_response = model.chat(
            #     follow_up_messages,
            #     max_tokens=4000,
            #     temperature=0.7,
            #     log_dir=CHAT_LOGS,
            #     verbose=True,
            # )
        return response_text, messages
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
