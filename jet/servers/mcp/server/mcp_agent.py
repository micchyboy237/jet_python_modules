import asyncio
import json
from pathlib import Path
import re
from typing import List, Dict, Any, Tuple
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from pydantic import BaseModel, ValidationError, validate_call
from jet.logger import CustomLogger
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

LOGS_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/playwright_mcp/server/logs"
logger = CustomLogger(f"{LOGS_DIR}/mcp_agent.log", overwrite=True)
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")
MODEL_PATH = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"


class ToolRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any]


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
                    # Wrap arguments in a nested 'arguments' field
                    wrapped_arguments = {"arguments": arguments}
                    result = await session.call_tool(tool_name, wrapped_arguments)
                    return str(result[0]["content"] if isinstance(result, list) and result and "content" in result[0] else result)
                except Exception as e:
                    if attempt == 2:
                        logger.error(
                            f"Tool execution failed after 3 attempts: {str(e)}")
                        return f"Error executing {tool_name}: {str(e)}"
                    await asyncio.sleep(0.5)


def parse_tool_requests(llm_response: str, logger: CustomLogger) -> List[ToolRequest]:
    """
    Parse multiple JSON tool requests from the LLM response.

    Args:
        llm_response: Raw response string from the LLM.
        logger: Logger instance for debugging and error logging.

    Returns:
        List of valid ToolRequest objects.
    """
    json_matches = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', llm_response)
    logger.debug(f"Found {len(json_matches)} JSON objects in response")
    tool_requests = []
    for json_str in json_matches:
        try:
            tool_request = ToolRequest.model_validate_json(json_str)
            logger.debug(f"Valid tool request: {tool_request}")
            tool_requests.append(tool_request)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(
                f"Invalid JSON object skipped: {json_str}, error: {str(e)}")
    return tool_requests


async def query_llm(prompt: str, tool_info: List[Dict], previous_messages: List[Dict] = []) -> Tuple[str, List[Dict]]:
    tool_descriptions = "\n\n".join(
        [f"Tool: {t['name']}\nDescription: {t['description']}\nInput Schema: {json.dumps(t['schema'], indent=2)}\nOutput Schema: {json.dumps(t['outputSchema'], indent=2)}" for t in tool_info])
    system_prompt = f"You are an AI assistant with MCP tools:\n{tool_descriptions}\nUse JSON for tool requests: {{'tool': 'name', 'arguments': {{'arg': 'value'}}}}. For summarization tasks, first call navigate_to_url, then use its text_content for summarize_text."
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
        logger.debug(f"LLM raw response: {llm_response}")
        tool_requests = parse_tool_requests(llm_response, logger)
        if not tool_requests:
            logger.debug("No valid tool requests detected in response")
            return llm_response, messages

        for tool_request in tool_requests:
            for tool in tool_info:
                if tool["name"] == tool_request.tool:
                    try:
                        @validate_call(config=dict(validate_json_schema=True))
                        def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
                            pass
                        validate_schema(tool_request.arguments, tool["schema"])
                    except ValidationError as e:
                        logger.error(f"Invalid tool arguments: {str(e)}")
                        return f"Invalid tool arguments: {str(e)}", messages
                    tool_result = await execute_tool(tool_request.tool, tool_request.arguments)
                    logger.debug(f"Tool result: {tool_result}")
                    if tool_request.tool == "navigate_to_url":
                        try:
                            url_output = json.loads(tool_result)
                            if url_output.get("text_content"):
                                messages.append(
                                    {"role": "assistant", "content": llm_response})
                                messages.append(
                                    {"role": "user", "content": f"Tool result: {tool_result}"})
                                summarize_request = {
                                    "tool": "summarize_text",
                                    "arguments": {"text": url_output["text_content"], "max_words": 100}
                                }
                                logger.debug(
                                    f"Chaining to summarize_text: {summarize_request}")
                                tool_result = await execute_tool(summarize_request["tool"], summarize_request["arguments"])
                                messages.append(
                                    {"role": "user", "content": f"Tool result: {tool_result}"})
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse navigate_to_url output: {str(e)}")
                            return f"Error processing URL output: {str(e)}", messages
                    messages.append(
                        {"role": "assistant", "content": llm_response})
                    messages.append(
                        {"role": "user", "content": f"Tool result: {tool_result}"})
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
                    logger.debug(f"Follow-up LLM response: {llm_response}")
                    return llm_response, messages
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
