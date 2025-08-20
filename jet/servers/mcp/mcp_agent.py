import asyncio
import json
import re
import mcp.types
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from pydantic import BaseModel, ValidationError, validate_call
from jet.llm.mlx.mlx_types import ChatTemplateArgs
from jet.logger import CustomLogger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelType, Message, Tool
from jet.servers.mcp.mcp_classes import ToolInfo
from jet.servers.mcp.utils import parse_tool_requests
from jet.servers.mcp.mcp_utils import discover_tools, execute_tool
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache

LOGS_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/playwright_mcp/server/logs"
CHAT_LOGS = f"{LOGS_DIR}/chats"
logger = CustomLogger(f"{LOGS_DIR}/mcp_agent.log", overwrite=True)
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")
MODEL_PATH: LLMModelType = "qwen3-1.7b-4bit"


async def query_llm(prompt: str, tool_info: List[Dict], previous_messages: List[Dict] = []) -> tuple[str, List[Dict]]:
    tool_descriptions = "\n\n".join(
        [f"Tool: {t['name']}\nDescription: {t['description']}\nInput Schema: {json.dumps(t['schema'], indent=2)}\nOutput Schema: {json.dumps(t['outputSchema'], indent=2)}" for t in tool_info])
    system_prompt = f"You are an AI assistant with MCP tools:\n{tool_descriptions}\nUse JSON for tool requests: {{'tool': 'name', 'arguments': {{'arg': 'value'}}}}."
    messages = [m for m in previous_messages if m["role"]
                != "system"] + [{"role": "user", "content": prompt}]

    # Format messages for MLX
    formatted_messages = [
        {"role": "system", "content": system_prompt}] + messages

    try:
        # Load MLX model
        model = MLXModelRegistry.load_model(MODEL_PATH)
        chat_template_args: ChatTemplateArgs = {
            "enable_thinking": False,
        }

        # Generate response
        llm_response = model.chat(
            formatted_messages,
            # prompt=tokenizer.apply_chat_template(
            #     formatted_messages, tokenize=False, enable_thinking=False),
            max_tokens=4000,
            temperature=0.7,
            verbose=True,
            chat_template_args=chat_template_args,
        )
        llm_response_text = llm_response["content"]

        tool_request_list = parse_tool_requests(llm_response_text, logger)

        for tool_request in tool_request_list:
            logger.debug(f"Tool request: {tool_request}")
            for tool in tool_info:
                if tool["name"] == tool_request.tool:
                    try:
                        # Validate arguments against the tool's input schema
                        @validate_call(config=dict(validate_json_schema=True))
                        def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
                            pass
                        validate_schema(
                            tool_request.arguments, tool["schema"])
                    except ValidationError as e:
                        return f"Invalid tool arguments: {str(e)}", messages
                    tool_result = await execute_tool(tool_request.tool, tool_request.arguments, MCP_SERVER_PATH)
                    messages.append(
                        {"role": "assistant", "content": llm_response_text})
                    messages.append(
                        {"role": "user", "content": f"Tool result: {tool_result}"})
                    # Generate follow-up response with tool result
                    follow_up_messages = [
                        {"role": "system", "content": system_prompt}] + messages
                    llm_follow_up_response = model.chat(
                        follow_up_messages,
                        max_tokens=4000,
                        verbose=True,
                    )
                    # return llm_follow_up_response["content"], messages
        return llm_response_text, messages
    except Exception as e:
        logger.error(f"MLX inference failed: {str(e)}")
        return f"Error querying LLM: {str(e)}", messages


async def chat_session():
    tools = await discover_tools(MCP_SERVER_PATH)
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
