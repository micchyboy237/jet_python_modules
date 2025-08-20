import json
from typing import List, Dict, Tuple
from pydantic import ValidationError
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from jet.servers.mcp.utils import parse_tool_requests
from jet.servers.mcp.mcp_classes import ToolRequest
from jet.logger import CustomLogger

MODEL_PATH = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"


async def query_llm(prompt: str, tool_info: List[Dict], previous_messages: List[Dict], logger: CustomLogger) -> Tuple[str, List[Dict]]:
    tool_descriptions = "\n\n".join(
        [f"Tool: {t['name']}\nDescription: {t['description']}\nInput Schema: {json.dumps(t['schema'], indent=2)}\nOutput Schema: {json.dumps(t['outputSchema'], indent=2)}" for t in tool_info])
    system_prompt = (
        f"You are an AI assistant with MCP tools:\n{tool_descriptions}\n"
        "Use JSON for tool requests: {'tool': 'name', 'arguments': {'arg': 'value'}}.\n"
        "For chained tools, use placeholders like {{tool_name.output_field}} (e.g., {{navigate_to_url.text_content}})."
    )
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
        tool_requests = parse_tool_requests(llm_response, logger)
        if tool_requests:
            messages.append({"role": "assistant", "content": llm_response})
            return llm_response, messages, tool_requests
        return llm_response, messages, []
    except Exception as e:
        logger.error(f"MLX inference failed: {str(e)}")
        return f"Error querying LLM: {str(e)}", messages, []
