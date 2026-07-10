import json
import os
from typing import Any, Callable, Dict

from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_two_numbers",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first number"},
                    "b": {"type": "integer", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract_two_numbers",
            "description": "Subtract two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first number"},
                    "b": {"type": "integer", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

available_functions: Dict[str, Callable[..., Any]] = {}
for tool in tools:
    name = tool["function"]["name"]
    if name == "add_two_numbers":
        available_functions[name] = lambda a, b: int(a) + int(b)
    elif name == "subtract_two_numbers":
        available_functions[name] = lambda a, b: int(a) - int(b)

messages = [{"role": "user", "content": "What is three plus one?"}]
print("Prompt:", messages[0]["content"])

stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model=os.getenv("LLAMA_CPP_LLM_MODEL", "not-needed"),
    messages=messages,
    temperature=0.0,
    tools=tools,
    stream_options={"include_usage": True},
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    },
    stream=True,
)

tool_calls: list[dict] = []
content_parts = []
chunk_count = 0

for chunk in stream:
    chunk_count += 1

    if not chunk.choices:
        if chunk.usage:
            logger.success(
                f"\nUsage: prompt={chunk.usage.prompt_tokens}, "
                f"completion={chunk.usage.completion_tokens}, "
                f"total={chunk.usage.total_tokens}",
                flush=True,
            )
        continue

    delta = chunk.choices[0].delta

    # Process tool calls - stream argument content directly
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            while len(tool_calls) <= idx:
                tool_calls.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            tool_call = tool_calls[idx]

            if tc_delta.id:
                tool_call["id"] += tc_delta.id
            if tc_delta.function and tc_delta.function.name:
                tool_call["function"]["name"] += tc_delta.function.name
                logger.info(f"\nTool call: {tc_delta.function.name}")
            if tc_delta.function and tc_delta.function.arguments:
                tool_call["function"]["arguments"] += tc_delta.function.arguments
                # Stream the raw argument content without prefix
                logger.teal(tc_delta.function.arguments, flush=True, end="")

    # Flush text content as it streams
    if delta.content:
        content_parts.append(delta.content)
        logger.teal(delta.content, flush=True, end="")

print()
logger.info(f"Stream complete: {chunk_count} chunks")

# Parse and execute tool calls
parsed_tool_calls: list[tuple[dict, dict]] = []
for tc in tool_calls:
    args_str = tc["function"]["arguments"]
    try:
        args_dict = json.loads(args_str)
        logger.success(f"Parsed tool call: {tc['function']['name']}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse args: {e}")
        continue

    clean_tc = {
        "id": tc["id"],
        "type": "function",
        "function": {
            "name": tc["function"]["name"],
            "arguments": args_str,
        },
    }
    parsed_tool_calls.append((clean_tc, args_dict))

if parsed_tool_calls:
    messages.append(
        {"role": "assistant", "tool_calls": [tc for tc, _ in parsed_tool_calls]}
    )

    for tool_call, arguments in parsed_tool_calls:
        func_name = tool_call["function"]["name"]
        logger.info(f"Executing: {func_name}", flush=True)

        if function_to_call := available_functions.get(func_name):
            print("\n\nCalling function:", func_name)
            print("Arguments:", arguments)
            output: Any = function_to_call(**arguments)
            print("Function output:", output)

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps({"result": output}),
                    "tool_call_id": tool_call["id"],
                }
            )
        else:
            logger.error(f"Function {func_name} not found")

    final_response = client.chat.completions.create(
        model=os.getenv("LLAMA_CPP_LLM_MODEL", "not-needed"),
        messages=messages,
    )
    print("Final response:", final_response.choices[0].message.content)
else:
    content = "".join(content_parts)
    print("No tool calls. Response:", content)
