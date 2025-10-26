# jet_python_modules/jet/libs/llama_cpp/jet_examples/tools-stream.py
import json
from openai import OpenAI
from typing import Any, Dict, Callable

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

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

# Map function name → callable
available_functions: Dict[str, Callable[..., Any]] = {}
for tool in tools:
    name = tool["function"]["name"]
    if name == "add_two_numbers":
        available_functions[name] = lambda a, b: int(a) + int(b)
    elif name == "subtract_two_numbers":
        available_functions[name] = lambda a, b: int(a) - int(b)

messages = [{"role": "user", "content": "What is three plus one?"}]
print("Prompt:", messages[0]["content"])

create_kwargs = {
    "messages": messages,
    "stream": True,
    "model": "qwen3-instruct-2507:4b",
    "temperature": 0.0,
    # "tool_choice": "auto",
    "tools": tools,
}

stream = client.chat.completions.create(**create_kwargs)
chunks = list(stream)

# ----------------------------------------------------------------------
# 1. Accumulate streamed tool-call deltas
# ----------------------------------------------------------------------
tool_calls: list[dict] = []
current_tool_call = None
accumulated_args = ""

for chunk in chunks:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            # Ensure we have a slot for this index
            while len(tool_calls) <= idx:
                tool_calls.append(
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
            tool_call = tool_calls[idx]

            if tc_delta.id:
                tool_call["id"] += tc_delta.id
            if tc_delta.function and tc_delta.function.name:
                tool_call["function"]["name"] += tc_delta.function.name
            if tc_delta.function and tc_delta.function.arguments:
                tool_call["function"]["arguments"] += tc_delta.function.arguments

# ----------------------------------------------------------------------
# 2. Parse arguments **once** for execution, but keep original JSON string
# ----------------------------------------------------------------------
parsed_tool_calls: list[tuple[dict, dict]] = []   # (tool_call_for_history, args_dict)

for tc in tool_calls:
    args_str = tc["function"]["arguments"]
    try:
        args_dict = json.loads(args_str)
    except json.JSONDecodeError:
        print(f"Invalid JSON in arguments: {args_str}")
        continue

    # Build a clean tool-call dict that contains the *raw* JSON string
    clean_tc = {
        "id": tc["id"],
        "type": "function",
        "function": {
            "name": tc["function"]["name"],
            "arguments": args_str,               # ← critical: string, not dict
        },
    }
    parsed_tool_calls.append((clean_tc, args_dict))

# ----------------------------------------------------------------------
# 3. Append assistant message with **string** arguments
# ----------------------------------------------------------------------
messages.append(
    {"role": "assistant", "tool_calls": [tc for tc, _ in parsed_tool_calls]}
)

# ----------------------------------------------------------------------
# 4. Execute tools (if any) and send tool responses back to the model
# ----------------------------------------------------------------------
if parsed_tool_calls:
    for tool_call, arguments in parsed_tool_calls:
        func_name = tool_call["function"]["name"]
        if function_to_call := available_functions.get(func_name):
            print("Calling function:", func_name)
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
            print("Function", func_name, "not found")

    # Final non-streaming call to get the model’s answer
    final_response = client.chat.completions.create(
        model="qwen3-instruct-2507:4b",
        messages=messages,
    )
    print("Final response:", final_response.choices[0].message.content)
else:
    # No tool calls – just stream the text response
    content = "".join(
        c.choices[0].delta.content or ""
        for c in chunks
        if c.choices and c.choices[0].delta.content
    )
    print("No tool calls. Response:", content)
