import asyncio
from openai import AsyncOpenAI
from typing import Any, Dict
import json

client = AsyncOpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

# === Tools schema (single source of truth) ===
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

# === Dynamically derive callables from tool names ===
available_functions: Dict[str, callable] = {}

for tool in tools:
    name = tool["function"]["name"]
    if name == "add_two_numbers":
        available_functions[name] = lambda a, b: int(a) + int(b)
    elif name == "subtract_two_numbers":
        available_functions[name] = lambda a, b: int(a) - int(b)
    # Add more ops here or use registry pattern for scale

messages = [{'role': 'user', 'content': 'What is three plus one?'}]
print('Prompt:', messages[0]['content'])

async def main():
    response = await client.chat.completions.create(
        model="qwen3-instruct-2507:4b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if function_to_call := available_functions.get(func_name):
                print('Calling function:', func_name)
                arguments = json.loads(tool_call.function.arguments)
                print('Arguments:', arguments)
                output: Any = function_to_call(**arguments)
                print('Function output:', output)

                messages.append(response.choices[0].message)
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"result": output}),
                    "tool_call_id": tool_call.id,
                })

                # Final response
                final_response = await client.chat.completions.create(
                    model="qwen3-instruct-2507:4b",
                    messages=messages,
                )
                print('Final response:', final_response.choices[0].message.content)
            else:
                print("Function", func_name, "not found")
    else:
        print('No tool calls returned from model')

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nGoodbye!')