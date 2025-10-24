from openai import OpenAI
from typing import Any
import json

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        int: The sum of the two numbers
    """
    return int(a) + int(b)

def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """
    return int(a) - int(b)

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

available_functions = {
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
}

messages = [{'role': 'user', 'content': 'What is three plus one?'}]
print('Prompt:', messages[0]['content'])

response = client.chat.completions.create(
    model="qwen3-instruct-2507:4b",
    messages=messages,
    tools=tools,
    tool_choice='auto',
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
        else:
            print('Function', func_name, 'not found')

    # Second call to get final response
    final_response = client.chat.completions.create(
        model='llama3.2',
        messages=messages,
    )
    print('Final response:', final_response.choices[0].message.content)
else:
    print('No tool calls returned from model')