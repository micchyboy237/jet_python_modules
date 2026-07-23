# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

import argparse
import json
import math
from typing import Any, Callable

from jet.adapters.llama_cpp.llm_utils import ChatMessage, chat_with_tools

# --- Tool definitions ---
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt(), sin(), cos(), log().",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4' or 'sqrt(16) + cos(0)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a city (simulated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'New York'",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


# --- Tool implementations ---
def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
    }
    try:
        # Compile for safety — only allow allowed names and basic operations
        code = compile(expression, "<calc>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                return json.dumps(
                    {
                        "error": f"Function '{name}' is not allowed. Use: {', '.join(allowed_names.keys())}"
                    }
                )
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


def get_current_weather(city: str) -> str:
    """Simulate weather lookup."""
    weather_data = {
        "tokyo": {"temp": 28, "condition": "sunny", "humidity": 55},
        "new york": {"temp": 22, "condition": "partly cloudy", "humidity": 60},
        "london": {"temp": 16, "condition": "rainy", "humidity": 80},
        "manila": {"temp": 32, "condition": "thunderstorms", "humidity": 85},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return json.dumps(weather_data[city_lower])
    return json.dumps(
        {
            "temp": 20,
            "condition": "unknown",
            "humidity": 50,
            "note": f"Simulated data for '{city}'",
        }
    )


available_functions: dict[str, Callable[..., Any]] = {
    "calculate": calculate,
    "get_current_weather": get_current_weather,
}

# --- CLI setup ---
parser = argparse.ArgumentParser(
    description="Chat demo with tool calling using standalone llm_utils functions"
)
parser.add_argument(
    "prompt",
    type=str,
    nargs="?",
    default="What's the weather in Tokyo and what's 15% of 250?",
    help="User input prompt (default: %(default)s)",
)
parser.add_argument(
    "-s",
    "--system",
    type=str,
    default="You are a helpful assistant with access to a calculator and weather lookup tools. "
    "Use them whenever the user asks about math or weather. "
    "Always explain the tool results in a friendly way.",
    help="Optional system prompt for the chat model",
)
parser.add_argument(
    "--no-stream",
    action="store_true",
    help="Disable streaming output",
)
args = parser.parse_args()

messages: list[ChatMessage] = [
    {"role": "system", "content": args.system},
    {"role": "user", "content": args.prompt},
]

stream = not args.no_stream

print(f"{'=' * 60}")
print(f"System: {args.system}")
print(f"User: {args.prompt}")
print(f"Stream: {stream}")
print(f"{'=' * 60}")
print("Assistant: ", end="", flush=True)

# Use standalone tool-calling function — no context needed
response = chat_with_tools(
    messages=messages,
    tools=TOOLS,
    available_functions=available_functions,
    stream=stream,
    temperature=0.0,
)

if stream:
    for token in response:
        pass  # verbose logger handles printing
else:
    print(response)

print(f"\n{'=' * 60}")
print("Demo complete.")
