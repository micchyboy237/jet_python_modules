# examples.py
# Usage examples for custom_models.py (OpenAIModel with llama.cpp server)

from pathlib import Path

from jet.libs.smolagents.custom_models import (
    CODEAGENT_RESPONSE_FORMAT,
    ChatMessage,
    MessageRole,
    OpenAIModel,
    TokenUsage,
    Tool,
)
from rich import print as rprint
from rich.console import Console
from rich.table import Table

console = Console()


# ------------------------------------------------------------------------
# Example 1: Basic text generation (no tools, no streaming)
# ------------------------------------------------------------------------
def example_basic_generation():
    model = OpenAIModel(
        model_id="qwen3-instruct-2507:4b",
        temperature=0.7,
        max_tokens=300,
        verbose=True,
    )

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM, content="You are a helpful coding assistant."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Write a Python function to reverse a string.",
        ),
    ]

    response = model.generate(messages)

    rprint("[bold green]Basic Generation Result:[/bold green]")
    rprint(f"Content: {response.content}")
    rprint(f"Tokens used: {response.token_usage}")


# ------------------------------------------------------------------------
# Example 2: Streaming response with rich progress bar
# ------------------------------------------------------------------------
def example_streaming():
    model = OpenAIModel(
        model_id="qwen3-instruct-2507:4b",
        temperature=0.9,
        max_tokens=500,
    )

    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Tell me a short story about a robot learning to dance.",
        ),
    ]

    rprint("[bold cyan]Streaming Response:[/bold cyan]")
    full_content = ""
    token_usage = TokenUsage(input_tokens=0, output_tokens=0)

    for delta in model.generate_stream(messages):
        if delta.content:
            console.print(delta.content, end="", style="white")
            full_content += delta.content
        if delta.token_usage:
            token_usage = delta.token_usage  # last one wins (cumulative)

    console.print("\n")
    rprint(
        f"[dim]Total tokens: input={token_usage.input_tokens}, output={token_usage.output_tokens}[/dim]"
    )


# ------------------------------------------------------------------------
# Example 3: Tool calling (model decides when to use tools)
# ------------------------------------------------------------------------
def example_tool_calling():
    # Define a simple tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        # Mock response
        return f"The weather in {city} is sunny, 28°C."

    weather_tool = Tool(
        name="get_weather",
        description="Get current weather for any city.",
        inputs={"city": {"type": "string", "description": "City name, e.g. Tokyo"}},
    )

    model = OpenAIModel(
        model_id="qwen3-instruct-2507:4b",
        temperature=0.3,
    )

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a weather assistant. Use tools when needed.",
        ),
        ChatMessage(
            role=MessageRole.USER, content="What's the weather like in Paris today?"
        ),
    ]

    response = model.generate(
        messages=messages,
        tools_to_call_from=[weather_tool],
        tool_choice="required",  # force tool use
    )

    rprint("[bold yellow]Tool Calling Result:[/bold yellow]")
    if response.tool_calls:
        for call in response.tool_calls:
            rprint(f"→ Calling tool: {call.function.name}")
            rprint(f"  Arguments: {call.function.arguments}")
    else:
        rprint("No tool call made.")
        rprint(response.content)


# ------------------------------------------------------------------------
# Example 4: Logging enabled + JSON response format
# ------------------------------------------------------------------------
def example_with_logging_and_json():
    logs_dir = Path("llm_logs/example_run")
    logs_dir.mkdir(parents=True, exist_ok=True)

    model = OpenAIModel(
        model_id="qwen3-instruct-2507:4b",
        logs_dir=str(logs_dir),
        response_format=CODEAGENT_RESPONSE_FORMAT,  # forces {"thought": ..., "code": ...}
    )

    messages = [
        ChatMessage(
            role=MessageRole.USER, content="Write a function that calculates factorial."
        ),
    ]

    response = model.generate(messages)

    rprint("[bold magenta]JSON Structured Response (logged):[/bold magenta]")
    rprint(response.content)  # should be valid JSON

    # Show where logs were saved
    latest_call = sorted(logs_dir.glob("generate/llm_call_*"))[-1]
    rprint(f"Logs saved in: {latest_call}")


# ------------------------------------------------------------------------
# Example 5: Quick table of model stats (rich output)
# ------------------------------------------------------------------------
def example_model_info_table():
    model = OpenAIModel(
        model_id="qwen3-instruct-2507:4b",
    )

    table = Table(title="Model Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model ID", model.model_id)
    table.add_row("API Base", model.client.base_url)
    table.add_row("Logs Directory", str(model.logs_dir))
    table.add_row("Supports Stop", str(model.supports_stop_parameter))
    table.add_row("Flatten Messages", str(model.flatten_messages_as_text))

    console.print(table)


if __name__ == "__main__":
    console.rule("Custom Models Usage Examples")

    examples = [
        example_basic_generation,
        example_streaming,
        example_tool_calling,
        example_with_logging_and_json,
        example_model_info_table,
    ]

    for i, example in enumerate(examples, 1):
        console.rule(f"Example {i}", style="blue")
        example()
        console.print("\n")
