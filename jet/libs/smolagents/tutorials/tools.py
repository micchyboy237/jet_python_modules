# demo_advanced_tools_local.py
"""
Advanced tool usage demos with smolagents – reusing create_local_model()
Shows custom tools, dynamic toolbox, Hub loading, Gradio Spaces, LangChain, etc.
"""

from typing import Optional, Any

from rich.console import Console
from rich.panel import Panel

from smolagents import (
    Tool,
    CodeAgent,
    load_tool,
    # Tool.from_space,
    # Tool.from_langchain,
)

# Optional – uncomment if you want to try MCP or LangChain tools
# from mcp import StdioServerParameters
# from langchain_community.tools import DuckDuckGoSearchRun

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Reuse your local model factory
# ──────────────────────────────────────────────────────────────────────────────

def create_local_model(
    temperature: float = 0.7,
    max_tokens: Optional[int] = 1024,
    model_id: str = "local-model",
) -> Any:
    """Default factory – your local llama.cpp OpenAI-compatible endpoint."""
    from smolagents import OpenAIModel

    return OpenAIModel(
        model_id=model_id,
        base_url="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reusable agent factory
# ──────────────────────────────────────────────────────────────────────────────

def create_tool_demo_agent(
    tools: list = None,
    max_steps: int = 8,
    verbosity_level: int = 1,
) -> CodeAgent:
    """Creates a simple CodeAgent with local model + provided tools."""
    model = create_local_model(temperature=0.65)
    return CodeAgent(
        tools=tools or [],
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demo 1 – Custom tool by subclassing Tool
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMathTool(Tool):
    name = "simple_math"
    description = "Performs basic arithmetic operations on two numbers."
    inputs = {
        "operation": {"type": "string", "description": "One of: add, subtract, multiply, divide"},
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"},
    }
    output_type = "number"

    def forward(self, operation: str, a: float, b: float) -> float:
        op = operation.lower()
        if op == "add":
            return a + b
        elif op == "subtract":
            return a - b
        elif op == "multiply":
            return a * b
        elif op == "divide":
            return a / b if b != 0 else float("inf")
        else:
            raise ValueError(f"Unknown operation: {operation}")


def demo_1_custom_subclass_tool():
    """Demo 1: Create and use custom tool by subclassing Tool"""
    console.rule("Demo 1 – Custom subclassed tool", style="blue")

    math_tool = SimpleMathTool()
    agent = create_tool_demo_agent(tools=[math_tool], max_steps=5)

    task = "What is 17 multiplied by 42?"
    console.print(f"\n[bold cyan]Task:[/bold cyan] {task}")

    try:
        result = agent.run(task)
        console.print(Panel(result, title="Final Answer", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 2 – Dynamic toolbox management
# ──────────────────────────────────────────────────────────────────────────────

def demo_2_dynamic_toolbox():
    """Demo 2: Add / replace tools at runtime"""
    console.rule("Demo 2 – Dynamic toolbox modification", style="blue")

    agent = create_tool_demo_agent(max_steps=5)

    # Start empty
    console.print("[dim]Initial tools:[/dim]", list(agent.tools.keys()))

    # Add custom tool
    math_tool = SimpleMathTool()
    agent.tools[math_tool.name] = math_tool

    console.print("[dim]After adding simple_math:[/dim]", list(agent.tools.keys()))

    task = "Compute 123 + 456 using the math tool."
    console.print(f"\n[bold cyan]Task:[/bold cyan] {task}")

    try:
        result = agent.run(task)
        console.print(Panel(result, title="Final Answer", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 3 – Load tool from Hugging Face Hub Space
# ──────────────────────────────────────────────────────────────────────────────

def demo_3_load_tool_from_hub():
    """Demo 3: Load a public tool from HF Hub Space (trust_remote_code=True)"""
    console.rule("Demo 3 – Load tool from Hub Space", style="blue")

    console.print("[yellow]WARNING:[/yellow] Requires trust_remote_code=True – only use trusted repos!")

    try:
        # Example public tool – you can replace with your own
        tool_id = "m-ric/hf-model-downloads"  # from the doc example
        hf_tool = load_tool(tool_id, trust_remote_code=True)

        agent = create_tool_demo_agent(tools=[hf_tool], max_steps=6)

        task = "Which model has the most downloads in text-classification?"
        console.print(f"\n[bold cyan]Task:[/bold cyan] {task}")

        result = agent.run(task)
        console.print(Panel(result, title="Final Answer", border_style="green"))

    except Exception as e:
        console.print(f"[yellow]Could not load tool:[/yellow] {str(e)}")
        console.print("[dim]Try a public tool or set trust_remote_code=True for known repos[/dim]")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 4 – Import Gradio Space as tool
# ──────────────────────────────────────────────────────────────────────────────

def demo_4_gradio_space_tool():
    """Demo 4: Import a Gradio Space directly as a tool"""
    console.rule("Demo 4 – Gradio Space as tool", style="blue")

    console.print("[dim]Example: using a simple public text-to-image or similar Space[/dim]")

    try:
        # Choose a small, fast public Space (replace if needed)
        space_id = "stabilityai/stable-diffusion"  # or any fast public Space
        image_tool = Tool.from_space(
            space_id,
            name="txt2img",
            description="Generate an image from a text prompt",
        )

        agent = create_tool_demo_agent(tools=[image_tool], max_steps=6)

        task = "Generate an image of a futuristic city at night."
        console.print(f"\n[bold cyan]Task:[/bold cyan] {task}")

        result = agent.run(task)
        console.print(Panel(str(result), title="Result (image path or description)", border_style="green"))

    except Exception as e:
        console.print(f"[yellow]Could not load Space:[/yellow] {str(e)}")
        console.print("[dim]Many Spaces require GPU / long startup – try a fast one[/dim]")


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def main():
    console.rule("Advanced Tools Demos – smolagents + LOCAL llama.cpp", style="bold magenta")

    console.print(
        "[dim]Showing different ways to create, load and manage tools[/dim]\n"
        "[yellow]Security note:[/yellow] Only load tools / Spaces / MCP servers you trust!\n"
    )

    demos = [
        ("1", "Custom tool subclass", demo_1_custom_subclass_tool),
        ("2", "Dynamic toolbox", demo_2_dynamic_toolbox),
        ("3", "Load from Hub Space", demo_3_load_tool_from_hub),
        ("4", "Gradio Space as tool", demo_4_gradio_space_tool),
    ]

    table = rich.table.Table(title="Available Demos")
    table.add_column("No", style="cyan")
    table.add_column("Description", style="magenta")
    for num, desc, _ in demos:
        table.add_row(num, desc)
    console.print(table)
    console.print()

    # Run selected demos (uncomment what you want)
    demo_1_custom_subclass_tool()
    demo_2_dynamic_toolbox()
    # demo_3_load_tool_from_hub()        # requires trust_remote_code
    # demo_4_gradio_space_tool()         # may take time / fail if Space slow

    console.rule("Done", style="bold green")


if __name__ == "__main__":
    main()