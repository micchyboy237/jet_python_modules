# examples/smolagents_mem0_demo.py
"""
Demonstration of mem0 integration with smolagents.

Shows:
1. Basic agent with mem0 memory
2. Memory-enhanced system prompts
3. Multi-agent with shared memory
4. Searching and retrieving past memories
"""

import logging

from jet.adapters.smolagents.factory import (
    create_code_agent,
    create_llm_model,
    create_mem0_agent_memory,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from smolagents import Tool

console = Console()
logging.basicConfig(level=logging.WARNING)


def create_calculator_tool() -> Tool:
    """Simple calculator tool for demo."""
    from smolagents import tool

    @tool
    def calculator(expression: str) -> str:
        """
        Evaluate a mathematical expression.

        Args:
            expression: Mathematical expression to evaluate
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    return calculator


def demo_basic_agent_with_memory():
    """Demo 1: Basic agent with mem0 memory."""
    console.rule("[bold blue]Demo 1: Agent with Mem0 Memory")

    # Create model
    model = create_llm_model(
        temperature=0.1,
        agent_name="demo_agent",
    )

    # Create tool
    tools = [create_calculator_tool()]

    # Create mem0 memory
    mem0_memory = create_mem0_agent_memory(
        agent_id="demo_agent",
        auto_extract=True,
        auto_store_steps=True,
    )

    # Create agent with memory callback
    agent = create_code_agent(
        tools=tools,
        model=model,
        step_callbacks=mem0_memory.create_step_callbacks_dict(),
    )

    console.print("[green]Agent created with mem0 memory[/green]")

    # Run some tasks
    tasks = [
        "Calculate 15 * 23",
        "What is 100 / 4?",
    ]

    for task in tasks:
        console.print(f"\n[bold yellow]Task:[/bold yellow] {task}")
        result = agent.run(task)
        console.print(f"[green]Result:[/green] {result}")

    # Show stored memories
    console.print("\n[bold cyan]Stored Memories:[/bold cyan]")
    all_memories = mem0_memory.get_all()

    table = Table(title="Agent Memories", show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Memory", style="white")
    table.add_column("Type", style="cyan")

    for mem in all_memories.get("results", []):
        table.add_row(
            mem.get("id", "")[:8] + "...",
            str(mem.get("memory", ""))[:100],
            mem.get("metadata", {}).get("step_type", "unknown"),
        )

    console.print(table)

    # Cleanup
    mem0_memory.reset()
    mem0_memory.close()


def demo_memory_enhanced_prompt():
    """Demo 2: Memory-enhanced system prompts."""
    console.rule("[bold blue]Demo 2: Memory-Enhanced System Prompts")

    mem0_memory = create_mem0_agent_memory(
        agent_id="enhanced_agent",
        auto_extract=True,
    )

    # Pre-load some knowledge
    preload_memories = [
        "The user prefers concise answers with code examples.",
        "The user's name is Alice and she works with Python daily.",
        "Alice is building an AI agent system using smolagents and mem0.",
    ]

    for mem in preload_memories:
        mem0_memory.add_memory(mem, infer=False)
        console.print(f"  [dim]Preloaded: {mem[:60]}...[/dim]")

    # Enhance a system prompt
    base_prompt = "You are a helpful AI assistant."
    task = "Help Alice with her Python AI agent project"

    enhanced = mem0_memory.enhance_system_prompt(base_prompt, task)

    console.print("\n[bold yellow]Base Prompt:[/bold yellow]")
    console.print(base_prompt)

    console.print("\n[bold green]Enhanced Prompt:[/bold green]")
    console.print(Panel(enhanced, title="Enhanced System Prompt"))

    # Search memories
    console.print("\n[bold cyan]Search: 'Alice Python'[/bold cyan]")
    results = mem0_memory.search("Alice Python", top_k=3)

    for mem in results.get("results", []):
        console.print(f"  [dim][{mem['score']:.2f}][/dim] {mem['memory'][:80]}")

    mem0_memory.reset()
    mem0_memory.close()


def demo_multi_agent_shared_memory():
    """Demo 3: Multiple agents sharing memory."""
    console.rule("[bold blue]Demo 3: Multi-Agent Shared Memory")

    # Shared mem0 memory for all agents
    shared_memory = create_mem0_agent_memory(
        agent_id="agent_team",
        auto_extract=True,
    )

    shared_memory.set_run_id("team_run_001")

    # Agent 1: Research agent
    research_model = create_llm_model(
        temperature=0.1,
        agent_name="researcher",
    )
    research_agent = create_code_agent(
        tools=[create_calculator_tool()],
        model=research_model,
        step_callbacks=shared_memory.create_step_callbacks_dict(),
        name="researcher",
        description="Research agent that gathers information",
    )

    # Store research findings
    shared_memory.add_memory(
        "Research finding: Python 3.12 has significant performance improvements.",
        infer=False,
        metadata={"agent": "researcher", "step_type": "research"},
    )

    console.print("[green]Research agent stored findings[/green]")

    # Agent 2: Analyst agent (can access shared memory)
    analyst_model = create_llm_model(
        temperature=0.1,
        agent_name="analyst",
    )

    # Analyst searches shared memory
    context = shared_memory.get_relevant_context(
        "Python performance improvements",
        top_k=3,
    )

    console.print("\n[bold cyan]Analyst retrieves context:[/bold cyan]")
    console.print(Panel(context, title="Shared Memory Context"))

    shared_memory.reset()
    shared_memory.close()


if __name__ == "__main__":
    console.print("\n[bold]🚀 SmoLagents + Mem0 Integration Demo[/bold]\n")

    try:
        demo_basic_agent_with_memory()
    except Exception as e:
        console.print(f"[red]Demo 1 failed: {e}[/red]")

    console.print("\n" + "=" * 60 + "\n")

    try:
        demo_memory_enhanced_prompt()
    except Exception as e:
        console.print(f"[red]Demo 2 failed: {e}[/red]")

    console.print("\n" + "=" * 60 + "\n")

    try:
        demo_multi_agent_shared_memory()
    except Exception as e:
        console.print(f"[red]Demo 3 failed: {e}[/red]")

    console.rule("[bold green]✨ Demo Complete")
