# demo_async_agent_local.py
"""
Demonstrations of running synchronous smolagents CodeAgent inside async Starlette apps
using anyio.to_thread.run_sync to avoid blocking the event loop.
Reuses create_local_model() → your local llama.cpp OpenAI-compatible endpoint.
"""

import asyncio
import time

import anyio.to_thread
import httpx
from rich.console import Console
from rich.panel import Panel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from smolagents import CodeAgent, OpenAIModel

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Reuse from previous examples
# ──────────────────────────────────────────────────────────────────────────────

def create_local_model(
    temperature: float = 0.7,
    max_tokens: int | None = 1024,
    model_id: str = "local-model",
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
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

def create_sync_agent(max_steps: int = 6, verbosity_level: int = 1) -> CodeAgent:
    """Creates a simple synchronous CodeAgent using local model."""
    model = create_local_model(temperature=0.65)
    return CodeAgent(
        tools=[],
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Async helpers & Starlette route examples
# ──────────────────────────────────────────────────────────────────────────────

async def async_run_agent(task: str, agent: CodeAgent) -> str:
    """Runs synchronous agent.run() in a background thread."""
    console.print(f"[dim]Starting agent task in background thread: {task[:60]}{'...' if len(task) > 60 else ''}[/dim]")
    start = time.time()
    result = await anyio.to_thread.run_sync(agent.run, task)
    duration = time.time() - start
    console.print(f"[dim]Agent completed in {duration:.1f}s[/dim]")
    return str(result)


# Example Starlette routes

async def simple_agent_endpoint(request: Request) -> Response:
    """POST /run-agent → {"task": "..."} → runs agent synchronously in thread"""
    try:
        data = await request.json()
        task = data.get("task", "").strip()
        if not task:
            return JSONResponse({"error": "Missing or empty 'task'"}, status_code=400)

        agent = create_sync_agent(max_steps=5)
        result = await async_run_agent(task, agent)

        return JSONResponse({"result": result})
    except Exception as e:
        console.print(f"[red]Error in endpoint:[/red] {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def streaming_like_endpoint(request: Request) -> Response:
    """Simulates a longer-running task with basic progress feedback"""
    # For real streaming you'd use ServerSentEvents or WebSockets,
    # but here we just fake progress reporting via console
    try:
        data = await request.json()
        task = data.get("task", "").strip()
        if not task:
            return JSONResponse({"error": "Missing task"}, status_code=400)

        agent = create_sync_agent(max_steps=8, verbosity_level=2)

        console.rule("Long-running agent task started")
        result = await async_run_agent(task, agent)
        console.rule("Task completed", style="green")

        return JSONResponse({"result": result, "status": "completed"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Minimal app factory (used in demos)

def create_minimal_app() -> Starlette:
    return Starlette(routes=[
        Route("/run-agent", simple_agent_endpoint, methods=["POST"]),
        Route("/long-task", streaming_like_endpoint, methods=["POST"]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Demos (console-based simulations of HTTP calls)
# ──────────────────────────────────────────────────────────────────────────────

async def demo_async_1_simple_task():
    """Demo 1: Simple math / reasoning task via async endpoint simulation"""
    console.rule("Demo 1: Simple async agent call", style="blue")

    app = create_minimal_app()
    agent = create_sync_agent(max_steps=4)

    task = "What is the 10th Fibonacci number?"

    console.print(f"\n[bold cyan]Task:[/bold cyan] {task}")
    result = await async_run_agent(task, agent)
    console.print(Panel(result, title="Result", border_style="green"))


async def demo_async_2_via_http_client():
    """Demo 2: Run real HTTP request against in-memory Starlette app"""
    console.rule("Demo 2: Real HTTP POST simulation", style="blue")

    app = create_minimal_app()

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        task = "Explain in one sentence why the sky is blue."

        payload = {"task": task}
        console.print(f"[bold cyan]Sending task:[/bold cyan] {task}")

        try:
            resp = await client.post("/run-agent", json=payload, timeout=90.0)
            resp.raise_for_status()
            data = resp.json()
            console.print(Panel(
                data.get("result", "No result"),
                title=f"Response (status {resp.status_code})",
                border_style="green" if resp.status_code == 200 else "red"
            ))
        except Exception as e:
            console.print(f"[red]Request failed:[/red] {str(e)}")


async def demo_async_3_concurrent_tasks():
    """Demo 3: Run multiple agent tasks concurrently (thread pool handles blocking)"""
    console.rule("Demo 3: Concurrent agent tasks", style="blue")

    tasks = [
        "Calculate 8 factorial",
        "What is the capital of Mongolia?",
        "Convert 100 degrees Celsius to Fahrenheit",
    ]

    console.print(f"\nRunning {len(tasks)} tasks concurrently...\n")

    async def run_one(task: str):
        agent = create_sync_agent(max_steps=5)
        result = await async_run_agent(task, agent)
        return task, result

    start = time.time()
    results = await asyncio.gather(*(run_one(t) for t in tasks))
    duration = time.time() - start

    table = rich.table.Table(title=f"Concurrent Results ({duration:.1f}s total)")
    table.add_column("Task", style="cyan")
    table.add_column("Result", style="green")
    for task, res in results:
        table.add_row(task, str(res)[:120] + "..." if len(str(res)) > 120 else str(res))
    console.print(table)


def main():
    console.rule("Async + smolagents Demos — LOCAL llama.cpp", style="bold magenta")

    console.print(
        "[dim]Shows how to run blocking agent.run() safely inside async Starlette apps[/dim]\n"
    )

    async def run_all():
        await demo_async_1_simple_task()
        # await demo_async_2_via_http_client()      # real HTTP
        # await demo_async_3_concurrent_tasks()     # concurrent

    asyncio.run(run_all())

    console.rule("Done", style="bold green")


if __name__ == "__main__":
    main()