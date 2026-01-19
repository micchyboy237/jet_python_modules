"""Common helpers for clean, readable MCP demos"""

import asyncio
from typing import Awaitable, Callable, Any
from rich.console import Console
from rich.status import Status
from fastmcp import Client

console = Console()

async def step(
    client: Client,
    message: str,
    action: Callable[[], Awaitable[Any]],
    sleep_after: float = 1.2
) -> None:
    """Run one demo step with status + optional pause"""
    with console.status(f"[bold cyan]{message}[/bold cyan]", spinner="dots"):
        try:
            result = await action()
            console.print(f"[green]✓ {message} — done[/green]")
            if result is not None and not isinstance(result, (bool, type(None))):
                console.print(f"  → Result: {result}", style="dim")
        except Exception as exc:
            console.print(f"[red]✗ {message} failed[/red]")
            console.print(f"  → {type(exc).__name__}: {exc}", style="red")
            raise
    if sleep_after > 0:
        await asyncio.sleep(sleep_after)

async def navigate_to(client: Client, url: str, label: str | None = None) -> None:
    name = label or url.split("//")[-1].split("/")[0]
    await step(
        client,
        f"Navigate to {name}",
        lambda: client.call_tool("browser_navigate", {"url": url})
    )
