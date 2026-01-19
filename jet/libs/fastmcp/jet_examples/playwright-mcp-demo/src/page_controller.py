from typing import Optional
from rich.console import Console
from .mcp_client import MCPClient

console = Console()


class PageController:
    """Higher level interface following the winning recipe pattern"""

    def __init__(self, client: MCPClient):
        self.client = client

    def navigate(self, url: str) -> None:
        console.print(f"[cyan]→ Navigating to[/cyan] {url}")
        self.client.call("browser_navigate", {"url": url})

    def wait_for_text(self, text: str, timeout_ms: int = 15000) -> bool:
        console.print(f"[yellow]Waiting for text:[/yellow] '{text}' …")
        try:
            self.client.call(
                "browser_wait_for",
                {"type": "text", "value": text, "timeout": timeout_ms},
            )
            console.print("[green]✓ Found![/green]")
            return True
        except Exception as e:
            console.print(f"[red]✗ Wait failed[/red] — {e}")
            return False

    def take_snapshot(self) -> str:
        console.print("[blue]Taking accessibility snapshot...[/blue]")
        snapshot = self.client.call("browser_snapshot")
        console.print("[dim]Snapshot length:[/dim]", len(snapshot))
        return snapshot

    def click(self, selector: dict) -> None:
        """selector example: {"role": "button", "name": "Submit"}"""
        console.print(f"[magenta]Clicking[/magenta] {selector}")
        self.client.call("browser_click", selector)

    def type_text(self, text: str) -> None:
        console.print(f"[magenta]Typing:[/magenta] {text!r}")
        self.client.call("browser_type", {"text": text})

    def close(self) -> None:
        console.print("[dim]Closing browser...[/dim]")
        self.client.call("browser_close")
