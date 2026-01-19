from rich.console import Console
from ..page_controller import PageController
from ..mcp_client import MCPClient

console = Console()


def run_winning_recipe_demo(url: str = "https://www.google.com"):
    """Demonstrates the core winning recipe pattern"""
    client = MCPClient()
    page = PageController(client)

    try:
        # Step 1
        page.navigate(url)

        # Step 2 + 3
        page.wait_for_text("Google")
        snapshot1 = page.take_snapshot()

        console.rule("First Snapshot")
        console.print(snapshot1[:1200], markup=False)  # truncated for console
        console.rule()

        # Step 4 - Very simple analysis (in real agent → LLM would do this)
        console.print("[bold yellow]Quick analysis:[/bold yellow]")
        if "role: textbox" in snapshot1 and "name: Search" in snapshot1:
            console.print("→ Found search box")

        # Meaningful action
        page.click({"role": "textbox", "name": "Search"})
        page.type_text("playwright mcp accessibility tree 2026")
        page.click({"role": "button", "name": "Google Search"})

        # Step 2+3 again (critical loop)
        page.wait_for_text("results", timeout_ms=20000)
        snapshot2 = page.take_snapshot()

        console.rule("Snapshot after search")
        console.print(snapshot2[:1500], markup=False)

        console.print("\n[bold green]✓ Winning recipe loop completed![/bold green]")

    finally:
        page.close()


if __name__ == "__main__":
    run_winning_recipe_demo()
