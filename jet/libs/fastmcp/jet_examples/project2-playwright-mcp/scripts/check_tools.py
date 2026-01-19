"""Tool check and listing utility for playwright-mcp."""

import asyncio
import yaml
from rich.console import Console

from fastmcp import Client
from utils.args import parse_common_args

console = Console()

async def main():
    args = parse_common_args("Playwright-MCP Tool Check")

    console.print("[bold green]Playwright-MCP Tool Checker[/bold green]")
    console.print("[dim]Configuration used:[/dim]")
    console.print(f"  • Config   : {args.config}\n")

    # Load multi-server config explicitly
    with open(args.config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    client = Client(config_dict)

    async with client:
        try:
            tools = await client.list_tools()
            console.print("\n[bold yellow]Available tools:[/]")
            for tool in tools:
                console.print(f"  • {tool.name}")
        except Exception as e:
            console.print(f"[red]Error during tool listing:[/] {e}")

if __name__ == "__main__":
    asyncio.run(main())