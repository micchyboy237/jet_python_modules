"""Tool check and listing utility for playwright-mcp with rich logging."""

import asyncio
from mcp.types import Tool
import yaml

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from fastmcp import Client  # assuming Tool is imported from your models
from utils.args import parse_common_args

console = Console()

def print_tool_info(tool: Tool) -> None:
    """Print richly formatted information about a single tool."""
    
    # Header with tool name
    title = Text.assemble(
        ("Tool: ", "bold cyan"),
        (tool.name, "bold white on dark_cyan")
    )
    
    console.print(Panel(
        title,
        expand=False,
        border_style="bright_blue",
        padding=(0, 2)
    ))

    # Description
    if tool.description:
        console.print(Text("Description:", style="bold italic magenta"))
        console.print(f"  {tool.description}\n", style="dim")
    else:
        console.print("  [italic grey]No description provided[/italic grey]\n")

    # Quick status line
    status_parts = []
    if tool.inputSchema:
        status_parts.append(f"[green]✓ input schema[/green] ({len(tool.inputSchema.get('properties', {}))} params)")
    else:
        status_parts.append("[red]✗ no input schema[/red]")

    if tool.outputSchema:
        status_parts.append("[green]✓ output schema[/green]")
    if tool.execution:
        status_parts.append("[green]✓ has execution[/green]")
    if tool.icons:
        status_parts.append(f"[blue]icons: {len(tool.icons)}[/blue]")
    if tool.annotations:
        status_parts.append("[blue]has annotations[/blue]")
    if tool.meta:
        status_parts.append("[magenta]has _meta[/magenta]")

    console.print(" • " + "  •  ".join(status_parts) + "\n", highlight=False)

    # Parameters summary (nice table)
    if tool.inputSchema and isinstance(tool.inputSchema, dict):
        props = tool.inputSchema.get("properties", {})
        if props:
            table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED, border_style="dim")
            table.add_column("Parameter", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Required", justify="center")
            table.add_column("Description", style="dim")

            for name, schema in props.items():
                param_type = schema.get("type", "—")
                if isinstance(param_type, list):
                    param_type = " | ".join(param_type)
                required = "Yes" if name in tool.inputSchema.get("required", []) else "—"
                desc = schema.get("description", "—")[:120]

                table.add_row(name, str(param_type), required, desc)

            console.print(table)
            console.print("")

    # Very verbose mode - full schemas (optional, toggleable later)
    # if args.verbose:
    #     console.print(Panel(Pretty(tool.inputSchema), title="Input Schema", border_style="dim blue"))
    #     if tool.outputSchema:
    #         console.print(Panel(Pretty(tool.outputSchema), title="Output Schema", border_style="dim green"))


async def main():
    args = parse_common_args("Playwright-MCP Rich Tool Checker")

    console.print(
        Panel(
            "[bold bright_green]Playwright-MCP Tool Inspector[/bold bright_green]\n"
            "[dim italic]Enhanced tool information viewer[/dim italic]",
            expand=False,
            border_style="bright_green",
            padding=(1, 2)
        )
    )

    console.print("[dim]Configuration used:[/dim]")
    console.print(f"  • Config   : [cyan]{args.config}[/cyan]")
    console.print(f"  • Timestamp: [dim]{asyncio.get_event_loop().time():.2f}[/dim]\n")

    # Load config
    try:
        with open(args.config, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]Failed to load config[/bold red]  {e}")
        return

    client = Client(config_dict)

    async with client:
        try:
            tools = await client.list_tools()
            
            console.print(f"\n[bold yellow]Found {len(tools)} available tool{'s' if len(tools) != 1 else ''}:[/bold yellow]\n")

            for i, tool in enumerate(tools, 1):
                print_tool_info(tool)
                
                # Separator between tools (except last one)
                if i < len(tools):
                    console.print("─" * 80, style="dim")

        except Exception as e:
            console.print("\n[bold red]Error during tool listing:[/bold red]")
            console.print(f"  {e}", style="red")

        finally:
            console.print("\n[grey50]Inspection completed.[/grey50]")


if __name__ == "__main__":
    asyncio.run(main())