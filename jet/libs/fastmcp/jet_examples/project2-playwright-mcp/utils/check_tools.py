"""Tool check and listing utility for playwright-mcp with rich logging."""

import json
import asyncio
import shutil
import yaml

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from mcp.types import Tool
from fastmcp import Client

# ────────────── New Imports for Output & Utility ──────────────
from pathlib import Path
from typing import List
import os


def get_entry_file_dir() -> Path:
    return Path(__file__).parent.resolve()

def get_entry_file_name() -> str:
    return Path(__file__).name


BASE_OUTPUT_DIR = Path(get_entry_file_dir()) / "generated" / os.path.splitext(get_entry_file_name())[0]
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)

console = Console()


async def get_tools(client: Client) -> list[Tool]:
    """
    Fetch all available tools and return them as a list of Tool objects.

    Note: These objects need to be converted with `model_dump` if you wish to export 
    them in a JSON-serializable format.
    """
    raw_tools = await client.list_tools()
    return raw_tools


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

    # ────────────────────────────────────────────────
    #  All fields overview
    # ────────────────────────────────────────────────
    fields_table = Table(show_header=False, box=box.SIMPLE, border_style="dim", padding=(0,1))
    fields_table.add_column("Field", style="bold bright_black", width=16)
    fields_table.add_column("Status / Summary", style="white")

    def add_field_status(name: str, value, summary: str | None = None):
        if value is None:
            fields_table.add_row(name, "[grey50]None[/grey50]")
        elif value == [] or value == {}:
            fields_table.add_row(name, "[grey50]empty[/grey50]")
        else:
            count = len(value) if isinstance(value, (list, dict)) else "—"
            fields_table.add_row(name, f"[green]✓ {summary or type(value).__name__}[/green]" + (f" ({count})" if count != "—" else ""))

    add_field_status("description", tool.description, "text")
    add_field_status("inputSchema", tool.inputSchema, "schema")
    add_field_status("outputSchema", tool.outputSchema)
    add_field_status("icons", tool.icons)
    add_field_status("annotations", tool.annotations)
    add_field_status("meta / _meta", tool.meta)
    add_field_status("execution", tool.execution)

    console.print(fields_table)
    console.print("")

    # Description (full)
    if tool.description:
        console.print(Text("Description:", style="bold italic magenta"))
        console.print(f"  {tool.description}\n", style="dim white")

    # ── Input Parameters Table ────────────────────────
    if tool.inputSchema and isinstance(tool.inputSchema, dict):
        props = tool.inputSchema.get("properties", {})
        if props:
            table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED, border_style="dim")
            table.add_column("Parameter", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Required", justify="center")
            table.add_column("Description", style="dim")

            required_set = set(tool.inputSchema.get("required", []))

            for name, schema in sorted(props.items()):
                param_type = schema.get("type", "—")
                if isinstance(param_type, list):
                    param_type = " | ".join(str(t) for t in param_type)
                required = "Yes" if name in required_set else "—"
                desc = (schema.get("description") or "—")[:140]

                table.add_row(name, str(param_type), required, desc)

            console.print(Panel(table, title="Input Parameters", border_style="blue", padding=(1,1)))
            console.print("")

    # Optional: Show output schema summary / existence
    if tool.outputSchema:
        console.print("[bold green]→ Has output schema defined[/bold green]")
        # You can add Pretty(tool.outputSchema) later when verbose mode is added

    # Extra rich info for interesting fields
    if tool.icons:
        console.print("[blue]Icons:[/blue]", end=" ")
        console.print(", ".join(f"[dim]{getattr(i, 'name', None) or getattr(i, 'type', '?')}[/dim]" for i in tool.icons[:3]) + (" ..." if len(tool.icons) > 3 else ""))

    if tool.annotations:
        cat = getattr(tool.annotations, "category", None)
        console.print(f"[blue]Annotations present[/blue]  (category: {cat or '?'})")

    if tool.meta:
        console.print("[magenta]Has _meta data[/magenta]")

    if tool.execution:
        console.print("[green]Has execution configuration[/green]")


import argparse
import os
from pathlib import Path
from dotenv import load_dotenv


# ── Project root detection (robust, works from any subdirectory) ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # utils/ → project2-playwright-mcp/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mcp-config.yaml"


def parse_common_args(
    description: str,
) -> argparse.Namespace:
    """
    Create and parse common command-line arguments for Playwright-MCP clients.

    Args:
        description: Description text shown in --help
        add_extra_args_callback: Optional function that receives parser
                                 and can add script-specific arguments

    Returns:
        Parsed arguments namespace
    """
    # Load .env from project root if exists
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        default=os.getenv("MCP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)),
        help="Path to MCP configuration file",
    )

    return parser.parse_args()


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

    # Prepare output directory
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = BASE_OUTPUT_DIR / "tools.json"
    all_tools: List[dict] = []   # type hint for clarity

    client = Client(config_dict)

    async with client:
        try:
            tools = await client.list_tools()
            
            console.print(f"\n[bold yellow]Found {len(tools)} available tool{'s' if len(tools) != 1 else ''}:[/bold yellow]\n")

            for i, tool in enumerate(tools, 1):
                all_tools.append(tool.model_dump(mode="json", by_alias=True))
                print_tool_info(tool)
                
                if i < len(tools):
                    console.print("─" * 90, style="dim")

            # ── Save collected tools at the end ───────────────────────────────
            if all_tools:
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(all_tools, f, indent=2, ensure_ascii=False)
                    console.print(f"[green]✓ Saved {len(all_tools)} tools →[/green]\n[white]{json_path}[/white]")
                except Exception as save_err:
                    console.print(f"[bold red]Failed to save tools JSON:[/bold red] {save_err}", style="red")
            else:
                console.print("[yellow]No tools were collected → skipping JSON save[/yellow]")

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}", style="red")

        finally:
            console.print("\n[grey50]Inspection completed.[/grey50]")


if __name__ == "__main__":
    asyncio.run(main())