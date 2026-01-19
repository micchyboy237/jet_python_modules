#!/usr/bin/env bash
# =============================================================================
# create_playwright_mcp_demos.sh
# Creates educational demo file structure for Playwright-MCP tool usage
# Run:  bash create_playwright_mcp_demos.sh
# =============================================================================

set -euo pipefail

ROOT_DIR="playwright-mcp-demos"

echo "Creating structure in: ${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/utils"

cd "${ROOT_DIR}" || exit 1

# ──────────────────────────────────────────────────────────────────────────────
#  utils/starting_points.py
# ──────────────────────────────────────────────────────────────────────────────
cat > utils/starting_points.py << 'EOF'
"""Common starting URLs with descriptions for demos"""

from typing import TypedDict, Literal

class StartConfig(TypedDict):
    url: str
    description: str

STARTING_POINTS: dict[Literal[
    "ecommerce_demo",
    "todo_app",
    "practice_form",
    "tables_and_dialogs",
    "complex_ui"
], StartConfig] = {
    "ecommerce_demo": {
        "url": "https://www.saucedemo.com/",
        "description": "Classic simple e-commerce demo — login, product list, cart, checkout"
    },
    "todo_app": {
        "url": "https://todomvc.com/examples/react/dist/",
        "description": "TodoMVC React - very good for form, list, filtering, state interaction"
    },
    "practice_form": {
        "url": "https://demoqa.com/automation-practice-form",
        "description": "Rich practice form — inputs, dropdowns, datepicker, file upload, radio, checkbox"
    },
    "tables_and_dialogs": {
        "url": "https://the-internet.herokuapp.com/",
        "description": "Many small demo pages — tables, dialogs, drag & drop, multiple windows, etc"
    },
    "complex_ui": {
        "url": "https://mui.com/material-ui/getting-started/templates/dashboard/",
        "description": "Modern Material UI dashboard — many interactive components"
    }
}

DEFAULT_START: Literal["ecommerce_demo"] = "ecommerce_demo"
EOF

# ──────────────────────────────────────────────────────────────────────────────
#  utils/__init__.py
# ──────────────────────────────────────────────────────────────────────────────
touch utils/__init__.py

# ──────────────────────────────────────────────────────────────────────────────
#  utils/demo_helpers.py  (small reusable helpers)
# ──────────────────────────────────────────────────────────────────────────────
cat > utils/demo_helpers.py << 'EOF'
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
EOF

# ──────────────────────────────────────────────────────────────────────────────
#  Template for numbered demo files
# ──────────────────────────────────────────────────────────────────────────────
cat > 00_template.py << 'EOF'
"""
TEMPLATE – copy this for new numbered demos
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from fastmcp import Client
from utils.starting_points import STARTING_POINTS, DEFAULT_START
from utils.demo_helpers import step, navigate_to

console = Console()

async def main() -> None:
    console.print(Panel.fit(
        "[bold cyan]DEMO TITLE[/bold cyan]\nWhich tools are shown here",
        border_style="bright_blue"
    ))

    start_cfg = STARTING_POINTS[DEFAULT_START]
    console.print(f"\n[bold]Demo site:[/] {start_cfg['description']}")
    console.print(f" URL: [link={start_cfg['url']}]{start_cfg['url']}[/link]\n")

    client = Client.from_config_file()      # ← adjust if needed

    async with client:
        await navigate_to(client, start_cfg["url"])

        # ── Add your demo steps here ────────────────────────────────────────
        await step(client, "Example step", lambda: None, sleep_after=0.8)

        console.print("\n[bold green]✓ Demo finished[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# ──────────────────────────────────────────────────────────────────────────────
#  Create all numbered demo files (placeholders)
# ──────────────────────────────────────────────────────────────────────────────
for num in {01..10}; do
    title=""
    case $num in
        01) title="Basic Navigation & Browser Control" ;;
        02) title="Interaction Basics (click, type, hover…)" ;;
        03) title="Forms, Inputs & File Upload" ;;
        04) title="Waiting, Synchronization & Network" ;;
        05) title="Content Extraction & Evaluation" ;;
        06) title="Screenshots & Page Snapshots" ;;
        07) title="Tabs, Windows & Frames" ;;
        08) title="Advanced Interaction (drag, double-click…)" ;;
        09) title="JavaScript Execution & Evaluate" ;;
        10) title="Error Handling, Dialogs & Debug" ;;
    esac

    cat > "${num}_$(echo "$title" | tr '[:upper:] ' '[:lower:]_' | tr -s '_' | sed 's/_$//').py" << EOF
"""
${title}
Shows: ...

Recommended starting point: ...
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from fastmcp import Client
from utils.starting_points import STARTING_POINTS, DEFAULT_START
from utils.demo_helpers import step, navigate_to

console = Console()

async def main() -> None:
    console.print(Panel.fit(
        "[bold cyan]${title}[/bold cyan]",
        border_style="bright_blue"
    ))

    start_cfg = STARTING_POINTS[DEFAULT_START]
    console.print(f"\n[bold]Starting point:[/] {start_cfg['description']}")
    console.print(f" URL: [link={start_cfg['url']}]{start_cfg['url']}[/link]\n")

    client = Client.from_config_file()

    async with client:
        await navigate_to(client, start_cfg["url"])

        # TODO: implement demo steps for this topic
        console.print("[yellow]→ Placeholder — add your steps here[/yellow]")
        await asyncio.sleep(1.0)

        console.print("\n[bold green]✓ Demo finished[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
EOF
done

# ──────────────────────────────────────────────────────────────────────────────
#  README.md stub
# ──────────────────────────────────────────────────────────────────────────────
cat > README.md << 'EOF'
# Playwright-MCP Demos

Educational, progressive examples showing how to use **Playwright-MCP** tools.

## Structure

```
01_basic_navigation.py           ← start here
02_interaction_basics.py
...
10_error_handling_and_debug.py
└── utils/
    ├── __init__.py
    ├── demo_helpers.py          ← reusable step(), status, error printing
    └── starting_points.py       ← good demo websites
```

## Recommended order

1. Basic navigation
2. Click / type / hover / select
3. Forms & file upload
4. Waiting strategies
...

## Running a demo

```bash
uv run python 01_basic_navigation.py
# or
poetry run python 01_basic_navigation.py
# or just
python -m asyncio 01_basic_navigation.py
```

Most demos start on https://www.saucedemo.com/ — change in `main()` if desired.

Happy automating!
EOF

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "Done! Created ${ROOT_DIR} with:"
find . -type f | sort | sed 's|^./|  - |'
echo ""
echo "Next step suggestion:"
echo "  cd ${ROOT_DIR}"
echo "  # Start editing 01_basic_navigation.py"
echo ""
