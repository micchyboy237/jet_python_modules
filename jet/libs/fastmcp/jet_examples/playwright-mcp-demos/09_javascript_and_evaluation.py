# File: demos/09_javascript_and_evaluation.py
"""
JavaScript execution & advanced evaluation demo
Tools: browser_run_code, browser_evaluate (complex)
"""

import asyncio
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]09 - JavaScript & Advanced Evaluation[/bold cyan]",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/text-box"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})

        # Simple evaluation
        count = await client.call_tool("browser_evaluate", {
            "function": "() => document.querySelectorAll('input').length"
        })
        console.print(f"[green]Number of input fields:[/] {count.get('result', '?')}\n")

        # Fill field using JavaScript
        console.print("[yellow]Filling Current Address via JavaScript...[/yellow]")
        await client.call_tool("browser_evaluate", {
            "element": "Current Address textarea",
            "function": "(el) => { el.value = '123 JavaScript Street\\nSuite 404'; el.dispatchEvent(new Event('input')); }"
        })
        console.print("[green]✓ Address filled via JS[/green]\n")

        # More complex run_code example
        console.print("[yellow]Running Playwright-style code snippet...[/yellow]")
        await client.call_tool("browser_run_code", {
            "code": """
            async ({page}) => {
                await page.fill('#userName', 'JS Master');
                await page.waitForTimeout(800);
                return await page.title();
            }
            """
        })
        console.print("[green]✓ Complex JS snippet executed[/green]")

    console.print("\n[bold bright_green]JavaScript evaluation demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())