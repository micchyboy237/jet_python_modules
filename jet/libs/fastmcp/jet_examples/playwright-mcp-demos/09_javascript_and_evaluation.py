# File: demos/09_javascript_and_evaluation.py
"""
JavaScript execution & advanced evaluation demo
Tools: browser_run_code, browser_evaluate (complex)
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from utils.base import get_client, get_output_dir, BASE_OUTPUT_DIR
from jet.file.utils import save_file


console = Console()


async def fill_textarea(client, selector: str, value: str):
    # 2025-pattern, robust JS fill function
    escaped = value.replace("'", "\\'").replace("\n", "\\n")

    js = f"""
    (el) => {{
        if (!el) throw new Error('Element not found: ' + '{selector}');
        el.value = '{escaped}';
        el.dispatchEvent(new Event('input', {{bubbles: true}}));
        el.dispatchEvent(new Event('change', {{bubbles: true}}));
        el.dispatchEvent(new Event('blur', {{bubbles: true}}));
        return el.value;
    }}
    """

    return await client.call_tool("browser_evaluate", {
        "selector": selector,
        "function": js
    })


async def main():
    console.print(Panel.fit(
        "[bold cyan]09 - JavaScript & Advanced Evaluation[/bold cyan]",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/text-box"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    client = get_client()
    output_dir = get_output_dir()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})

        # Simple evaluation
        count = await client.call_tool("browser_evaluate", {
            "function": "() => document.querySelectorAll('input').length"
        })
        count_text = str(count.content[0]) if count.content else "—"
        console.print(f"[green]Number of input fields: {count_text}[/green]\n")
        save_file(count_text, f"{BASE_OUTPUT_DIR}/count_text.md")

        # Fill field using JavaScript - more robust version
        console.print("[yellow]Filling Current Address via JavaScript...[/yellow]")

        result = await client.call_tool("browser_evaluate", {
            "function": """
            () => {
                const textarea = document.querySelector('#currentAddress');
                
                if (!textarea) {
                    throw new Error("Could not find textarea with id='currentAddress'");
                }
                
                const value = '123 JavaScript Street\\nSuite 404';
                textarea.value = value;
                
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                textarea.dispatchEvent(new Event('change', { bubbles: true }));
                
                return value;  // optional - for confirmation
            }
            """
        })

        current_address_text = result.content[0].text if result.content else "—"
        console.print(f"[green]✓ Address filled via JS: {current_address_text}[/green]\n")

        address_text = str(result.content[0]) if result.content else "—"
        console.print(f"[green]✓ Address filled via JS: {address_text}[/green]\n")
        save_file(address_text, f"{BASE_OUTPUT_DIR}/current_address_text.md")

        # Optionally show the "2025 style" for bonus usage
        # -- Uncomment this block to show even safer pattern --
        # filled = await fill_textarea(client, "#currentAddress", "123 JavaScript Street\nSuite 404\nApt 13")
        # filled_text = str(filled.content[0]) if filled.content else "—"
        # console.print(f"[green]✓ Address filled using fill_textarea: {filled_text}[/green]")

        # More complex run_code example
        console.print("[yellow]Running Playwright-style code snippet...[/yellow]")
        await client.call_tool("browser_run_code", {
            "code": """
            async (page) => {
                await page.fill('#userName', 'JS Master');
                await page.waitForTimeout(800);
                return await page.title();
            }
            """
        })
        console.print("[green]✓ Complex JS snippet executed[/green]")

        # Full page screenshot
        console.print("[yellow]Taking full page screenshot...[/yellow]")
        await client.call_tool("browser_take_screenshot", {
            "type": "png",
            "filename": f"{output_dir}/screenshot-full.png",
            "fullPage": True
        })
        console.print("[green]✓ Full page screenshot saved[/green]")

    console.print("\n[bold bright_green]JavaScript evaluation demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())