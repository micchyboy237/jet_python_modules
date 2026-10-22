"""
Playwright JavaScript Evaluator
Executes JavaScript expressions on live URLs or local HTML files using Playwright.
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from rich.console import Console

from playwright.async_api import async_playwright

console = Console()


def parse_arg_value(value_str: str) -> Any:
    """Try to parse a string argument as JSON, otherwise return as string."""
    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        return value_str


async def _evaluate_on_url(
    url: str,
    expression: str,
    args: Optional[List[Any]] = None,
    timeout: float = 30000,
    headless: bool = False,
) -> Any:
    """
    Evaluate a JavaScript expression on a webpage from a URL.
    """
    console.print(f"[bold blue]Starting evaluation for URL:[/bold blue] {url}")
    console.print(f"[dim]Expression: {expression}[/dim]")
    if args:
        console.print(f"[dim]Arguments: {args}[/dim]")

    async with async_playwright() as p:
        console.print("[bold blue]Launching browser...[/bold blue]")
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
            await page.goto(url, timeout=timeout)
            console.print("[green]Page loaded successfully[/green]")

            await page.wait_for_load_state("networkidle", timeout=timeout)
            console.print("[dim]Page network idle state reached[/dim]")

            console.print("[bold blue]Executing JavaScript...[/bold blue]")

            if args:
                # If multiple args, pass as list, else single value
                js_arg = args[0] if len(args) == 1 else args
                result = await page.evaluate(expression, js_arg)
            else:
                result = await page.evaluate(expression)

            console.print("[green]Evaluation completed successfully[/green]")
            return result

        except Exception as e:
            console.print(f"[bold red]Error during evaluation:[/bold red] {e}")
            raise
        finally:
            console.print("[bold blue]Closing browser...[/bold blue]")
            await browser.close()
            console.print("[green]Browser closed[/green]")


async def _evaluate_on_html(
    html_string: str,
    expression: str,
    args: Optional[List[Any]] = None,
    timeout: float = 30000,
    headless: bool = False,
) -> Any:
    """
    Evaluate a JavaScript expression on an HTML string.
    """
    console.print("[bold blue]Starting evaluation from HTML string[/bold blue]")
    console.print(f"[dim]Expression: {expression}[/dim]")
    if args:
        console.print(f"[dim]Arguments: {args}[/dim]")

    async with async_playwright() as p:
        console.print("[bold blue]Launching browser...[/bold blue]")
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            console.print("[bold blue]Loading HTML content...[/bold blue]")
            await page.set_content(html_string)
            console.print("[green]HTML content loaded successfully[/green]")

            await page.wait_for_load_state("networkidle", timeout=timeout)
            console.print("[dim]Page network idle state reached[/dim]")

            console.print("[bold blue]Executing JavaScript...[/bold blue]")

            if args:
                js_arg = args[0] if len(args) == 1 else args
                result = await page.evaluate(expression, js_arg)
            else:
                result = await page.evaluate(expression)

            console.print("[green]Evaluation completed successfully[/green]")
            return result

        except Exception as e:
            console.print(f"[bold red]Error during evaluation:[/bold red] {e}")
            raise
        finally:
            console.print("[bold blue]Closing browser...[/bold blue]")
            await browser.close()
            console.print("[green]Browser closed[/green]")


async def evaluate(
    source: str | Path,
    expression: str,
    args: Optional[List[Any]] = None,
    timeout: float = 30000,
    headless: bool = False,
) -> Any:
    """
    Dynamically detect if source is a file or URL and evaluate JS.

    Args:
        source: URL string or Path to an HTML file.
        expression: JavaScript expression/function to evaluate.
        args: Arguments to pass to the function.
        timeout: Maximum time in milliseconds.
        headless: Run in headless mode.

    Returns:
        The result of the evaluation.
    """
    source_path = Path(source)
    if source_path.is_file():
        console.print(f"[dim]Detected local file: {source_path}[/dim]")
        with open(source_path, "r", encoding="utf-8") as f:
            html = f.read()
        return await _evaluate_on_html(
            html_string=html,
            expression=expression,
            args=args,
            timeout=timeout,
            headless=headless,
        )
    else:
        console.print(f"[dim]Detected URL: {source}[/dim]")
        return await _evaluate_on_url(
            url=str(source),
            expression=expression,
            args=args,
            timeout=timeout,
            headless=headless,
        )
