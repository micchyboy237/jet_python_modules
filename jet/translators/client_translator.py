#!/usr/bin/env python3
# client_translator.py
# Simple, beautiful client for your new POST /translate (text → English) endpoint
# Works perfectly on Mac M1 (Apple Silicon) – runs on CPU or CUDA if you forward ports

import asyncio
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def translate_text(text: str, server_url: str = "http://shawn-pc.local:8001", device: str = "cuda") -> str:
    """Synchronous version: blocks until translation completes."""
    with httpx.Client(timeout=60.0) as client:
        payload = {"text": text}
        with console.status("[bold magenta]Translating...[/]"):
            r = client.post(f"{server_url}/translate", json=payload, params={"device": device})
        
        r.raise_for_status()
        data = r.json()
        return data["translation"]

async def atranslate_text(text: str, server_url: str = "http://shawn-pc.local:8001", device: str = "cuda") -> str:
    """Asynchronous version: non-blocking, awaitable."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {"text": text}
        with console.status("[bold magenta]Translating...[/]"):
            r = await client.post(f"{server_url}/translate", json=payload, params={"device": device})
        
        r.raise_for_status()
        data = r.json()
        return data["translation"]


async def interactive_mode():
    console.print(Panel("[bold green]Whisper Server – Text Translation Client[/]", style="bold blue"))
    console.print("Enter text below (empty line + Enter = quit)\n")

    server = Prompt.ask("Server URL", default="http://shawn-pc.local:8001")
    device = Prompt.ask("Device for translation model", choices=["cpu", "cuda"], default="cuda")

    while True:
        text = Prompt.ask("\n[yellow]Text to translate[/]")
        if not text.strip():
            console.print("[dim]Goodbye![/]")
            break

        try:
            translation = await atranslate_text(text, server_url=server, device=device)
            table = Table.grid(padding=(0, 2))
            table.add_column("Original", style="cyan")
            table.add_column("Translation", style="green")
            table.add_row(text, translation)
            console.print(table)
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Server error {e.response.status_code}: {e.response.text}[/]")
        except Exception as e:
            console.print(f"[red]Failed:[/] {e}")


async def file_mode(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    console.print(f"[bold]Loaded {len(text)} characters from[/] [bold]{file_path}[/]")
    translation = await atranslate_text(text)
    console.print("\n[bold green]Translation:[/]\n")
    console.print(translation)


def main():
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if not file_path.is_file():
            console.print(f"[red]File not found:[/] {file_path}")
            sys.exit(1)
        asyncio.run(file_mode(file_path))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Bye![/]")