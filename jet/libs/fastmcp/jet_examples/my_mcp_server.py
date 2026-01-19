import asyncio
from fastmcp import Client
from rich import print as rprint           # better looking output

async def quick_demo():
    # Using your local llama.cpp server
    client = Client("http://shawn-pc.local:8080/v1")

    async with client:
        # 1. Check connection
        await client.ping()
        rprint("[bold green]✓ Server is alive[/bold green]")

        # 2. See what tools are available
        tools = await client.list_tools()
        rprint("\n[cyan]Available tools:[/cyan]")
        for t in tools[:6]:  # limit output
            rprint(f"  • {t.name:18} {t.description or '—'}")

        # 3. Execute something
        try:
            result = await client.call_tool(
                "generate_chart",
                {"type": "bar", "data": [10, 30, 25, 60], "title": "Sales Q4"},
                timeout=25.0
            )
            rprint("\n[green]Result.data:[/green]", result.data)
        except Exception as e:
            rprint("[red]Tool call failed:[/red]", e)

        # 4. Get a prepared prompt (very common pattern)
        messages = await client.get_prompt(
            "code_review",
            {"language": "python", "code": "def add(a,b): return a+b"}
        )
        rprint("\n[yellow]Generated prompt messages:[/yellow]")
        for msg in messages.messages:
            rprint(f"  {msg.role:8} │ {msg.content.text[:70]!r}...")

asyncio.run(quick_demo())