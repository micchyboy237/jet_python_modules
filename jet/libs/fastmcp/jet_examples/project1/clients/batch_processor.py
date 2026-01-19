"""Batch processor example using background tasks."""
import asyncio
from rich.console import Console
from rich.progress import track
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()

console = Console()

async def main():
    client = Client("mcp-config.yaml")
    async with client:
        console.print("[bold green]Starting batch portfolio valuation...[/]")
        symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
        tasks = []
        for symbol in track(symbols, description="Fetching prices..."):
            task = await client.call_tool(
                "financial:get_stock_price",
                {"symbol": symbol},
                task=True # run in background
            )
            tasks.append(task)
        results = []
        for task in tasks:
            result = await task.result()
            results.append(result.data)
        console.print("\n[bold]Portfolio Prices:[/]")
        for r in results:
            console.print(f" • {r['symbol']}: ${r['price']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
