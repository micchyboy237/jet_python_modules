"""Interactive console client for FastMCP servers."""
import asyncio
from rich.console import Console
from rich.prompt import Prompt
from fastmcp import Client
from fastmcp.client.sampling.handlers.openai import OpenAISamplingHandler
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

console = Console()

async def main():
    client = Client("mcp-config.yaml")
    # Use your local llama.cpp for sampling
    sampling_handler = OpenAISamplingHandler(
        default_model="your-model-name", # e.g. "llama-3.1-70b"
        client=AsyncOpenAI(
            base_url=os.getenv("LLAMA_CPP_URL"),
            api_key="not-needed"
        )
    )
    async with client:
        console.print("[bold green]FastMCP Interactive Console[/] (type 'exit' to quit)")
        while True:
            query = Prompt.ask("You")
            if query.lower() in ("exit", "quit"):
                break
            try:
                # Example: call a tool and get structured result
                result = await client.call_tool(
                    "financial:get_stock_price",
                    {"symbol": "AAPL"},
                    sampling_handler=sampling_handler
                )
                console.print(f"[bold cyan]Result.data[/]: {result.data}")
            except Exception as e:
                console.print(f"[red]Error:[/] {e}")

if __name__ == "__main__":
    asyncio.run(main())
