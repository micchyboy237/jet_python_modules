"""Example agent service using multi-server + sampling."""
import asyncio
from rich.console import Console
from fastmcp import Client
from fastmcp.client.sampling.handlers.openai import OpenAISamplingHandler
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

console = Console()

async def main():
    client = Client("mcp-config.yaml")
    sampling_handler = OpenAISamplingHandler(
        default_model="your-model-name",
        client=AsyncOpenAI(base_url=os.getenv("LLAMA_CPP_URL"), api_key="fake")
    )
    async with client:
        console.print("[bold green]Agent Service Demo[/]")
        # Step 1: Get prompt template
        prompt_result = await client.get_prompt("financial:financial_report_template", {
            "data": {"total_value": 250000, "assets": ["AAPL", "TSLA"]}
        })
        # Step 2: Let LLM generate analysis (using llama.cpp)
        # In real agent you would chain tool calls here
        console.print("[dim]Generated system prompt:[/]")
        for msg in prompt_result.messages:
            console.print(f" {msg['role']}: {msg['content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
