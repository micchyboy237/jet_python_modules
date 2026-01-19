#!/bin/bash

# setup_fastmcp_project.sh
# Unix shell script to create the complete file structure and populate all files
# for the FastMCP example project described in Response A.
#
# Usage:
#   chmod +x setup_fastmcp_project.sh
#   ./setup_fastmcp_project.sh
#
# The script will create a directory named "project1" in the current working directory
# and overwrite any existing files with the exact content provided.

set -e  # Exit on any error

PROJECT_ROOT="project1"

echo "Creating project structure in ./$PROJECT_ROOT ..."

# Create directory structure
mkdir -p "$PROJECT_ROOT"/{mcp_servers/financial_analyzer,mcp_servers/rag_search,clients,tests}

# ------------------------------------------------------------------
# 1. requirements.txt
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/requirements.txt
fastmcp>=2.14.0
fastmcp[openai] # for OpenAI-compatible sampling handler
rich>=13.7.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
pydantic>=2.5.0 # used internally by fastmcp
EOF

# ------------------------------------------------------------------
# 2. .env
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/.env
# Local llama.cpp endpoints (adjust ports/models as needed)
LLAMA_CPP_URL=http://shawn-pc.local:8080/v1
EMBEDDINGS_URL=http://shawn-pc.local:8081/v1
# Optional: for future OpenAI-compatible usage
OPENAI_API_KEY=sk-fake-key-for-local-llama
# Logging
LOG_LEVEL=INFO
EOF

# ------------------------------------------------------------------
# 3. mcp-config.yaml
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/mcp-config.yaml
mcpServers:
  financial:
    command: python
    args: ["mcp_servers/financial_analyzer/server.py"]
    env:
      LOG_LEVEL: ${LOG_LEVEL}
    cwd: "."
  rag:
    command: python
    args: ["mcp_servers/rag_search/server.py"]
    env:
      LOG_LEVEL: ${LOG_LEVEL}
      EMBEDDINGS_URL: ${EMBEDDINGS_URL}
    cwd: "."
  # Optional: if you want to connect directly to llama.cpp as a sampling source
  # (not a real MCP server, but useful for hybrid setups)
  llm:
    transport: http
    url: ${LLAMA_CPP_URL}
EOF

# ------------------------------------------------------------------
# 4. mcp_servers/financial_analyzer/server.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/mcp_servers/financial_analyzer/server.py
"""Simple financial analysis MCP server with tools and prompts."""
import os
from datetime import datetime
from typing import Dict, Any, List
from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP(
    name="FinancialAnalyzer",
    instructions="You are a financial analysis assistant. Use the provided tools to help users analyze financial data."
)

class StockPriceInput(BaseModel):
    symbol: str
    date: str | None = None # YYYY-MM-DD

class PortfolioInput(BaseModel):
    holdings: Dict[str, float] # symbol -> shares

@mcp.tool(output_schema=StockPriceInput)
def get_stock_price(symbol: str, date: str | None = None) -> Dict[str, Any]:
    """Fetch current or historical stock price (mocked for demo)."""
    # In real life → call real API (e.g., yfinance, alpha vantage)
    price = 150.25 if date is None else 142.80
    return {
        "symbol": symbol.upper(),
        "price": price,
        "currency": "USD",
        "timestamp": datetime.utcnow().isoformat(),
        "source": "mock"
    }

@mcp.tool(output_schema=Dict[str, float])
def calculate_portfolio_value(holdings: Dict[str, float]) -> Dict[str, Any]:
    """Calculate total portfolio value from holdings (mocked prices)."""
    total = sum(shares * 150.25 for shares in holdings.values()) # mocked price
    return {
        "total_value": total,
        "currency": "USD",
        "timestamp": datetime.utcnow().isoformat(),
        "breakdown": {sym: shares * 150.25 for sym, shares in holdings.items()}
    }

@mcp.prompt(name="financial_report_template")
def financial_report_template(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate a formatted financial report prompt."""
    return [
        {"role": "system", "content": "You are a professional financial analyst."},
        {"role": "user", "content": f"Analyze this portfolio data:\n{data}"}
    ]

@mcp.resource(uri="resource://reports/example.json")
def example_report() -> Dict[str, Any]:
    """Static example financial report."""
    return {
        "title": "Sample Q4 Report",
        "total_value": 125000.75,
        "assets": ["AAPL", "TSLA", "GOOGL"]
    }

if __name__ == "__main__":
    mcp.run()
EOF

# ------------------------------------------------------------------
# 5. mcp_servers/rag_search/server.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/mcp_servers/rag_search/server.py
"""Simple RAG-style search server using embeddings."""
import os
from typing import List, Dict, Any
from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP(
    name="RAGSearch",
    instructions="You are a document search assistant using vector embeddings."
)

class SearchInput(BaseModel):
    query: str
    top_k: int = 3

@mcp.tool(output_schema=List[Dict[str, Any]])
def semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Perform semantic search over documents (mocked)."""
    # In real life → call your embeddings endpoint + vector DB
    mock_results = [
        {"id": "doc1", "text": f"Mock result 1 for '{query}'", "score": 0.95},
        {"id": "doc2", "text": f"Mock result 2 for '{query}'", "score": 0.88},
        {"id": "doc3", "text": f"Mock result 3 for '{query}'", "score": 0.75},
    ]
    return mock_results[:top_k]

@mcp.prompt(name="rag_answer")
def rag_answer_template(query: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Prompt template for answering with retrieved context."""
    context_str = "\n".join([f"- {r['text']} (score: {r['score']})" for r in context])
    return [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_str}"}
    ]

if __name__ == "__main__":
    mcp.run()
EOF

# ------------------------------------------------------------------
# 6. clients/interactive_console.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/clients/interactive_console.py
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
EOF

# ------------------------------------------------------------------
# 7. clients/batch_processor.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/clients/batch_processor.py
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
EOF

# ------------------------------------------------------------------
# 8. clients/agent_service.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/clients/agent_service.py
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
EOF

# ------------------------------------------------------------------
# 9. tests/test_in_memory.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/tests/test_in_memory.py
"""In-memory FastMCP server tests."""
import pytest
from fastmcp import FastMCP, Client
from dataclasses import dataclass

@dataclass
class MockResult:
    value: str

@pytest.mark.asyncio
async def test_in_memory_tool():
    # Given
    mcp = FastMCP("TestServer")
    @mcp.tool
    def echo(text: str) -> str:
        return text.upper()
    client = Client(mcp)
    # When
    async with client:
        result = await client.call_tool("echo", {"text": "hello world"})
    # Then
    expected = "HELLO WORLD"
    assert result.data == expected
EOF

# ------------------------------------------------------------------
# 10. tests/test_stdio.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/tests/test_stdio.py
"""Stdio transport tests."""
import pytest
from fastmcp import Client

@pytest.mark.asyncio
async def test_stdio_connection():
    # Given
    client = Client("mcp_servers/financial_analyzer/server.py")
    # When / Then
    async with client:
        tools = await client.list_tools()
        expected_tool_names = ["get_stock_price", "calculate_portfolio_value"]
        assert [t.name for t in tools] == expected_tool_names
EOF

# ------------------------------------------------------------------
# 11. tests/test_multi_server.py
# ------------------------------------------------------------------
cat << 'EOF' > "$PROJECT_ROOT"/tests/test_multi_server.py
"""Multi-server config tests."""
import pytest
from fastmcp import Client

@pytest.mark.asyncio
async def test_multi_server_tools():
    # Given
    client = Client("mcp-config.yaml")
    # When
    async with client:
        tools = await client.list_tools()
    # Then
    expected = ["financial_get_stock_price", "financial_calculate_portfolio_value", "rag_semantic_search"]
    actual = [t.name for t in tools]
    assert sorted(actual) == sorted(expected)
EOF

echo "Project setup complete!"
echo ""
echo "Next steps:"
echo "  cd $PROJECT_ROOT"
echo "  python -m venv venv && source venv/bin/activate  # or use uv, poetry, etc."
echo "  pip install -r requirements.txt"
echo "  python clients/interactive_console.py   # to try the interactive client"
echo "  pytest                                 # to run the tests"
echo ""
echo "Remember to adjust the URLs/models in .env to match your local llama.cpp setup."
