# FastMCP Examples – Project 1

Starter project demonstrating modern usage of **[FastMCP](https://github.com/jlowin/fastmcp)**  
(local-first Model Context Protocol servers & clients) together with local LLMs (llama.cpp).

Focuses on:
- Multi-server setup via `mcp-config.yaml`
- Structured tool outputs with Pydantic hydration
- Background tasks with progress tracking
- LLM sampling using local llama.cpp endpoints
- Interactive console, batch processing, and agent-style patterns
- Basic pytest testing (in-memory + real stdio + multi-server)

Current date reference: January 2026

## Features Demonstrated

- Two example MCP servers:  
  • `financial` – mocked stock prices & portfolio calculations  
  • `rag` – mocked semantic search with RAG-style prompt

- Three client patterns:  
  • `interactive_console.py` – REPL-style interaction  
  • `batch_processor.py` – parallel background tasks  
  • `agent_service.py` – prompt + sampling chain

- Local llama.cpp integration for LLM completions  
- Rich console output + progress bars  
- Environment variable support via `.env`

## Requirements

- Python 3.10+
- Local llama.cpp server running (OpenAI-compatible endpoint)  
  Recommended ports:  
  • LLM completions → `http://shawn-pc.local:8080/v1`  
  • Embeddings (optional) → `http://shawn-pc.local:8081/v1`

## Quick Start

```bash
# 1. Enter project directory
cd jet/libs/fastmcp/jet_examples/project1

# 2. Create & activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# or faster with uv:
# uv sync

# 4. (Optional) Customize LLM endpoint/model in .env file
#    Default already points to your shawn-pc.local:8080

# 5. Run examples (choose one)
python clients/interactive_console.py
python clients/batch_processor.py
python clients/agent_service.py
```

## Project Structure

```text
project1/
├── mcp_servers/
│   ├── financial_analyzer/
│   │   └── server.py                # Financial tools & prompts
│   └── rag_search/
│       ├── server.py                # Mock RAG semantic search
│       └── vector_store.py          # (placeholder for real vector DB)
├── clients/
│   ├── interactive_console.py       # REPL-style user interaction
│   ├── batch_processor.py           # Parallel background tasks demo
│   └── agent_service.py             # Prompt + LLM sampling chain
├── tests/
│   ├── test_in_memory.py
│   ├── test_stdio.py
│   └── test_multi_server.py
├── mcp-config.yaml                  # ← Single source of truth for all servers
├── .env                             # LLM endpoints & secrets
└── requirements.txt
```

## Running the Servers Manually (optional)

You normally **don't need** to run servers manually — clients launch them via stdio when using `mcp-config.yaml`.

For debugging:

```bash
# Financial analyzer
python mcp_servers/financial_analyzer/server.py

# RAG search
python mcp_servers/rag_search/server.py
```

Or run in HTTP mode (useful for remote access):

```bash
# Example for financial server
python -m fastmcp run --transport http --port 8765 mcp_servers/financial_analyzer/server.py
```

## Running Tests

```bash
# All tests
pytest

# Watch mode (great for development)
pytest -f

# Single file
pytest tests/test_multi_server.py
```

## Current Limitations / Next Steps

- All tool results are **mocked** — replace with real APIs (yfinance, Alpha Vantage, etc.)
- RAG server is mocked — next step: integrate real embeddings + vector DB (chromadb, qdrant, lancedb)
- No real error recovery/retry logic yet
- Sampling currently uses placeholder model name — update `.env` + code with your actual llama.cpp model

## Recommended Improvements (future work)

- Add real financial data source (yfinance / polygon.io)
- Implement persistent vector store + real embedding calls
- Add streaming responses from LLM
- Better error handling + rich error display
- Add CLI entrypoint with `typer` / `click`
- Containerize with Docker (multi-service compose)

## License

MIT – feel free to use as starter template

Happy local AI hacking! 🚀  
Makati City, 2026
```

You can now create this file with:

```bash
cat > README.md << 'EOF'
# paste the content above
EOF
```

or just copy-paste it manually.

Let me know if you want to expand any section (installation variants with uv/pipx, screenshots, badges, architecture diagram ideas, etc.)!