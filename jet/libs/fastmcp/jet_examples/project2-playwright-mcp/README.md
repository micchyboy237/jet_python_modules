# project2 – Playwright-MCP + FastMCP Python Client

Modern example of using **[playwright-mcp](https://github.com/microsoft/playwright-mcp)**  
(an official/community MCP server exposing Playwright browser automation)  
together with **FastMCP** Python client.

Main goal: building reliable, structured web research & automation agents  
running completely locally.

Current state: January 2026

## Why this combination?

playwright-mcp advantages over custom python playwright + vision:
- Uses **structured accessibility tree** → much more reliable for LLMs
- Fast & deterministic (no screenshot OCR noise)
- Maintained by Microsoft + active community forks
- Works great in stdio (local) and http (remote/team) modes

FastMCP python client advantages:
- Clean async interface
- Structured outputs (`.data`)
- Background tasks
- Multi-server config
- Easy integration with your existing llama.cpp sampling

## Requirements

### Server side (Node.js)
- Node.js ≥ 18
- npm / npx

### Client side (Python)
- Python ≥ 3.10
- Local llama.cpp endpoint (optional – for summarization)

## Quick Start

### 1. Install dependencies

```bash
# Python part
cd project2-playwright-mcp
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start playwright-mcp server

**Option A – Recommended: HTTP mode** (easiest with python client)

```bash
# Terminal 1 - keep this running
npx @playwright/mcp@latest --port 8931
# or using popular community fork with more features:
# npx @executeautomation/playwright-mcp-server --port 8931 --headless
```

**Option B – Stdio mode** (more like Claude Desktop style)

```yaml
# Use this version in mcp-config.yaml instead
playwright:
  command: npx
  args: ["@playwright/mcp@latest", "--headless"]
```

### 3. Run the examples

```bash
# Basic navigation & content extraction (now starts on your local SearXNG)
python clients/simple_browser.py

# Or pass a custom search query
python clients/simple_browser.py --url "http://jethros-macbook-air.local:8888/search?q=python%20asyncio"
```

## Running deep research (now bypasses SearXNG challenges)

```bash
# Basic run (regex only – should now work even with challenge pages)
python clients/deep_research.py \
  --url "http://jethros-macbook-air.local:8888/search?q=playwright%20mcp%20agents" \
  -t "Playwright MCP agents" \
  --max-depth 4

# Smarter run with local LLM link scoring
python clients/deep_research.py \
  --url "http://jethros-macbook-air.local:8888/search?q=playwright%20mcp%20agents" \
  -t "Playwright MCP agents" \
  --use-llm \
  --max-depth 3
```

## Project Structure

```text
project2-playwright-mcp/
├── clients/
│   ├── simple_browser.py         # Basic navigation + content extract
│   └── deep_research.py          # Multi-page chained research pattern
├── tests/
│   └── test_browser_connection.py  # Basic connection & tool discovery
├── mcp-config.yaml               # ← Single source of truth (http recommended)
├── .env                          # Server URL + browser settings
├── requirements.txt
└── README.md
```

## Detailed Setup Options

### Using HTTP mode (recommended for development)

```yaml
# mcp-config.yaml
mcpServers:
  playwright:
    transport: http
    url: http://localhost:8931
```

### Using Stdio mode (single process, more isolated)

```yaml
mcpServers:
  playwright:
    command: npx
    args: ["@playwright/mcp@latest"]
    env:
      HEADLESS: "true"
      TIMEOUT: "45000"
```

## Available Tools (current server – January 2026)

Your current playwright-mcp server exposes the following tools:

```text
browser_close
browser_resize
browser_console_messages
browser_handle_dialog
browser_evaluate
browser_file_upload
browser_fill_form
browser_install
browser_press_key
browser_type
browser_navigate
browser_navigate_back
browser_network_requests
browser_run_code
browser_take_screenshot
browser_snapshot           ← most important: structured accessibility tree/text
browser_click
browser_drag
browser_hover
browser_select_option
browser_tabs
browser_wait_for           ← very useful for reliability!
```

> **Tip:** Always run `await client.list_tools()` to confirm which tools are available on your connected server.

## Running Tests

```bash
pytest

# Watch mode (very useful during development)
pytest -f
```

## Common Patterns & Next Steps

1. **Research → Summarize**  
   Extract → feed content to your llama.cpp → ask for summary → decide next action

2. **Login flows**  
   Use `type()` + `click()` in sequence (many forks support persistent context)

3. **Screenshot + Vision**  
   Get screenshot → send base64 to multimodal model (llava, qwen-vl, etc.)

4. **Parallel browsing**  
   Use background tasks (`task=True`) + multiple pages/contexts

5. **Reliability improvements**  
   - Add retry logic on tool calls  
   - Use `wait_for_*` tools  
   - Save/restore browser state

## Troubleshooting Tips

Problem                              Quick fix
─────────────────────────────────────────────────────────────────────
"Connection refused"                 Make sure server is running on correct port
Tools not found                      Run `await client.list_tools()` and check names
Headless=false but no window         Some forks need `--headless false --no-sandbox`
Slow navigation                      Increase timeout in .env or per-call
Stuck after redirect                 Use `wait_until="networkidle"` if available
