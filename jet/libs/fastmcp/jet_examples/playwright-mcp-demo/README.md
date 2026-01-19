# Playwright MCP Demo

Demonstration of structured accessibility-based browser automation using Playwright MCP.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure the MCP server is running:

```bash
npx @playwright/mcp@latest --port 8931
```

Then run the demo:

```bash
python run_demo.py
```

See `src/recipes/basic_interaction.py` for the core pattern.
