# qwen-studio-style-agent

A minimal, Docker-free agentic AI assistant that replicates the **Qwen Studio single-model + tool-calling** architecture. Implements the exact **search → read → reason → answer** loop used by Qwen3.8, running entirely on a local `llama.cpp` server with secure WebAssembly-based code execution.

## Why "Qwen Studio Style"?

This project mirrors the operational pattern of Qwen Studio / Qwen3.8:

- ✅ Single reasoning model (no multi-agent orchestration)
- ✅ Conditional tool invocation driven by the model itself
- ✅ Iterative retrieval: search → extract → re-search if needed → synthesize
- ✅ Goal-focused extraction to conserve context budget
- ✅ Secure code execution as a first-class tool
- ✅ Stateless LLM + stateful orchestrator separation

## Features

- 🔍 **Web Search** via SearXNG
- 📄 **Web Extraction** with goal-focused LLM summarization
- 🐍 **Secure Code Interpreter** via Pyodide (WASM sandbox, no Docker)
- 🔄 **Agentic Loop** with automatic tool selection and iteration limits
- 🧩 **Modular Tool Registry** — add/remove tools without touching core logic

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
cp .env.example .env

# 2. Start SearXNG (no Docker needed)
pip install searxng && searxng-run &

# 3. Start llama.cpp server (separate terminal)
./llama-server \
  -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --port 8080 --ctx-size 8192 --jinja

# 4. Run the agent
python main.py
```

## Project Structure

```
agent/          # Orchestrator, LLM client, config
tools/          # Pluggable tools (search, extractor, code interpreter)
main.py         # CLI entry point
.env.example    # Configuration template
```

## Requirements

- Python ≥ 3.11
- `llama.cpp` server with `--jinja` flag
- A tool-calling-capable GGUF model (Qwen2.5 recommended for closest behavioral match)

## Security Note

Code execution uses **Pyodide WASM sandboxing** — no file system, network, or host access. Suitable for personal/local use. For multi-tenant deployments, use containerized isolation.

## License

MIT
