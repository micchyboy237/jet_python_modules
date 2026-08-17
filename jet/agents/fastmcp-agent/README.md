# llama-mcp-bridge

Exposes a local llama.cpp `llama-server` instance as MCP tools via FastMCP,
so any MCP client (Claude Desktop, Claude Code, or the included demo client)
can call your local model.

```
llama-mcp-bridge/
├── llama_mcp_server.py   # FastMCP server — wraps llama-server as MCP tools
├── demo_client.py         # FastMCP client — discovers, selects, and calls those tools
├── requirements.txt       # Python dependencies
└── README.md
```

## 1. Start llama-server

On the machine with the GPU (e.g. Windows box with a GTX 1660):

```powershell
llama-server -hf unsloth/Qwen2.5-7B-Instruct-GGUF:Q4_K_M ^
  --host 0.0.0.0 --port 8080 --n-gpu-layers 28 --ctx-size 4096
```

- `--host 0.0.0.0` lets other machines on your LAN reach it (use `127.0.0.1` if everything runs on one machine).
- Tune `--n-gpu-layers` down if you hit VRAM out-of-memory errors.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Point the bridge at llama-server

```bash
export LLAMA_SERVER_URL=http://192.168.1.50:8080   # your llama-server's address
```

## 4. Try it with the demo client

```bash
# Interactive: pick a tool, fill in its arguments
python demo_client.py

# Non-interactive
python demo_client.py --tool check_llama_server_health
python demo_client.py --tool ask_local_llm --prompt "Explain recursion in one sentence"
```

The demo client spawns `llama_mcp_server.py` itself over stdio, so set
`LLAMA_SERVER_URL` in the same shell before running `demo_client.py`.

## 5. Or register it with Claude Desktop / Claude Code

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-llama": {
      "command": "python",
      "args": ["/absolute/path/to/llama_mcp_server.py"],
      "env": { "LLAMA_SERVER_URL": "http://192.168.1.50:8080" }
    }
  }
}
```

## Tools exposed

| Tool                                                                                                                                                    | Purpose                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `ask_local_llm(prompt, system_prompt="", max_tokens=512, temperature=None, top_p=None, min_p=None, top_k=None, repeat_penalty=None, extra_params=None)` | Sends a prompt to llama-server and returns the reply |
| `check_llama_server_health()`                                                                                                                           | Pings llama-server to confirm it's reachable         |

### Custom generation params

`ask_local_llm` exposes the common sampling params directly. Only params you set
are sent — anything left as `None` falls back to llama-server's own default.

```python
result = await client.call_tool("ask_local_llm", {
    "prompt": "Write a short poem about the sea",
    "temperature": 0.9,
    "top_p": 0.95,
    "min_p": 0.05,
})
```

For anything not listed as an explicit parameter (e.g. `dry_multiplier`, `grammar`,
`repeat_last_n`), pass `extra_params` — it's merged directly into the JSON body
sent to llama-server, since llama-server accepts these as plain top-level fields
(no `extra_body` wrapper needed — that's an OpenAI SDK-only concept):

```python
result = await client.call_tool("ask_local_llm", {
    "prompt": "Say hi",
    "extra_params": {"dry_multiplier": 0.5, "repeat_last_n": 128},
})
```
