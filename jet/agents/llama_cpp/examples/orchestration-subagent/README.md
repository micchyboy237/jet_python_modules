# Orchestrator–Subagent Handoff Pattern (local llama.cpp)

Implements the patterns discussed:

- **File-based artifact handoffs** (`artifact_store.py`) — subagents write full
  results to disk; only a short summary + reference id goes back into the
  orchestrator's live context.
- **Context isolation** (`subagent.py`) — each subagent gets a fresh, scoped
  conversation with only the task context the orchestrator explicitly hands
  it, never the full conversation history.
- **Forced compression on return** — every subagent run ends with a dedicated
  summarization call before anything crosses back to the orchestrator.
- **Parallel dispatch** (`orchestrator.py`) — subtasks run concurrently via a
  thread pool, each in its own isolated context.
- **Explicit aggregation/dedup pass** — parallel subagent findings are merged
  and deduplicated before final synthesis, instead of being naively
  concatenated.
- **Token budget with atomic reservation** (`token_budget.py`) — a shared,
  thread-safe budget across the whole run, with per-agent sub-budgets so one
  runaway subagent can't consume the entire run's allocation. Uses a
  reserve → call → settle pattern to stay correct under concurrency.

## Run it

```bash
# Start a local llama.cpp server with its OpenAI-compatible API:
llama-server -m ./models/your-model.Q4_K_M.gguf -c 8192 --port 8080

pip install requests
python main.py
```

## Files

| File | Role |
|---|---|
| `llama_client.py` | HTTP client for llama.cpp's `/v1/chat/completions` and `/tokenize` |
| `token_budget.py` | Thread-safe reservation-based token budget |
| `artifact_store.py` | Disk-backed store for full subagent output; returns compressed references |
| `subagent.py` | Isolated-context agent that always hands back a compressed summary |
| `orchestrator.py` | Plan → dispatch (parallel) → aggregate/dedup → synthesize |
| `main.py` | Runnable example task |

## Tuning knobs

- `Orchestrator(total_token_budget=..., per_agent_budget=...)` — hard caps.
- `AgentDefinition(max_turns=, max_tokens_per_call=, summary_max_tokens=)` —
  per-subagent-role shape, set in `orchestrator.dispatch()`.
- Swap the in-process `TokenBudget` lock for a Redis-backed store if you need
  the budget shared across multiple processes/machines.
