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

## Per-agent token budgeting (dedicated implementation)

`token_budget.py` in the orchestrator flow above does basic per-agent
reservation, but production per-agent budgeting needs more than a cap on
cumulative spend. `agent_budget_gateway.py` adds the rest:

| Mechanism | File | Catches |
|---|---|---|
| Hard per-agent budget + shared reserve with burst borrowing | `agent_budget_gateway.py` | Total overspend, while still letting a busy agent borrow instead of hard-failing |
| Token bucket (TPM) rate limit | `token_bucket.py` | An agent burning tokens too *fast* (retry loop, runaway generation) even under its total cap |
| Per-agent circuit breaker | `circuit_breaker.py` | A persistently failing/rejected agent hammering the server — it gets cut off locally for a cooldown, no server hit |
| Cost/usage ledger | `agent_budget_gateway.report()` | "Which agent/session spent what" attribution after the fact |

All of an agent's calls go through `gateway.call(agent_id, messages)` — there
is no path to the model that bypasses the gateway, so budgets can't be
sidestepped by agent code.

Run the fleet demo (5 concurrent agents, one deliberately configured to blow
its budget, to watch isolation, borrowing, and circuit-breaking in action):

```bash
python demo_per_agent_budget.py
```

Tune per-agent shape via `AgentTier(hard_budget=, tpm=, burst_capacity=,
max_tokens_per_call=, cost_per_1k_tokens=)`, and the shared borrowing pool via
`AgentBudgetGateway(shared_reserve=...)`.

## Tuning knobs

- `Orchestrator(total_token_budget=..., per_agent_budget=...)` — hard caps.
- `AgentDefinition(max_turns=, max_tokens_per_call=, summary_max_tokens=)` —
  per-subagent-role shape, set in `orchestrator.dispatch()`.
- Swap the in-process `TokenBudget` lock for a Redis-backed store if you need
  the budget shared across multiple processes/machines.
