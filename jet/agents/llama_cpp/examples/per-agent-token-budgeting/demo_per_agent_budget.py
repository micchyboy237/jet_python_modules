"""
Demo: several agents sharing one llama.cpp server, each with its own tier.

Run:
    llama-server -m ./models/your-model.gguf -c 8192 --port 8080
    pip install requests
    python demo_per_agent_budget.py

What to look for in the output:
- worker-1..3 stay within their own tight budgets.
- lead-1 has a larger tier and can burst-borrow from the shared reserve.
- runaway-1 is deliberately configured to blow past its own tiny budget
  fast; watch it get BudgetExceededError, its circuit breaker trip, and
  its subsequent calls short-circuit locally (no server hit at all) until
  the cooldown elapses -- it never affects the other agents' throughput.
"""

import concurrent.futures as cf

from llama_client import LlamaCppClient
from agent_budget_gateway import AgentBudgetGateway, AgentTier, BudgetExceededError
from circuit_breaker import CircuitOpenError
from token_bucket import RateLimitedError


WORKER_TIER = AgentTier(name="worker", hard_budget=2_500, tpm=3_000, burst_capacity=1_500, max_tokens_per_call=300)
LEAD_TIER = AgentTier(name="lead", hard_budget=6_000, tpm=6_000, burst_capacity=4_000, max_tokens_per_call=600)
RUNAWAY_TIER = AgentTier(name="runaway", hard_budget=600, tpm=1_000, burst_capacity=600, max_tokens_per_call=500)


def make_prompt(topic: str) -> list[dict]:
    return [
        {"role": "system", "content": "Answer in two sentences, be concrete."},
        {"role": "user", "content": f"Briefly explain: {topic}"},
    ]


def agent_task(gateway: AgentBudgetGateway, agent_id: str, topics: list[str]) -> list[str]:
    log = []
    for topic in topics:
        try:
            result = gateway.call(agent_id, make_prompt(topic))
            log.append(f"[{agent_id}] ok  ({result.total_tokens} tok) -> {result.text[:60]}...")
        except BudgetExceededError as e:
            log.append(f"[{agent_id}] BUDGET EXCEEDED: {e}")
        except CircuitOpenError as e:
            log.append(f"[{agent_id}] CIRCUIT OPEN, skipped call: {e}")
        except RateLimitedError as e:
            log.append(f"[{agent_id}] RATE LIMITED: {e}")
    return log


def main():
    client = LlamaCppClient(base_url="http://localhost:8080")
    gateway = AgentBudgetGateway(client, shared_reserve=8_000)

    gateway.register("worker-1", WORKER_TIER)
    gateway.register("worker-2", WORKER_TIER)
    gateway.register("worker-3", WORKER_TIER)
    gateway.register("lead-1", LEAD_TIER)
    gateway.register("runaway-1", RUNAWAY_TIER)

    fleet = {
        "worker-1": ["what is a B-tree index"],
        "worker-2": ["what is write-ahead logging"],
        "worker-3": ["what is columnar storage"],
        "lead-1": [
            "compare row-oriented vs columnar storage",
            "when would you pick DuckDB over SQLite",
        ],  # larger tier, uses more -- may need to borrow
        "runaway-1": [f"topic {i}" for i in range(6)],  # deliberately overshoots its tiny budget
    }

    with cf.ThreadPoolExecutor(max_workers=len(fleet)) as pool:
        futures = {
            pool.submit(agent_task, gateway, agent_id, topics): agent_id
            for agent_id, topics in fleet.items()
        }
        for future in cf.as_completed(futures):
            for line in future.result():
                print(line)

    print("\n" + "=" * 70)
    print("GATEWAY REPORT (per-agent attribution)")
    print("=" * 70)
    import json
    print(json.dumps(gateway.report(), indent=2))


if __name__ == "__main__":
    main()
