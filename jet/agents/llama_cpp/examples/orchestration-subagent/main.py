"""
Example run.

Prerequisite: a local llama.cpp server running with an OpenAI-compatible API
and a context size comfortably larger than any single subagent call, e.g.:

    llama-server -m ./models/your-model.Q4_K_M.gguf -c 8192 --port 8080

Then:

    pip install requests
    python main.py
"""

from llama_client import LlamaCppClient
from orchestrator import Orchestrator


def main():
    client = LlamaCppClient(base_url="http://localhost:8080")

    orchestrator = Orchestrator(
        client=client,
        total_token_budget=60_000,   # hard ceiling for the whole run
        per_agent_budget=12_000,     # no single subagent can exceed this
        artifact_dir="artifacts",
    )

    task = (
        "Compare SQLite and DuckDB for embedding as an analytics engine "
        "inside a desktop app: query performance on aggregations, "
        "footprint/dependencies, and ecosystem maturity."
    )

    result = orchestrator.run(task)

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result.final_answer)

    print("\n" + "=" * 70)
    print("SUBAGENT ARTIFACTS (full detail stays on disk, referenced by id)")
    print("=" * 70)
    for r in result.subagent_results:
        print(f"- {r.artifact.id} [{r.artifact.agent_id}] "
              f"({r.tokens_used} tokens) -> artifacts/{r.artifact.id}.json")

    print("\n" + "=" * 70)
    print("TOKEN BUDGET REPORT")
    print("=" * 70)
    import json
    print(json.dumps(result.budget_report, indent=2))


if __name__ == "__main__":
    main()
