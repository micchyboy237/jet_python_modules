"""
Orchestrator: plans, dispatches subagents in parallel, aggregates, synthesizes.

Structural choices that map back to the design discussion:

- Decompose once, dispatch in parallel  -> keeps wall-clock down and lets
  each subagent explore its slice with a clean, isolated context.
- Orchestrator context only ever grows by artifact *references*
  (id + short summary), never full subagent transcripts.
- An explicit aggregation/dedup pass runs before synthesis, since parallel
  subagents commonly return overlapping findings -- without this step they
  just get concatenated and silently re-inflate the context you isolated
  subagents specifically to avoid.
- A shared TokenBudget is enforced across every call, orchestrator and
  subagents alike, with per-agent isolation so one runaway subagent can't
  eat the whole run's budget.
"""

from __future__ import annotations

import json
import concurrent.futures as cf
from dataclasses import dataclass

from llama_client import LlamaCppClient, ChatResult
from token_budget import TokenBudget, spend, BudgetExceededError
from artifact_store import ArtifactStore
from subagent import Subagent, AgentDefinition, SubagentResult


ORCHESTRATOR_SYSTEM_PROMPT = """You are a lead orchestrator agent. You decompose \
a user task into 2-4 independent subtasks that can be researched in parallel by \
specialized subagents. Each subtask should be self-contained: include everything \
a subagent would need to know, since subagents cannot see this conversation.

Respond ONLY with JSON in this exact shape, no other text:
{"subtasks": [{"name": "short-id", "instructions": "full task-context for the subagent"}]}
"""

AGGREGATION_SYSTEM_PROMPT = """You are given several subagent finding-summaries for \
the same overall task. Merge them into a single deduplicated list of findings: \
combine restatements of the same fact, flag direct contradictions explicitly, \
and drop nothing that is unique. Be terse. No preamble."""

SYNTHESIS_SYSTEM_PROMPT = """You are the lead orchestrator. Write the final answer \
to the user's original task using only the aggregated findings provided. Do not \
invent information not present in the findings."""


@dataclass
class OrchestratorResult:
    final_answer: str
    subagent_results: list[SubagentResult]
    aggregated_findings: str
    budget_report: dict


class Orchestrator:
    def __init__(
        self,
        client: LlamaCppClient,
        total_token_budget: int = 60_000,
        per_agent_budget: int = 12_000,
        artifact_dir: str = "artifacts",
    ):
        self.client = client
        self.budget = TokenBudget(total_token_budget, per_agent_budget)
        self.store = ArtifactStore(artifact_dir)

    # ---- step 1: decompose -------------------------------------------------

    def plan(self, task: str) -> list[dict]:
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        estimate = sum(self.client.count_tokens(m["content"]) for m in messages) + 600
        with spend(self.budget, "orchestrator:plan", estimate) as tx:
            result = self.client.chat(messages, max_tokens=600, temperature=0.2)
            tx.actual = result.total_tokens

        # Persist the plan immediately -- if we later run out of budget or
        # need to resume with a fresh subagent, the plan survives in the
        # artifact store rather than only living in a context that might
        # get truncated.
        self.store.write(
            kind="plan",
            agent_id="orchestrator",
            summary=f"Decomposed task into subtasks: {result.text[:200]}",
            content=result.text,
        )
        return json.loads(result.text)["subtasks"]

    # ---- step 2: dispatch subagents in parallel, isolated context ---------

    def dispatch(self, subtasks: list[dict]) -> list[SubagentResult]:
        results: list[SubagentResult] = []

        def run_one(subtask: dict) -> SubagentResult | None:
            definition = AgentDefinition(
                name=subtask["name"],
                system_prompt=(
                    f"You are the '{subtask['name']}' subagent. Investigate only "
                    f"your assigned slice of the task. Be concrete: cite specific "
                    f"facts, numbers, and sources where relevant. When done, prefix "
                    f"your final answer with FINAL:."
                ),
            )
            subagent = Subagent(definition, self.client, self.budget, self.store)
            try:
                return subagent.run(task_id=subtask["name"], task_context=subtask["instructions"])
            except BudgetExceededError as e:
                # Graceful degradation: a runaway subagent gets cut off,
                # everyone else keeps working within their own allocation.
                print(f"[budget] {subtask['name']} skipped/truncated: {e}")
                return None

        # Parallel dispatch -- each subagent has its own isolated context,
        # so there's no shared-state hazard beyond the thread-safe budget.
        with cf.ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
            for res in pool.map(run_one, subtasks):
                if res is not None:
                    results.append(res)

        return results

    # ---- step 3: explicit aggregation/dedup pass ---------------------------

    def aggregate(self, results: list[SubagentResult]) -> str:
        # Only references + summaries go into context here -- never the raw
        # subagent transcripts, which stay on disk in the artifact store.
        combined = "\n\n".join(r.reference for r in results)
        messages = [
            {"role": "system", "content": AGGREGATION_SYSTEM_PROMPT},
            {"role": "user", "content": combined},
        ]
        estimate = sum(self.client.count_tokens(m["content"]) for m in messages) + 500
        with spend(self.budget, "orchestrator:aggregate", estimate) as tx:
            result = self.client.chat(messages, max_tokens=500, temperature=0.1)
            tx.actual = result.total_tokens
        return result.text

    # ---- step 4: synthesize final answer -----------------------------------

    def synthesize(self, task: str, aggregated_findings: str) -> str:
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original task: {task}\n\nFindings:\n{aggregated_findings}"},
        ]
        estimate = sum(self.client.count_tokens(m["content"]) for m in messages) + 700
        with spend(self.budget, "orchestrator:synthesize", estimate) as tx:
            result = self.client.chat(messages, max_tokens=700, temperature=0.3)
            tx.actual = result.total_tokens
        return result.text

    # ---- full run ------------------------------------------------------------

    def run(self, task: str) -> OrchestratorResult:
        subtasks = self.plan(task)
        subagent_results = self.dispatch(subtasks)

        if not subagent_results:
            return OrchestratorResult(
                final_answer="No subagents completed within budget.",
                subagent_results=[],
                aggregated_findings="",
                budget_report=self.budget.report(),
            )

        aggregated = self.aggregate(subagent_results)
        final = self.synthesize(task, aggregated)

        return OrchestratorResult(
            final_answer=final,
            subagent_results=subagent_results,
            aggregated_findings=aggregated,
            budget_report=self.budget.report(),
        )
