"""
Subagent: a fresh, isolated conversation with the model, scoped to one task.

Two rules enforced structurally (not just by prompting):

1. Context isolation -- a subagent NEVER sees the orchestrator's full
   conversation history. It only gets what's explicitly handed to it in
   `task_context`. This is the orchestrator's responsibility, not the
   subagent's: everything the subagent needs (file paths, prior decisions,
   constraints) must be forwarded explicitly.

2. Compression on return -- a subagent's full transcript is written to the
   artifact store and never returned directly. What comes back to the
   orchestrator is a short, forced summary turn, capped in token size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llama_client import LlamaCppClient, ChatResult
from token_budget import TokenBudget, spend
from artifact_store import ArtifactStore, Artifact


@dataclass
class AgentDefinition:
    name: str
    system_prompt: str
    max_turns: int = 4
    max_tokens_per_call: int = 800
    summary_max_tokens: int = 250


@dataclass
class SubagentResult:
    artifact: Artifact          # full transcript, on disk
    reference: str               # what goes back into orchestrator context
    tokens_used: int


class Subagent:
    def __init__(
        self,
        definition: AgentDefinition,
        client: LlamaCppClient,
        budget: TokenBudget,
        store: ArtifactStore,
    ):
        self.definition = definition
        self.client = client
        self.budget = budget
        self.store = store

    def run(self, task_id: str, task_context: str) -> SubagentResult:
        """
        task_context: everything this subagent is allowed to know. The
        orchestrator builds this deliberately -- not the full conversation,
        just the task-relevant slice (see orchestrator.py).
        """
        agent_id = f"{self.definition.name}:{task_id}"
        messages = [
            {"role": "system", "content": self.definition.system_prompt},
            {"role": "user", "content": task_context},
        ]
        tokens_used = 0

        # The subagent works in its own isolated loop -- nothing here is
        # visible to the orchestrator until the final compressed handoff.
        result: ChatResult = self._budgeted_call(agent_id, messages)
        tokens_used += result.total_tokens
        transcript = [f"[{self.definition.name}] {result.text}"]

        # Optional multi-turn self-work loop (subagent can iterate on its
        # own task, e.g. refine an answer) up to max_turns, still isolated.
        turn = 1
        while turn < self.definition.max_turns and self._needs_another_turn(result.text):
            messages.append({"role": "assistant", "content": result.text})
            messages.append({
                "role": "user",
                "content": "Continue. If you have a final answer, prefix it with FINAL:.",
            })
            result = self._budgeted_call(agent_id, messages)
            tokens_used += result.total_tokens
            transcript.append(f"[{self.definition.name}] {result.text}")
            turn += 1

        full_transcript = "\n\n".join(transcript)

        # Force compression: a dedicated summarization call, capped small,
        # so the orchestrator never has to ingest the raw transcript.
        summary = self._summarize(agent_id, full_transcript)
        tokens_used += summary.total_tokens

        artifact = self.store.write(
            kind="subagent_report",
            agent_id=agent_id,
            summary=summary.text,
            content=full_transcript,
            metadata={"turns": turn, "tokens_used": tokens_used},
        )

        return SubagentResult(
            artifact=artifact,
            reference=self.store.reference(artifact),
            tokens_used=tokens_used,
        )

    def _budgeted_call(self, agent_id: str, messages: list[dict]) -> ChatResult:
        estimate = sum(self.client.count_tokens(m["content"]) for m in messages)
        estimate += self.definition.max_tokens_per_call
        with spend(self.budget, agent_id, estimate) as tx:
            result = self.client.chat(
                messages,
                max_tokens=self.definition.max_tokens_per_call,
                temperature=0.3,
            )
            tx.actual = result.total_tokens
        return result

    def _summarize(self, agent_id: str, transcript: str) -> ChatResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "Compress the following agent transcript into the fewest "
                    "sentences that preserve every concrete finding, number, "
                    "and decision. Drop reasoning and filler. No preamble."
                ),
            },
            {"role": "user", "content": transcript},
        ]
        estimate = sum(self.client.count_tokens(m["content"]) for m in messages)
        estimate += self.definition.summary_max_tokens
        with spend(self.budget, f"{agent_id}:summary", estimate) as tx:
            result = self.client.chat(
                messages,
                max_tokens=self.definition.summary_max_tokens,
                temperature=0.1,
            )
            tx.actual = result.total_tokens
        return result

    @staticmethod
    def _needs_another_turn(text: str) -> bool:
        return "FINAL:" not in text
