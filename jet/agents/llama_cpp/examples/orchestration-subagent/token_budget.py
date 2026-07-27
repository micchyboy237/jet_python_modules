"""
Token budget enforcement with a reservation pattern.

The naive approach -- "check remaining budget, then spend" -- breaks under
concurrency: if 5 subagents check the budget at the same instant, they can
all see the same "20K remaining" and all proceed, blowing the total.

The fix: reserve an *estimated* token cost atomically before the call runs,
then true-up (release unused tokens, or record overage) once the actual
usage is known. This file implements that for a single process using a
lock; swap the lock + dict for a Redis + Lua script if you need this
shared across multiple processes/machines.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    pass


@dataclass
class AgentUsage:
    reserved: int = 0
    spent: int = 0
    calls: int = 0


class TokenBudget:
    def __init__(self, total_budget: int, per_agent_budget: int | None = None):
        """
        total_budget:     hard ceiling across the whole orchestrator run.
        per_agent_budget: optional ceiling per individual agent/subagent id,
                           so one runaway subagent can't eat the whole run's budget.
        """
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self._lock = threading.Lock()
        self._total_reserved = 0
        self._total_spent = 0
        self._per_agent: dict[str, AgentUsage] = {}

    def _agent(self, agent_id: str) -> AgentUsage:
        if agent_id not in self._per_agent:
            self._per_agent[agent_id] = AgentUsage()
        return self._per_agent[agent_id]

    def reserve(self, agent_id: str, estimated_tokens: int) -> None:
        """Atomically reserve tokens before making an LLM call. Raises if it
        would blow the total or per-agent budget."""
        with self._lock:
            agent = self._agent(agent_id)

            if self.per_agent_budget is not None:
                agent_projected = agent.reserved + agent.spent + estimated_tokens
                if agent_projected > self.per_agent_budget:
                    raise BudgetExceededError(
                        f"[{agent_id}] would use {agent_projected} tokens, "
                        f"exceeding per-agent budget of {self.per_agent_budget}"
                    )

            total_projected = self._total_reserved + self._total_spent + estimated_tokens
            if total_projected > self.total_budget:
                raise BudgetExceededError(
                    f"Reserving {estimated_tokens} tokens for [{agent_id}] would push "
                    f"total to {total_projected}, exceeding budget of {self.total_budget}"
                )

            agent.reserved += estimated_tokens
            self._total_reserved += estimated_tokens

    def settle(self, agent_id: str, estimated_tokens: int, actual_tokens: int) -> None:
        """Release the reservation and record what was actually spent.
        Called after the LLM call completes (success or failure)."""
        with self._lock:
            agent = self._agent(agent_id)
            agent.reserved = max(0, agent.reserved - estimated_tokens)
            agent.spent += actual_tokens
            agent.calls += 1

            self._total_reserved = max(0, self._total_reserved - estimated_tokens)
            self._total_spent += actual_tokens

    def remaining_total(self) -> int:
        with self._lock:
            return self.total_budget - self._total_reserved - self._total_spent

    def remaining_for(self, agent_id: str) -> int | None:
        if self.per_agent_budget is None:
            return None
        with self._lock:
            agent = self._agent(agent_id)
            return self.per_agent_budget - agent.reserved - agent.spent

    def report(self) -> dict:
        with self._lock:
            return {
                "total_budget": self.total_budget,
                "total_spent": self._total_spent,
                "total_reserved": self._total_reserved,
                "total_remaining": self.total_budget - self._total_spent - self._total_reserved,
                "per_agent": {
                    aid: {"spent": u.spent, "reserved": u.reserved, "calls": u.calls}
                    for aid, u in self._per_agent.items()
                },
            }


class spend:
    """Context manager that reserves an estimate, then settles with actual
    usage on exit -- so callers can't forget to release a reservation.

        with spend(budget, agent_id="researcher-1", estimated_tokens=4000) as tx:
            result = client.chat(...)
            tx.actual = result.total_tokens
    """

    def __init__(self, budget: TokenBudget, agent_id: str, estimated_tokens: int):
        self.budget = budget
        self.agent_id = agent_id
        self.estimated_tokens = estimated_tokens
        self.actual: int = estimated_tokens  # default if caller forgets to set it

    def __enter__(self):
        self.budget.reserve(self.agent_id, self.estimated_tokens)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.budget.settle(self.agent_id, self.estimated_tokens, self.actual)
        return False  # don't suppress exceptions
