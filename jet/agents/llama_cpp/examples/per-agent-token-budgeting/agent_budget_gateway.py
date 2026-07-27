"""
Per-agent token budget gateway for a local llama.cpp server.

This is the piece the earlier orchestrator/subagent code didn't have:
production-style per-agent budget *enforcement*, not just bookkeeping.
Every agent's calls go through `gateway.call(agent_id, ...)` -- there is no
path to the model that skips the gateway, so agents structurally cannot
bypass their budget (the same reasoning production teams use for putting
this at an API gateway rather than trusting each agent's own code).

Four mechanisms compose here, each catching a different failure mode:

1. Per-agent hard budget + shared reserve with burst borrowing
   -> caps total spend per agent, but lets an agent temporarily borrow from
      a shared pool instead of hard-failing the moment its own slice is
      gone. Trade-off: isolation vs utilization, tunable via reserve size.

2. Per-agent token bucket (rate limit)
   -> catches an agent trying to burn tokens too *fast* (tight retry loop,
      runaway generation loop) even while still under its total budget.

3. Per-agent circuit breaker
   -> if an agent keeps failing or getting rejected, stop calling out on
      its behalf for a cooldown window instead of hammering the server.

4. Cost/usage attribution
   -> per-agent-id ledger so you can answer "which agent/session spent
      what" after the fact, independent of whether it hit any limit.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from llama_client import LlamaCppClient, ChatResult
from token_bucket import TokenBucket, RateLimitedError
from circuit_breaker import CircuitBreaker, CircuitOpenError


class BudgetExceededError(Exception):
    pass


@dataclass
class AgentTier:
    """Config for a class of agent (e.g. 'lead', 'worker', 'verifier').
    Different tiers can get different allocations and rate limits, the way
    a production gateway routes cheap/expensive model access by API-key tier."""
    name: str
    hard_budget: int              # tokens this agent can spend total, own allocation
    tpm: int                      # tokens-per-minute sustained rate
    burst_capacity: int           # bucket capacity (max instantaneous burst)
    max_tokens_per_call: int = 800
    cost_per_1k_tokens: float = 0.0   # for attribution; 0 for a fully local model


@dataclass
class _AgentLedger:
    own_remaining: int
    borrowed: int = 0
    spent: int = 0
    calls: int = 0
    rejections: int = 0


class AgentBudgetGateway:
    def __init__(
        self,
        client: LlamaCppClient,
        shared_reserve: int = 20_000,
        default_tier: AgentTier | None = None,
    ):
        self.client = client
        self.default_tier = default_tier or AgentTier(
            name="default", hard_budget=10_000, tpm=6_000, burst_capacity=3_000
        )
        self._shared_reserve_total = shared_reserve
        self._shared_reserve_remaining = shared_reserve

        self._tiers: dict[str, AgentTier] = {}          # agent_id -> tier
        self._ledgers: dict[str, _AgentLedger] = {}
        self._buckets: dict[str, TokenBucket] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    # ---- registration --------------------------------------------------

    def register(self, agent_id: str, tier: AgentTier | None = None) -> None:
        tier = tier or self.default_tier
        with self._lock:
            self._tiers[agent_id] = tier
            self._ledgers[agent_id] = _AgentLedger(own_remaining=tier.hard_budget)
            self._buckets[agent_id] = TokenBucket(
                capacity=tier.burst_capacity, refill_per_second=tier.tpm / 60.0
            )
            self._breakers[agent_id] = CircuitBreaker(failure_threshold=3, cooldown_s=15.0)

    def _ensure_registered(self, agent_id: str) -> None:
        if agent_id not in self._tiers:
            self.register(agent_id)

    # ---- budget reservation with burst borrowing ------------------------

    def _reserve(self, agent_id: str, amount: int) -> tuple[int, int]:
        """Returns (from_own, from_shared). Raises BudgetExceededError if
        neither the agent's own allocation nor the shared reserve can cover it."""
        with self._lock:
            ledger = self._ledgers[agent_id]
            if ledger.own_remaining >= amount:
                ledger.own_remaining -= amount
                return amount, 0

            shortfall = amount - ledger.own_remaining
            if self._shared_reserve_remaining >= shortfall:
                from_own = ledger.own_remaining
                ledger.own_remaining = 0
                self._shared_reserve_remaining -= shortfall
                ledger.borrowed += shortfall
                return from_own, shortfall

            ledger.rejections += 1
            raise BudgetExceededError(
                f"[{agent_id}] needs {amount} tokens "
                f"(has {ledger.own_remaining} own, {self._shared_reserve_remaining} shared "
                f"available) -- budget exhausted"
            )

    def _settle(self, agent_id: str, from_own: int, from_shared: int, actual: int) -> None:
        with self._lock:
            ledger = self._ledgers[agent_id]
            reserved_total = from_own + from_shared

            if actual <= reserved_total:
                unused = reserved_total - actual
                # release shared first, so borrowed capacity doesn't linger
                release_shared = min(unused, from_shared)
                release_own = unused - release_shared
                self._shared_reserve_remaining += release_shared
                ledger.borrowed -= release_shared
                ledger.own_remaining += release_own
            else:
                # ran over estimate (rare, since max_tokens caps output) --
                # absorb the overage from the shared reserve as a last resort
                overage = actual - reserved_total
                self._shared_reserve_remaining -= overage
                ledger.borrowed += overage

            ledger.spent += actual
            ledger.calls += 1

    # ---- the actual call path -------------------------------------------

    def call(self, agent_id: str, messages: list[dict], tier: AgentTier | None = None) -> ChatResult:
        self._ensure_registered(agent_id)
        if tier is not None and agent_id not in self._tiers:
            self.register(agent_id, tier)
        agent_tier = self._tiers[agent_id]
        breaker = self._breakers[agent_id]
        bucket = self._buckets[agent_id]

        breaker.before_call()  # raises CircuitOpenError if tripped

        estimate = sum(self.client.count_tokens(m["content"]) for m in messages)
        estimate += agent_tier.max_tokens_per_call

        try:
            bucket.consume_or_wait(estimate, max_wait_s=10.0)
        except RateLimitedError:
            breaker.on_failure()
            raise

        try:
            from_own, from_shared = self._reserve(agent_id, estimate)
        except BudgetExceededError:
            breaker.on_failure()
            raise

        try:
            result = self.client.chat(messages, max_tokens=agent_tier.max_tokens_per_call)
        except Exception:
            # release the reservation since no tokens were actually spent
            self._settle(agent_id, from_own, from_shared, actual=0)
            breaker.on_failure()
            raise

        self._settle(agent_id, from_own, from_shared, actual=result.total_tokens)
        breaker.on_success()
        return result

    # ---- attribution / reporting -----------------------------------------

    def report(self) -> dict:
        with self._lock:
            per_agent = {}
            total_cost = 0.0
            for agent_id, ledger in self._ledgers.items():
                tier = self._tiers[agent_id]
                cost = (ledger.spent / 1000.0) * tier.cost_per_1k_tokens
                total_cost += cost
                per_agent[agent_id] = {
                    "tier": tier.name,
                    "spent_tokens": ledger.spent,
                    "own_remaining": ledger.own_remaining,
                    "borrowed_from_shared": ledger.borrowed,
                    "rejections": ledger.rejections,
                    "calls": ledger.calls,
                    "circuit_state": self._breakers[agent_id].state.value,
                    "estimated_cost": round(cost, 4),
                }
            return {
                "shared_reserve_total": self._shared_reserve_total,
                "shared_reserve_remaining": self._shared_reserve_remaining,
                "total_estimated_cost": round(total_cost, 4),
                "agents": per_agent,
            }
