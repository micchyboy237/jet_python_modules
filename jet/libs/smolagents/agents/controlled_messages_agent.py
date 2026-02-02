"""
controlled_agents_with_memory_control_example.py

Demonstration of two controlled-memory agents:
- ControlledCodeAgent      (inherits from CodeAgent)
- ControlledToolCallingAgent (inherits from ToolCallingAgent)

Both use the same memory truncation / summarization controllers.

Requires:
    smolagents
    rich
    (and a model backend that smolagents supports, e.g. huggingface_hub)

2026-style example — using @tool decorator where available
"""

from __future__ import annotations

from dataclasses import dataclass

from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from rich.console import Console
from rich.panel import Panel

# ────────────────────────────────────────────────
# smolagents imports — adjust paths/names to match your installation
# ────────────────────────────────────────────────
from smolagents import tool
from smolagents.agents import (
    ChatMessage,
    CodeAgent,
    MessageRole,
    MultiStepAgent,
    ToolCallingAgent,
)
from smolagents.memory import (
    ActionStep,
    AgentMemory,
    PlanningStep,
    TaskStep,
)

console = Console()


# ────────────────────────────────────────────────
#           Memory / Message Controllers
# ────────────────────────────────────────────────


class MessageController:
    def get_messages_for_llm(
        self,
        full_memory: AgentMemory,
        current_step_number: int,
    ) -> list[ChatMessage]:
        raise NotImplementedError


@dataclass
class LastNTurnsController(MessageController):
    """Keep only the most recent N turns (very predictable token count)"""

    keep_last_turns: int = 6

    def get_messages_for_llm(
        self,
        full_memory: AgentMemory,
        current_step_number: int,
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = []

        # System prompt
        messages.extend(full_memory.system_prompt.to_messages())

        # Task message (usually the second item)
        if full_memory.steps and isinstance(full_memory.steps[0], TaskStep):
            messages.extend(full_memory.steps[0].to_messages())

        # Recent steps only
        recent_steps = full_memory.steps[-self.keep_last_turns :]
        for step in recent_steps:
            if isinstance(step, (ActionStep, PlanningStep)):
                messages.extend(step.to_messages(summary_mode=False))

        console.print(
            Panel.fit(
                f"[dim]LastNTurns — keeping last {len(recent_steps)} steps[/dim]",
                style="cyan",
            )
        )

        return messages


@dataclass
class SummaryPlusRecentController(MessageController):
    """Rolling summary + recent turns — better coherence on longer tasks"""

    summary_every_n_steps: int = 8
    keep_last_turns: int = 5
    max_summary_chars: int = 1400

    _last_summary: str | None = None
    _last_summary_step: int = 0

    def get_messages_for_llm(
        self,
        full_memory: AgentMemory,
        current_step_number: int,
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = []

        # System + task
        messages.extend(full_memory.system_prompt.to_messages())
        if full_memory.steps and isinstance(full_memory.steps[0], TaskStep):
            messages.extend(full_memory.steps[0].to_messages())

        # Refresh summary?
        if (
            current_step_number >= 5
            and current_step_number - self._last_summary_step
            >= self.summary_every_n_steps
        ):
            summary_text = self._build_summary(full_memory)
            self._last_summary = summary_text
            self._last_summary_step = current_step_number

            preview = (
                summary_text[:350] + "…" if len(summary_text) > 350 else summary_text
            )
            console.print(
                Panel(
                    preview,
                    title=f"Summary updated @ step {current_step_number}",
                    style="magenta dim",
                    expand=False,
                )
            )

        # Insert current summary if any
        if self._last_summary:
            messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"### Summary of earlier steps:\n{self._last_summary}",
                )
            )

        # Most recent detailed turns
        recent = full_memory.steps[-self.keep_last_turns :]
        for step in recent:
            if isinstance(step, (ActionStep, PlanningStep)):
                messages.extend(step.to_messages(summary_mode=False))

        console.print(
            Panel.fit(
                f"[dim]Summary + {len(recent)} recent detailed turns[/dim]",
                style="cyan",
            )
        )

        return messages

    def _build_summary(self, memory: AgentMemory) -> str:
        parts = []
        for step in memory.steps:
            if isinstance(step, TaskStep):
                parts.append(f"Task: {step.task[:180]}…")
            elif isinstance(step, ActionStep):
                if step.model_output:
                    parts.append(f"Thought: {step.model_output[:250]}…")
                if step.observations:
                    parts.append(f"Observation: {step.observations[:350]}…")
                if step.action_output and "final" in str(step.action_output).lower():
                    parts.append(f"Answer: {str(step.action_output)[:250]}…")
            elif isinstance(step, PlanningStep):
                parts.append(f"Plan: {step.plan[:250]}…")

        text = "\n".join(parts).strip()
        if len(text) > self.max_summary_chars:
            text = text[: self.max_summary_chars - 3] + "…"
        return text


# ────────────────────────────────────────────────
#         Controlled Agents (both styles)
# ────────────────────────────────────────────────


class MessageControlledAgentBase(MultiStepAgent):
    """Base class that delegates message selection to a controller"""

    def __init__(self, *, message_controller: MessageController, **kwargs):
        super().__init__(**kwargs)
        self.message_controller = message_controller

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        return self.message_controller.get_messages_for_llm(
            full_memory=self.memory,
            current_step_number=self.step_number,
        )


class ControlledCodeAgent(MessageControlledAgentBase, CodeAgent):
    """CodeAgent with controlled context window"""

    def __init__(self, *, message_controller: MessageController, **kwargs):
        CodeAgent.__init__(self, **kwargs)
        self.message_controller = message_controller


class ControlledToolCallingAgent(MessageControlledAgentBase, ToolCallingAgent):
    """ToolCallingAgent with controlled context window"""

    def __init__(self, *, message_controller: MessageController, **kwargs):
        ToolCallingAgent.__init__(self, **kwargs)
        self.message_controller = message_controller


# ────────────────────────────────────────────────
#                    Example Tools
# ────────────────────────────────────────────────


@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First factor
        b: Second factor
    """
    return a * b


@tool
def fake_web_search(query: str) -> str:
    """Mock web search — returns canned answers for demo purposes.

    Args:
        query: Search term or question
    """
    query = query.lower()
    if "capital" in query and "france" in query:
        return "The capital of France is Paris."
    if "population" in query and "france" in query:
        return "France has approximately 68 million inhabitants in 2025."
    if "day of the week" in query and "1995" in query and "march 15" in query:
        return "March 15, 1995 was a Wednesday."
    return f"No specific info found for: {query}"


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 4096,
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    agent_name: str | None = None,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
    )


# ────────────────────────────────────────────────
#                     Usage Demo
# ────────────────────────────────────────────────

if __name__ == "__main__":
    model = create_local_model()

    tools = [add, multiply, fake_web_search]

    # ── Controllers ──
    strict = LastNTurnsController(keep_last_turns=5)
    smart = SummaryPlusRecentController(summary_every_n_steps=6, keep_last_turns=4)

    # ── Code-style agent ──
    code_agent = ControlledCodeAgent(
        model=model,
        tools=tools,
        message_controller=smart,
        max_steps=25,
        additional_authorized_imports=["math"],
    )

    console.rule("Code-style agent demo")
    answer1 = code_agent.run("What is (17 ** 4 + 92) × 3? Please compute step by step.")
    console.print("\n[bold green]Code agent final answer:[/]", answer1, "\n")

    # ── Tool-calling agent ──
    tool_agent = ControlledToolCallingAgent(
        model=model,
        tools=tools,
        message_controller=strict,
        max_steps=25,
    )

    console.rule("Tool-calling agent demo")
    answer2 = tool_agent.run(
        "Find the capital of France using the tools, "
        "then tell me how many letters are in its name, "
        "and finally multiply that number by 7."
    )
    console.print("\n[bold green]Tool-calling agent final answer:[/]", answer2, "\n")
