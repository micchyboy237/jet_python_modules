# custom_memory.py
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, Optional

from jet.adapters.llama_cpp.tokens import count_tokens  # your existing token counter

# Only import for runtime (not for typing in this file) from smolagents.memory
from smolagents.memory import (
    ActionStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.models import (
    ChatMessage,
    MessageRole,
)
from smolagents.monitoring import AgentLogger, LogLevel, Timing

__all__ = ["AgentMemory", "CompressedSummaryStep"]

logger = getLogger(__name__)


@dataclass
class CompressedSummaryStep(MemoryStep):
    """Lightweight step that replaces many old steps with a structured summary."""

    summary: str
    original_step_count: int
    timing: Optional[Timing] = None

    def dict(self):
        return {
            "type": "compressed_summary",
            "summary": self.summary,
            "original_step_count": self.original_step_count,
            "timing": self.timing.dict() if self.timing else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": f"Summary of previous steps:\n{self.summary}",
                    }
                ],
            )
        ]


class AgentMemory:
    """
    Improved Memory for the agent with automatic context compression.

    This version intelligently compresses older steps to prevent context overflow
    while trying to preserve important facts and decisions.

    Args:
        system_prompt (str): System prompt for the agent.

    Attributes:
        system_prompt (SystemPromptStep): The initial system instructions.
        steps (list): List of all steps (including compressed summaries).
        max_tokens_before_compress (int): Trigger compression when tokens exceed this.
        keep_recent_steps (int): Always keep the last N steps in full detail.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt: SystemPromptStep = SystemPromptStep(
            system_prompt=system_prompt
        )
        self.steps: list[
            TaskStep | ActionStep | PlanningStep | CompressedSummaryStep
        ] = []
        self.max_tokens_before_compress: int = 10500  # slightly more aggressive
        self.keep_recent_steps: int = 10  # keep more recent detail

        self.facts: Dict[
            str, Any
        ] = {}  # persistent key facts (e.g., {"current_year": 2026, "top_anime": [...]})

    def reset(self):
        """Reset the agent's memory, clearing all steps but keeping the system prompt."""
        self.steps = []
        self.facts = {}

    def _get_total_tokens(self) -> int:
        """Estimate total tokens in the messages that would be sent to the model."""
        messages: list[dict] = []
        for step in self.steps:
            try:
                step_msgs = step.to_messages(summary_mode=False)
                for msg in step_msgs:
                    # Normalize for token counting (handles images, etc.)
                    content = msg.content
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        normalized = "\n".join(text_parts)
                    else:
                        normalized = str(content)
                    messages.append({"role": msg.role.value, "content": normalized})
            except Exception:
                continue  # skip problematic steps

        try:
            return count_tokens(messages, model=None)
        except Exception:
            # Very rough fallback
            return len(str(messages)) // 4

    def _generate_structured_summary(self, old_steps: list, model: Any = None) -> str:
        if not old_steps:
            return "No previous steps."

        raw_text_parts = []
        for s in old_steps:
            step_type = s.__class__.__name__
            step_num = getattr(s, "step_number", "N/A")
            if isinstance(s, ActionStep):
                obs = getattr(s, "observations", "") or ""
                output = getattr(s, "action_output", "") or ""
                # Better: show more observation content + try to highlight results
                raw_text_parts.append(
                    f"Step {step_num} (Action):\n"
                    f"Observation (key results):\n{obs[:1200] if len(obs) > 100 else obs}\n"
                    f"Action Output: {str(output)[:400]}..."
                )
            elif isinstance(s, PlanningStep):
                raw_text_parts.append(
                    f"Step {step_num} (Plan):\n{getattr(s, 'plan', '')[:1000]}..."
                )
            else:
                raw_text_parts.append(
                    f"Step {step_num} ({step_type}): {str(s)[:500]}..."
                )

        raw_text = "\n\n".join(raw_text_parts)

        if model is None:
            # improved fallback
            truncated = raw_text[:3000]
            return (
                f"Previous steps summary ({len(old_steps)} steps):\n"
                f"Extracted Facts:\n- (see raw observations below)\n\n"
                f"Key Observations:\n{truncated[:1500]}\n\n"
                f"(Details compressed...)"
            )

        try:
            summary_prompt = f"""You are a precise memory compressor for an AI agent.

Extract and preserve all concrete facts, numbers, names, dates, search results, URLs, and key observations. Do NOT focus only on tool-calling methods.

Guidelines:
- Prioritize: current year, anime titles, release dates, plots, sources, any numbers or specific data found.
- Include successful tool outputs and important observations verbatim when short.
- Note errors briefly.
- Structure exactly like this:

Previous steps summary ({len(old_steps)} steps):

Key Facts:
- bullet with specific data (e.g., "Current year: 2026")
- bullet with anime info...

Important Decisions & Progress:
- ...

Open Items:
- ...

Raw steps to summarize:
{raw_text[:7500]}
"""

            response = model.call(
                summary_prompt, max_tokens=900, temperature=0.2
            )  # lower temp for factualness
            summary_text = (
                response.strip() if isinstance(response, str) else str(response)
            )

            # Optional: try to parse and update self.facts (if model returns JSON section)
            # For now, keep simple

            return summary_text or "Summary failed."

        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}. Falling back...")
            return self._generate_structured_summary(old_steps, model=None)

    def compress_old_steps(self, agent=None):
        """Compress older steps if the context is getting too large.

        Pass agent (or agent.model) to enable high-quality LLM summarization.
        """
        if len(self.steps) <= self.keep_recent_steps + 4:
            return  # too few steps

        total_tokens = self._get_total_tokens()
        if total_tokens < self.max_tokens_before_compress:
            return

        logger.info(
            f"Compressing memory: ~{total_tokens} tokens. "
            f"Keeping last {self.keep_recent_steps} steps full detail."
        )

        recent = self.steps[-self.keep_recent_steps :]
        old_steps = self.steps[: -self.keep_recent_steps]

        if not old_steps:
            return

        # Optional: clear old images to save tokens
        for step in old_steps:
            if isinstance(step, ActionStep) and hasattr(step, "observations_images"):
                step.observations_images = None

        # Generate structured summary
        model_for_summary = getattr(agent, "model", None) if agent else None
        summary = self._generate_structured_summary(old_steps, model_for_summary)

        compressed = CompressedSummaryStep(
            summary=summary,
            original_step_count=len(old_steps),
        )

        self.steps = [compressed] + recent

    def add_fact(self, key: str, value: Any):
        """Manually or via callback: store persistent important facts."""
        self.facts[key] = value
        logger.info(f"Added persistent fact: {key} = {value}")

    def get_facts_summary(self) -> str:
        if not self.facts:
            return ""
        return "Persistent Facts:\n" + "\n".join(
            f"- {k}: {v}" for k, v in self.facts.items()
        )

    def get_succinct_steps(self) -> list[dict]:
        """Return succinct steps (excluding heavy model_input_messages)."""
        return [
            {
                key: value
                for key, value in step.dict().items()
                if key != "model_input_messages"
            }
            for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """Return full steps including everything."""
        if not self.steps:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Pretty replay of all steps, including compressed summaries."""
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(
            title="System prompt",
            content=self.system_prompt.system_prompt,
            level=LogLevel.ERROR,
        )
        if self.facts:
            logger.log_markdown(
                title="Persistent Facts",
                content=self.get_facts_summary(),
                level=LogLevel.ERROR,
            )
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(
                    f"Step {getattr(step, 'step_number', 'N/A')}", level=LogLevel.ERROR
                )
                if detailed and getattr(step, "model_input_messages", None):
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if getattr(step, "model_output", None):
                    logger.log_markdown(
                        title="Agent output:",
                        content=step.model_output,
                        level=LogLevel.ERROR,
                    )
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and getattr(step, "model_input_messages", None):
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(
                    title="Agent output:", content=step.plan, level=LogLevel.ERROR
                )
            elif isinstance(step, CompressedSummaryStep):
                logger.log_markdown(
                    title=f"Compressed Summary ({step.original_step_count} steps)",
                    content=step.summary,
                    level=LogLevel.ERROR,
                )

    def return_full_code(self) -> str:
        """Return all generated code actions concatenated."""
        return "\n\n".join(
            [
                getattr(step, "code_action", "")
                for step in self.steps
                if isinstance(step, ActionStep) and getattr(step, "code_action", None)
            ]
        )
