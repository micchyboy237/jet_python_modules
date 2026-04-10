# custom_memory.py
from dataclasses import dataclass
from logging import getLogger
from typing import Optional

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

__all__ = ["AgentMemory"]

logger = getLogger(__name__)


@dataclass
class CompressedSummaryStep(MemoryStep):
    """Lightweight step that replaces many old steps with a short summary."""

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
    Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tasks, actions, planning steps, and compressed summaries.
    It can automatically compress older steps to summaries when the conversation context gets too long.

    Args:
        system_prompt (str): System prompt for the agent, which sets the context and instructions for the agent's behavior.

    Attributes:
        system_prompt (SystemPromptStep): System prompt step for the agent.
        steps (list): List of steps taken by the agent (TaskStep, ActionStep, PlanningStep, CompressedSummaryStep).
        max_tokens_before_compress (int): When memory tokens go above this, will compress older steps.
        keep_recent_steps (int): Minimum number of recent steps to always keep at full detail.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt: SystemPromptStep = SystemPromptStep(
            system_prompt=system_prompt
        )
        self.steps: list[
            TaskStep | ActionStep | PlanningStep | CompressedSummaryStep
        ] = []
        self.max_tokens_before_compress: int = 12000  # tunable threshold
        self.keep_recent_steps: int = 8  # always keep last N steps in detail

    def reset(self):
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def _get_total_tokens(self) -> int:
        """Rough token count of all messages that would be sent to the model."""
        messages = []
        for step in self.steps:
            messages.extend(step.to_messages(summary_mode=False))
        try:
            return count_tokens(messages, model=None)
        except Exception:
            # fallback very rough estimation
            return len(str(messages)) // 4

    def compress_old_steps(self, agent=None):
        """Compress older steps into a summary if context is getting too large."""
        if len(self.steps) <= self.keep_recent_steps + 5:
            # Not enough steps to bother compressing
            return

        total_tokens = self._get_total_tokens()
        if total_tokens < self.max_tokens_before_compress:
            return

        logger.info(
            f"Compressing memory: {total_tokens} tokens detected. Keeping last {self.keep_recent_steps} steps full."
        )

        # Keep recent steps untouched, summarize older ones
        recent = self.steps[-self.keep_recent_steps :]
        old_steps = self.steps[: -self.keep_recent_steps]

        if not old_steps:
            return

        # Concatenate some observables from older steps for summary
        old_text = "\n\n".join(
            f"Step {getattr(s, 'step_number', None) or 'planning'}: {getattr(s, 'observations', '') or getattr(s, 'plan', '')[:500]}"
            for s in old_steps
            if hasattr(s, "observations") or hasattr(s, "plan")
        )
        summary = (
            f"Previous steps summary ({len(old_steps)} steps): Key facts and progress: "
            f"{old_text[:2000]}... (details compressed to save context)"
        )

        compressed = CompressedSummaryStep(
            summary=summary,
            original_step_count=len(old_steps),
        )

        self.steps = [compressed] + recent

    def get_succinct_steps(self) -> list[dict]:
        """
        Return a succinct representation of the agent's steps, excluding model input messages.
        """
        return [
            {
                key: value
                for key, value in step.dict().items()
                if key != "model_input_messages"
            }
            for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """
        Return a full representation of the agent's steps, including model input messages.
        """
        if not self.steps:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """
        Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, default False): If True, also displays the memory at each step. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(
            title="System prompt",
            content=self.system_prompt.system_prompt,
            level=LogLevel.ERROR,
        )
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and getattr(step, "model_input_messages", None) is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if getattr(step, "model_output", None) is not None:
                    logger.log_markdown(
                        title="Agent output:",
                        content=step.model_output,
                        level=LogLevel.ERROR,
                    )
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and getattr(step, "model_input_messages", None) is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(
                    title="Agent output:", content=step.plan, level=LogLevel.ERROR
                )
            elif isinstance(step, CompressedSummaryStep):
                logger.log_markdown(
                    title="(Steps compressed to summary)",
                    content=step.summary,
                    level=LogLevel.ERROR,
                )

    def return_full_code(self) -> str:
        """
        Returns all code actions from the agent's steps, concatenated as a single script.
        """
        return "\n\n".join(
            [
                step.code_action
                for step in self.steps
                if isinstance(step, ActionStep)
                and getattr(step, "code_action", None) is not None
            ]
        )
