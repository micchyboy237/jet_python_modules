# custom_memory.py
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

from jet.adapters.llama_cpp.tokens import count_tokens  # your existing token counter
from jet.file.utils import save_file
from jet.transformers.object import make_serializable
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name

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

SUMMARY_PROMPT = """You are a structured memory compressor for an AI agent.

Your job is to condense previous steps into a compact, information-dense summary that preserves all important context.

Guidelines:
- Extract concrete facts: names, values, identifiers, results, outputs, errors.
- Preserve important observations from tools or actions.
- Capture decisions, reasoning outcomes, and progress made.
- Include relevant context that may be needed for future steps.
- Avoid unnecessary verbosity or repetition.
- Prefer structured, scannable bullets.

Output EXACTLY in this structure:

Previous steps summary ({old_steps_length} steps):

Key Facts:
- specific factual data (IDs, values, results, etc.)
- ...

Important Decisions & Progress:
- decisions made and why
- completed actions or milestones
- ...

Errors & Issues:
- brief description of failures or problems encountered (if any)
- ...

Open Items / Next Steps:
- pending tasks or unresolved questions
- ...

Raw steps:
{raw_steps}
"""

MAX_RAW_STEPS_TOKENS = 6000  # leave room for instructions + output


def get_next_call_number(logs_dir: Path) -> int:
    """Find the next available call number using 4-digit prefix (aligned with save_step_state)."""
    if not logs_dir.exists():
        return 1
    existing = [
        int(d.name.split("_")[0])
        for d in logs_dir.iterdir()
        if d.is_dir() and len(d.name) >= 5 and d.name[4] == "_" and d.name[:4].isdigit()
    ]
    return max(existing, default=0) + 1


def get_llm_call_subdir(
    logs_dir: Path,
) -> Path:
    # prefix = "generate_stream" if is_stream else "generate"
    call_number = get_next_call_number(logs_dir)
    prefix = f"{call_number:04d}"

    cleaned = "memory"

    subdir_name = f"{prefix}_{cleaned}"
    target_dir = logs_dir / subdir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


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
        self.max_tokens_before_compress: int = 8000  # slightly more aggressive
        self.keep_recent_steps: int = 10  # keep more recent detail

        self.facts: Dict[
            str, Any
        ] = {}  # persistent key facts (e.g., {"current_year": 2026, "top_anime": [...]})

        self.logs_dir = (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "agent_memory"
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget."""
        if not text:
            return text

        # Fast path
        try:
            tokens = count_tokens([{"role": "user", "content": text}], model=None)
            if tokens <= max_tokens:
                return text
        except Exception:
            pass

        # Progressive truncation (binary-ish reduction)
        left, right = 0, len(text)
        result = text

        while left < right:
            mid = (left + right) // 2
            candidate = text[:mid]

            try:
                tokens = count_tokens(
                    [{"role": "user", "content": candidate}], model=None
                )
            except Exception:
                # fallback to char heuristic
                tokens = len(candidate) // 4

            if tokens <= max_tokens:
                result = candidate
                left = mid + 1
            else:
                right = mid - 1

        return result

    def format_summary_prompt(self, old_steps: list) -> str:
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

        truncated_raw_text = self._truncate_to_token_limit(
            raw_text,
            max_tokens=MAX_RAW_STEPS_TOKENS,
        )

        summary_prompt = SUMMARY_PROMPT.format(
            old_steps_length=len(old_steps),
            raw_steps=truncated_raw_text,
        )

        return summary_prompt

    def _generate_structured_summary(
        self, old_steps: list, model: Any = None
    ) -> Optional[str]:
        if not old_steps:
            return None

        summary_prompt = self.format_summary_prompt(old_steps)

        # if model is None:
        #     # improved fallback
        #     truncated = raw_text[:3000]
        #     return (
        #         f"Previous steps summary ({len(old_steps)} steps):\n"
        #         f"Extracted Facts:\n- (see raw observations below)\n\n"
        #         f"Key Observations:\n{truncated[:1500]}\n\n"
        #         f"(Details compressed...)"
        #     )

        try:
            response = model.call(
                summary_prompt, temperature=0.2
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

        summary_prompt = self.format_summary_prompt(old_steps)
        input_tokens = count_tokens(summary_prompt, model=None)

        request_data = {
            "token_counts": {
                "input_tokens": input_tokens,
                "steps_tokens": total_tokens,
            },
            "old_steps": make_serializable(old_steps),
        }
        target_dir = get_llm_call_subdir(self.logs_dir)
        save_file(request_data, target_dir / "request.json")

        summary = self._generate_structured_summary(old_steps, model_for_summary)

        if summary:
            compressed = CompressedSummaryStep(
                summary=summary,
                original_step_count=len(old_steps),
            )
            output_tokens = count_tokens(summary, model=None)

            self.steps = [compressed] + recent

            response_data = {
                "prompt": summary_prompt,
                "token_counts": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "steps_tokens": self._get_total_tokens(),
                },
                "summary": summary,
            }
            save_file(response_data, target_dir / "response.json")

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
