# jet_python_modules/jet/libs/smolagents/step_callbacks/memory_window.py

from __future__ import annotations  # recommended for Python < 3.10

from collections.abc import Callable
from typing import Any

from smolagents import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.monitoring import Timing


def memory_window_limiter(
    max_recent_steps: int = 8,
    keep_final: bool = True,
    keep_system_and_task: bool = True,
    keep_last_plans: int = 1,
    insert_placeholder: bool = True,
    warn_at_steps: int = 12,
    force_truncate_at: int = 20,
) -> Callable[[MemoryStep, Any | None], None]:
    """
    Step callback factory: sliding window memory limiter.
    Keeps recent steps + critical fixed steps (system, initial task, final answer).
    Optionally preserves recent PlanningSteps.
    """

    def callback(memory_step: MemoryStep, agent: Any | None = None) -> None:
        if (
            not agent
            or not hasattr(agent, "memory")
            or not hasattr(agent.memory, "steps")
        ):
            return

        steps = agent.memory.steps
        current_len = len(steps)

        if current_len <= max_recent_steps + 4:
            return  # no need to trim yet

        # Collect steps we want to unconditionally keep
        preserved = []

        if keep_system_and_task:
            for step in steps:
                if isinstance(step, (SystemPromptStep, TaskStep)):
                    preserved.append(step)
                if len(preserved) >= 2:  # usually first two
                    break

        if keep_final:
            finals = [s for s in steps if isinstance(s, FinalAnswerStep)]
            if finals:
                preserved.extend(finals)  # normally 0 or 1

        # Optionally keep last N planning steps
        if keep_last_plans > 0:
            plans = [s for s in steps if isinstance(s, PlanningStep)][-keep_last_plans:]
            preserved.extend(plans)

        # Recent steps (sliding window)
        recent_start = max(0, current_len - max_recent_steps)
        recent = steps[recent_start:]

        # Build new list: preserved (in original order) + recent (deduped)
        new_steps = []
        preserved_ids = {id(s) for s in preserved}
        new_steps.extend(preserved)

        for s in recent:
            if id(s) not in preserved_ids:
                new_steps.append(s)

        # Insert lightweight placeholder if truncation occurred
        if len(new_steps) < current_len and insert_placeholder:
            placeholder = ActionStep(
                step_number=-1,
                timing=Timing(start_time=0.0, end_time=0.0),
                model_output="[Memory window applied — older steps truncated to prevent overflow]",
                observations=None,
                code_action=None,
                tool_calls=None,
            )
            # Insert after preserved items
            insert_pos = len(preserved) if preserved else 0
            new_steps.insert(insert_pos, placeholder)

        # Apply
        agent.memory.steps = new_steps

        dropped = current_len - len(new_steps)
        if (
            dropped > 0
            and hasattr(agent, "verbosity_level")
            and agent.verbosity_level >= 1
        ):
            print(
                f"[Memory limiter] Dropped {dropped} old steps → {len(new_steps)} remaining"
            )

        if (
            current_len >= warn_at_steps
            and hasattr(agent, "verbosity_level")
            and agent.verbosity_level >= 1
        ):
            print(
                f"[Memory warning] Step count high: {current_len} (warn threshold {warn_at_steps})"
            )

        if current_len >= force_truncate_at:
            print(
                f"[Memory force-truncate] Hard limit {force_truncate_at} exceeded → truncated"
            )

    return callback
