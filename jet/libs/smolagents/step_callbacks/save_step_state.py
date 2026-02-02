# jet_python_modules/jet/libs/smolagents/step_callbacks/save_step_state.py
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from smolagents import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)


def get_next_run_step_number(base_dir: Path) -> int:
    """
    Find the next available global step number in base_dir.
    Files are expected to start with four digits followed by underscore.
    """
    if not base_dir.exists():
        return 1

    numbers = []
    for f in base_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if len(name) >= 5 and name[4] == "_" and name[:4].isdigit():
            try:
                num = int(name[:4])
                numbers.append(num)
            except ValueError:
                continue

    return max(numbers, default=0) + 1 if numbers else 1


def save_step_state(
    agent_name: str | None = None,
    base_dir: str | Path | None = None,
    save_images: bool = False,  # currently not implemented — placeholder for future
) -> Callable[[MemoryStep, Any], None]:
    """
    Factory that creates a step callback which saves step information to disk.

    The returned callback saves only fields that actually exist in the smolagents
    MemoryStep dataclasses (ActionStep, PlanningStep, TaskStep, etc.).

    Files are saved as:
      {base_dir}/{cleaned_agent_name}/0001_task_my_agent.json
      {base_dir}/{cleaned_agent_name}/0003_action_my_agent.json
      {base_dir}/{cleaned_agent_name}/0005_plan_my_agent.json
      {base_dir}/{cleaned_agent_name}/0007_final_my_agent.json
      ...

    Recommended usage:
        from jet.libs.smolagents.step_callbacks.save_step_state import save_step_state

        callback = save_step_state(agent_name="researcher-v3")
        agent = CodeAgent(..., step_callbacks=[callback])
    """
    cleaned_name = (
        str(agent_name).strip().replace(" ", "_").replace("-", "_").lower()
        if agent_name
        else "agent"
    )

    # Default location: next to calling script, in generated/.../agent_tool_runs/
    caller_base_dir = (
        Path(get_entry_file_dir()) / "generated" / Path(get_entry_file_name()).stem
    ).resolve()
    caller_base_dir = caller_base_dir / "agent_tool_runs"

    run_dir = Path(base_dir).resolve() if base_dir else caller_base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    def callback(memory_step: MemoryStep, agent: Any = None) -> None:
        nonlocal run_dir

        # ── Determine prefix & filename style ────────────────────────────────
        if isinstance(memory_step, TaskStep):
            prefix = "task"
        elif isinstance(memory_step, ActionStep):
            prefix = "action"
        elif isinstance(memory_step, PlanningStep):
            prefix = "plan"
        elif isinstance(memory_step, FinalAnswerStep):
            prefix = "final"
        elif isinstance(memory_step, SystemPromptStep):
            prefix = "system"
        else:
            prefix = memory_step.__class__.__name__.lower().replace("step", "")
            if not prefix:
                prefix = "step"

        step_number = get_next_run_step_number(run_dir)
        base_name = f"{step_number:04d}_{prefix}_{cleaned_name}"
        normalized_name = "_".join(filter(None, base_name.split("_")))
        filename = f"{normalized_name}.json"
        filepath = run_dir / filename

        # ── Common fields (present in most steps) ─────────────────────────────
        data: dict[str, Any] = {
            "file_sequence_number": step_number,
            "step_type": memory_step.__class__.__name__,
            "step_prefix": prefix,
        }

        # ── Type-specific rich content ───────────────────────────────────────
        if isinstance(memory_step, ActionStep):
            data.update(
                {
                    "step_number": memory_step.step_number,
                    "is_final_answer": memory_step.is_final_answer,
                    "code_action": memory_step.code_action,
                    "model_output": memory_step.model_output,
                    "observations": memory_step.observations,
                    "action_output": memory_step.action_output,
                    "error": (
                        str(memory_step.error)
                        if memory_step.error is not None
                        else None
                    ),
                    "duration_seconds": memory_step.timing.duration
                    if memory_step.timing
                    else None,
                }
            )
            # images not saved yet — placeholder
            if save_images and memory_step.observations_images:
                pass  # future implementation

        elif isinstance(memory_step, PlanningStep):
            data.update(
                {
                    "plan": memory_step.plan,
                    "duration_seconds": memory_step.timing.duration
                    if memory_step.timing
                    else None,
                }
            )

        elif isinstance(memory_step, TaskStep):
            data["task"] = memory_step.task
            # task_images not saved yet

        elif isinstance(memory_step, FinalAnswerStep):
            data["output"] = memory_step.output

        elif isinstance(memory_step, SystemPromptStep):
            data["system_prompt"] = memory_step.system_prompt

        # ── Save ──────────────────────────────────────────────────────────────
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,  # fallback for non-serializable objects
            )

        # Optional feedback
        if agent and hasattr(agent, "verbosity_level") and agent.verbosity_level >= 1:
            print(
                f"[save_step_state] Saved {memory_step.__class__.__name__} → {filepath}"
            )

    return callback
