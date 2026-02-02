import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from smolagents import ActionStep, FinalAnswerStep, MemoryStep, PlanningStep


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
                pass
    return max(numbers, default=0) + 1 if numbers else 1


def save_step_state(
    agent_name: str | None = None,
    base_dir: str | None = None,
    save_images: bool = False,  # optional: can be enabled later
) -> Callable[[MemoryStep, Any], None]:
    """
    Factory that creates a step callback which saves step information to disk.

    The returned callback uses the provided agent_name (or falls back to "unnamed_agent").
    Intended usage:

        callback = save_step_state(agent_name="research_agent_2025")
        agent = CodeAgent(..., step_callbacks=[callback])

    Or inline:
        agent = CodeAgent(
            ...,
            step_callbacks=[save_step_state(agent_name="math_explorer")],
        )

    Files are saved as:
      {base_dir}/{cleaned_agent_name}/0001_plan_my_agent.json
      {base_dir}/{cleaned_agent_name}/0003_action_my_agent.json
      {base_dir}/{cleaned_agent_name}/0007_final_my_agent.json
      ...
    """
    # Clean and normalize the agent name once (at creation time)
    if agent_name:
        cleaned_name = (
            str(agent_name).strip().replace(" ", "_").replace("-", "_").lower()
        )
    else:
        cleaned_name = "agent"

    _caller_base_dir = (
        Path(get_entry_file_dir()) / "generated" / Path(get_entry_file_name()).stem
    ).resolve()
    _caller_base_dir = _caller_base_dir / "agent_runs"
    run_dir = Path(base_dir).resolve() if base_dir else _caller_base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    def callback(memory_step: MemoryStep, agent: Any = None) -> None:
        nonlocal run_dir

        # Determine prefix based on step type
        if isinstance(memory_step, ActionStep):
            prefix = "action"
        elif isinstance(memory_step, PlanningStep):
            prefix = "plan"
        elif isinstance(memory_step, FinalAnswerStep):
            prefix = "final"
        else:
            # Fallback for other / unknown step types
            step_type_name = memory_step.__class__.__name__.lower().replace("step", "")
            prefix = step_type_name if step_type_name else "step"

        # Get next global step number for this directory
        next_num = get_next_run_step_number(run_dir)

        # Build base filename — number comes first for natural sorting
        base_name = f"{next_num:04d}_{prefix}_{cleaned_name}"

        # Normalize: collapse consecutive underscores
        normalized_name = "_".join(filter(None, base_name.split("_")))
        filename = f"{normalized_name}.json"

        filepath = run_dir / filename

        data: dict[str, Any] = {
            "file_sequence_number": next_num,  # global per-run sequence
            "step_type_prefix": prefix,
            "step_type": memory_step.__class__.__name__,
            "timestamp": getattr(memory_step, "timestamp", None),
            "thought": getattr(memory_step, "thought", None),
            "llm_output": getattr(memory_step, "llm_output", None),
            "error": getattr(memory_step, "error", None),
        }

        if isinstance(memory_step, ActionStep):
            data.update(
                {
                    "code": getattr(memory_step, "code", None),
                    "tool_name": getattr(memory_step, "tool_name", None),
                    "tool_arguments": getattr(memory_step, "tool_arguments", None),
                    "observations": getattr(memory_step, "observations", None),
                }
            )

        elif isinstance(memory_step, PlanningStep):
            data["plan"] = getattr(memory_step, "plan", None)

        elif isinstance(memory_step, FinalAnswerStep):
            data["final_answer"] = getattr(memory_step, "final_answer", None)

        # Optional: base64 images (disabled by default)
        # if save_images and hasattr(memory_step, "observations_images") and memory_step.observations_images:
        #     import base64
        #     from io import BytesIO
        #     from PIL import Image
        #     b64_images = []
        #     for img in memory_step.observations_images:
        #         buffered = BytesIO()
        #         img.save(buffered, format="PNG")
        #         b64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        #     data["observations_images_base64"] = b64_images

        # Write file
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        # Optional feedback
        if agent and hasattr(agent, "verbosity_level") and agent.verbosity_level >= 1:
            print(
                f"[save_step_state] Saved {memory_step.__class__.__name__} → {filepath}"
            )

    return callback
