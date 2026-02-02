import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from smolagents import ActionStep, FinalAnswerStep, MemoryStep, PlanningStep


def get_next_step_number(base_dir: Path, prefix: str = "") -> int:
    """
    Find the next available number for files starting with {prefix} in base_dir.
    Similar logic to DebugSaver.get_next_call_number but per-prefix aware.
    """
    if not base_dir.exists():
        return 1
    numbers = []
    target_start = f"{prefix}_" if prefix else ""
    for f in base_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if name.startswith(target_start):
            # Try to extract number after prefix_
            rest = name[len(target_start) :]
            if "_" in rest:
                num_part = rest.split("_", 1)[0]
            else:
                num_part = rest
            try:
                num = int(num_part)
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
      {base_dir}/{cleaned_agent_name}/action_0003.json
      {base_dir}/{cleaned_agent_name}/plan_0001.json
      {base_dir}/{cleaned_agent_name}/final_0007.json
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

        # Get next sequential number for this prefix and directory
        next_num = get_next_step_number(run_dir, prefix=prefix)

        # Build base filename
        base_name = f"{prefix}_{next_num:04d}_{cleaned_name}"

        # Normalize: replace multiple underscores with single
        normalized_name = "_".join(filter(None, base_name.split("_")))
        filename = f"{normalized_name}.json"

        filepath = run_dir / filename

        data: dict[str, Any] = {
            "step_number": next_num,  # now the file-based sequence number
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
