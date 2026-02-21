import json
from pathlib import Path

from jet.transformers.object import make_serializable
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from openai.types import CompletionUsage
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

DEFAULT_LOGS_DIR = (
    Path(get_entry_file_dir())
    / "generated"
    / Path(get_entry_file_name()).stem
    / "llm_calls"
)


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


def _get_llm_call_subdir(
    logs_dir: Path,
    agent_name: str | None = None,
) -> Path:
    call_number = get_next_call_number(logs_dir)
    # prefix = "generate_stream" if is_stream else "generate"
    prefix = f"{call_number:04d}"

    cleaned = "default"
    if agent_name:
        cleaned = (
            str(agent_name)
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .lower()
        )
    elif hasattr(logs_dir, "name") and "_" in logs_dir.name:
        # fallback: try to reuse parent agent folder name if possible
        cleaned = logs_dir.name.split("_")[-1]

    subdir_name = f"{prefix}_{cleaned}"
    target_dir = logs_dir / subdir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_request_llm_call(
    messages: list[dict],
    metadata: dict | None = None,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    agent_name: str | None = None,
) -> None:
    target_dir = _get_llm_call_subdir(
        logs_dir,
        agent_name,
    )
    (target_dir / "messages.json").write_text(
        json.dumps(messages, indent=2, ensure_ascii=False)
    )
    if metadata:
        (target_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )


def save_response_llm_call(
    response: str,
    usage: CompletionUsage | None = None,
    choices: list[Choice] | None = None,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    agent_name: str | None = None,
) -> None:
    target_dir = _get_llm_call_subdir(
        logs_dir,
        agent_name,
    )

    # Save response in response.md as plain text (UTF-8)
    (target_dir / "response.md").write_text(response, encoding="utf-8")

    if usage:
        (target_dir / "usage.json").write_text(
            json.dumps(make_serializable(usage), indent=2, ensure_ascii=False)
        )

    if choices:
        (target_dir / "stream_choices.json").write_text(
            json.dumps(make_serializable(choices), indent=2, ensure_ascii=False)
        )

        stream_deltas: list[ChoiceDelta] = [c.delta for c in choices]

        # Extract tool_calls if present in any of the stream_deltas
        all_tool_calls = [
            delta.tool_calls
            for delta in stream_deltas
            if getattr(delta, "tool_calls", None)
        ]
        # Flatten and filter out None values
        flat_tool_calls = []
        for tc_list in all_tool_calls:
            if tc_list:
                flat_tool_calls.extend(tc_list)

        if flat_tool_calls:
            # Accumulate the function.arguments from each tool_call into a single string
            accumulated_args = "".join(
                tc.function.arguments
                if hasattr(tc, "function")
                and tc.function
                and hasattr(tc.function, "arguments")
                and tc.function.arguments
                else ""
                for tc in flat_tool_calls
            )
            (target_dir / "tool_calls.json").write_text(
                json.dumps(
                    make_serializable(accumulated_args), indent=2, ensure_ascii=False
                )
            )
        else:
            text = "".join(
                delta.content
                for delta in stream_deltas
                if getattr(delta, "content", None)
            )
            (target_dir / "response.json").write_text(
                json.dumps(make_serializable(text), indent=2, ensure_ascii=False)
            )
