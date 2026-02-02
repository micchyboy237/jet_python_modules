# tools/memory_tools.py
from memory.long_term import long_term_memory
from memory.shared_state import shared_state
from smolagents import Tool


def save_fact(content: str, *, step_number: int | None = None) -> str:
    """Save important reusable fact / entity / lesson / preference to long-term memory.

    Args:
        content: The fact text to store
        step_number: Optional step number (passed from callback when available)
    """
    if len(content) < 8:
        return "Fact too short – ignored."
    step_nr = step_number if step_number is not None else 0
    return long_term_memory.add_fact(content, step_nr)


def recall_facts(query: str, n_results: int = 5) -> str:
    """Search long-term memory for relevant past facts."""
    return long_term_memory.search(query, n_results)


def update_shared(key: str, value: str) -> str:
    """Update a value in the shared mutable state."""
    shared_state.set(key, value)
    return "Shared state updated → {key} = {value}"


def read_shared(key: str) -> str:
    """Read current value from shared state."""
    v = shared_state.get(key, "<not set>")
    return "{key}: {v}"


# Tools ready to be passed to CodeAgent
LongTermSaveTool = Tool(
    name="save_important_fact",
    description=(
        "Save concise, high-value facts, entities, goals, lessons or user preferences "
        "to long-term memory for future reuse."
    ),
    function=save_fact,
    parameters={"content": "string"},
)

LongTermRecallTool = Tool(
    name="recall_relevant_facts",
    description="Search long-term memory for facts relevant to the current task.",
    function=recall_facts,
    parameters={"query": "string", "n_results": "integer (optional, default 5)"},
)

SharedStateUpdateTool = Tool(
    name="update_shared_state",
    description="Update a named value in the persistent shared state.",
    function=update_shared,
    parameters={"key": "string", "value": "string"},
)

SharedStateReadTool = Tool(
    name="read_shared_state",
    description="Read a named value from the shared persistent state.",
    function=read_shared,
    parameters={"key": "string"},
)
