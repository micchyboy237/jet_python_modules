from jet.libs.smolagents.components.long_term_content_memory import (
    LongTermContentMemory,
)
from smolagents import tool


@tool
def add_to_research_knowledge(
    url: str,
    content: str,
) -> None:
    """
    Add a webpage or text snippet to the research knowledge base.

    Args:
        url: Source URL of the content (used for citation and avoiding duplicates).
        content: The extracted or cleaned main text from the page or source.

    Returns:
        None
    """
    memory: LongTermContentMemory | None = getattr(
        add_to_research_knowledge, "_memory", None
    )

    if memory is None:
        return "Error: Long-term memory not attached."

    if not content or len(content.strip()) < 40:
        return "Content too short — nothing was added."

    chunk_id = memory.add(
        content=content.strip(),
        source_url=url,
        step=getattr(add_to_research_knowledge, "_current_step", 0),
    )

    return f"Added to long-term knowledge (id: {chunk_id})"


def attach_memory_to_accumulate_tool(
    tool_instance,
    memory: LongTermContentMemory,
) -> None:
    """Helper to inject memory instance"""
    tool_instance._memory = memory


def set_current_step(tool_instance, step: int) -> None:
    """Update current research step number for better traceability"""
    tool_instance._current_step = step
