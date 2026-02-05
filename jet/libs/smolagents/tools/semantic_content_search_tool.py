from jet.libs.smolagents.components.long_term_content_memory import (
    LongTermContentMemory,
)
from smolagents import tool


@tool
def search_relevant_content(
    query: str,
    top_k: int = 6,
    min_relevance: float = 0.22,
) -> str:
    """
    Search semantically similar content across all previously accumulated research knowledge.
    Use this tool when:
    - you need to recall facts from earlier page visits
    - you want to cross-check or compare information
    - you have a specific question and suspect the answer is already in visited content

    Args:
        query: Precise description of what you are looking for
        top_k: Number of best matching chunks to return (3-10 recommended)
        min_relevance: Minimum hybrid score to include (0.0-1.0)

    Returns:
        Markdown formatted list of relevant excerpts with sources
    """
    memory: LongTermContentMemory | None = getattr(
        search_relevant_content, "_memory", None
    )

    if memory is None:
        return "Error: Long-term memory not attached to this tool."

    results = memory.search(
        query=query.strip(),
        top_k=top_k,
        min_score=min_relevance,
    )

    if not results:
        return f"No sufficiently relevant content found for: **{query}**"

    lines = [
        f"**Relevant accumulated knowledge** (query: {query})",
        f"Found {len(results)} matching pieces\n",
    ]

    for i, r in enumerate(results, 1):
        preview = r["content"][:320].strip()
        if len(r["content"]) > 320:
            preview += "..."
        source = r["source"] if r["source"] else "[no source]"
        lines.append(
            f"{i}. Score: {r['score']:.3f}  •  Step {r['step']}  •  {source}\n"
            f"{preview}\n"
        )

    return "\n".join(lines)


def attach_memory_to_search_tool(
    tool_instance,
    memory: LongTermContentMemory,
) -> None:
    """Helper to inject memory instance into the tool at runtime"""
    tool_instance._memory = memory
