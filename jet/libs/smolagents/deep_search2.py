# deep_research_system.py

import re
from dataclasses import dataclass

import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from smolagents import (
    CodeAgent,
    FinalAnswerTool,
    InferenceClientModel,
    ToolCallingAgent,
    WebSearchTool,
    tool,
)

console = Console()

# ────────────────────────────────────────────────
#  Memory for Persistent Fact Accumulation
# ────────────────────────────────────────────────


@dataclass
class ResearchMemory:
    """Simple shared memory for accumulating key facts across research rounds"""

    facts: list[dict[str, str]] = None

    def __post_init__(self):
        if self.facts is None:
            self.facts = []

    def add(self, content: str, source: str, round_id: int | None = None) -> str:
        """Add extracted facts / key information"""
        if not content.strip():
            return "No new facts to add."
        entry = {
            "content": content.strip(),
            "source": source,
            "round": round_id if round_id is not None else len(self.facts) + 1,
        }
        self.facts.append(entry)
        return f"Added {len(content)} characters from {source}"

    def get_summary(self, max_chars: int = 4000) -> str:
        if not self.facts:
            return "No facts accumulated yet."
        lines = []
        for i, f in enumerate(self.facts, 1):
            lines.append(f"Round {f['round']} | {f['source']}")
            lines.append(
                f"{f['content'][:500]}{'...' if len(f['content']) > 500 else ''}"
            )
            lines.append("─" * 60)
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text

    def get_raw(self) -> str:
        if not self.facts:
            return "Memory is empty."
        return "\n\n".join(
            f"[{f['round']}] {f['source']}:\n{f['content']}" for f in self.facts
        )


# ────────────────────────────────────────────────
#  Custom Tools
# ────────────────────────────────────────────────


@tool
def visit_webpage(url: str, max_length: int = 18000) -> dict[str, any]:
    """Fetches a webpage and returns both full markdown and semantically split chunks.
    Chunks are split on headings and large paragraph breaks for better relevance filtering."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        md = markdownify(
            response.text, strip=["script", "style", "noscript", "header", "footer"]
        )
        md = re.sub(r"\n{3,}", "\n\n", md.strip())

        # Naive semantic chunking: split on headings + double newlines
        chunks = []
        current = []
        for line in md.split("\n"):
            if line.strip().startswith("#") and current:
                chunks.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("\n".join(current).strip())

        chunks = [c for c in chunks if len(c.strip()) > 80]

        full_text = md
        if len(full_text) > max_length:
            full_text = full_text[:max_length] + "\n\n... (truncated)"

        return {
            "full_text": full_text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "total_length": len(md),
        }
    except RequestException as e:
        return {"error": f"Error fetching page: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@tool
def select_relevant_content(
    content: dict[str, any], query: str, max_chunks: int = 6
) -> str:
    """Selects and concatenates the most relevant chunks from a webpage based on query."""
    if "error" in content:
        return content["error"]

    chunks = content.get("chunks", [])
    if not chunks:
        return content.get("full_text", "No content available.")

    keywords = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for kw in keywords if kw in chunk_lower)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [chunk for _, chunk in scored[:max_chunks]]

    if not selected:
        selected = chunks[:max_chunks]

    return (
        "\n\n…\n\n".join(selected)
        + f"\n\n(Showing {len(selected)} / {len(chunks)} chunks)"
    )


@tool
def add_key_facts(memory, facts: str, source: str) -> str:
    """Accumulate important facts / conclusions into long-term research memory.
    Use this when you find reliable, novel or critical information worth remembering across rounds.
    facts: concise bullet points or paragraphs you want to keep
    source: where it came from (url, search query, previous agent name, etc)"""
    return memory.add(facts, source)


# ────────────────────────────────────────────────
#  Agents
# ────────────────────────────────────────────────


def create_research_agent(model, memory: ResearchMemory):
    return ToolCallingAgent(
        tools=[
            WebSearchTool(),
            visit_webpage,
            select_relevant_content,
            add_key_facts.partial(memory=memory),  # bind memory instance
            FinalAnswerTool(),
        ],
        model=model,
        max_steps=12,
        name="deep_research",
        description="Call this when you need to gather detailed web information. Performs deep web research. Give it a very clear, focused research question or extraction goal. Use add_key_facts tool when you discover valuable persistent facts.",
    )


def create_critic_agent(model, memory: ResearchMemory):
    return ToolCallingAgent(
        tools=[
            add_key_facts.partial(memory=memory),
            FinalAnswerTool(),
        ],
        model=model,
        max_steps=5,
        name="critic",
        description="Call this to evaluate if current evidence answers the query well. You are a strict research quality judge. ... You may also use add_key_facts if the current round produced valuable persistent insights.",
    )


def create_deep_research_system(
    model_id: str = "Qwen/Qwen3-Next-80B-A3B-Thinking",
    max_research_loops: int = 5,
) -> CodeAgent:
    model = InferenceClientModel(model_id=model_id)

    # Shared memory across all rounds
    research_memory = ResearchMemory()

    # Sub-agents
    research_agent = create_research_agent(model, research_memory)
    critic_agent = create_critic_agent(model, research_memory)

    # Manager — the brain that loops
    manager = CodeAgent(
        tools=[
            add_key_facts.partial(memory=research_memory),
            # Optional: could add get_memory_summary tool here too
        ],
        model=model,
        managed_agents=[research_agent, critic_agent],
        max_steps=30,
        additional_authorized_imports=["time"],
        name="deep_research_manager",
    )

    # Rich system prompt override for better loop control
    manager.system_prompt += (
        "\n\nYou are a deep research coordinator. Follow this strict protocol:\n"
        "1. Start by sending a focused question to deep_research\n"
        "2. After results arrive, consider using add_key_facts if valuable persistent information was found\n"
        "3. Ask critic to judge sufficiency\n"
        "4. If critic says 'good' → give final answer (you can include memory summary if helpful)\n"
        "5. If 'needs_more' or 'rerun_different_angle' → plan next angle and call deep_research again\n"
        "6. If 'impossible' → explain why and stop\n"
        f"Maximum {max_research_loops} research rounds. Be efficient.\n"
        "Always try to give the most accurate, sourced answer possible."
    )

    return manager


# ────────────────────────────────────────────────
#  Usage Example
# ────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel.fit("Deep Web Research System", style="bold green"))

    researcher = create_deep_research_system(
        model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
        max_research_loops=6,
    )

    query = (
        "What are the current most promising technical approaches "
        "to drastically reduce inference cost of large language models in 2026?"
    )

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    with track(sequence=range(1), description="Researching...", total=None):
        answer = researcher.run(query)

    console.print("\n[bold green]Final Answer[/bold green]")
    console.print(Panel(answer, border_style="green", expand=True))
