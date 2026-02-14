import hashlib
import logging
from typing import Any

from jet.libs.smolagents.utils.model_utils import create_local_model
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from selectolax.parser import HTMLParser
from smolagents import CodeAgent, ToolCallingAgent

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("HTMLMultiAgentSummarizer")


# ============================================================
# DOM TOOL LAYER
# ============================================================


class HTMLDOMTools:
    def __init__(self, html: str):
        self.parser = HTMLParser(html)
        self.root = self.parser.root

    def iter_semantic_sections(self) -> list[Any]:
        """
        Returns high-level semantic sections for sliding traversal.
        Fallback to body children if no semantic tags exist.
        """
        semantic_tags = [
            "section",
            "article",
            "nav",
            "main",
            "aside",
            "header",
            "footer",
        ]

        sections = []
        for tag in semantic_tags:
            sections.extend(self.parser.css(tag))

        if not sections:
            body = self.parser.css_first("body")
            if body:
                sections = list(body.iter_children())

        return sections

    def serialize_node(self, node) -> dict[str, Any]:
        """
        Safe recursive serializer using selectolax API.
        """

        def walk(n):
            if n is None:
                return {}

            # If the node has no tag (text or whitespace), use text_content
            tag = getattr(n, "tag", None)

            # Extract text
            text_value = ""
            try:
                text_value = n.text(strip=True)
            except Exception:
                pass

            # If no actual tag (text node), return text only
            if tag is None:
                return {"text": text_value}

            # Serialize node
            result: dict[str, Any] = {
                "tag": tag,
                "attributes": dict(n.attributes) if hasattr(n, "attributes") else {},
            }

            if text_value:
                result["text"] = text_value

            # Traverse children using selectolax ".iter()" API
            children_data = []
            for child in n.iter():
                # Skip the node itself
                if child is n:
                    continue
                # Only include element or non-empty text nodes
                child_text = child.text(strip=True)
                if getattr(child, "tag", None) or child_text:
                    children_data.append(walk(child))

            if children_data:
                result["children"] = children_data

            return result

        return walk(node)


# ============================================================
# SCALABLE MULTI-AGENT HTML SUMMARIZER
# ============================================================


class ScalableHTMLMultiAgentSummarizer:
    """
    Sliding-window DOM summarizer with:
    - Subtree traversal
    - Iterative merge
    - Caching
    - Semantic grouping
    """

    def __init__(
        self,
        model_id: str | None = None,
        enable_cache: bool = True,
    ):
        logger.info("[bold cyan]Initializing ScalableHTMLMultiAgentSummarizer[/]")

        self.enable_cache = enable_cache
        self.cache: dict[str, str] = {}

        model = (
            create_local_model(agent_name="html_manager")
            if not model_id
            else create_local_model(agent_name="html_manager", model_id=model_id)
        )

        self.subtree_agent = ToolCallingAgent(
            tools=[],
            model=model,
            name="SubtreeAnalyzer",
            description=(
                "Analyzes a single HTML subtree and produces a complete structured "
                "summary preserving hierarchy and metadata."
            ),
        )

        self.merge_agent = ToolCallingAgent(
            tools=[],
            model=model,
            name="StructureMerger",
            description=(
                "Merges multiple structured subtree summaries into a unified "
                "hierarchical representation without losing information."
            ),
        )

        self.manager = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[self.subtree_agent, self.merge_agent],
        )

    # ----------------------------------------------------------

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    # ----------------------------------------------------------

    def summarize(self, html: str) -> str:
        logger.info("[bold magenta]Parsing HTML...[/]")

        dom_tools = HTMLDOMTools(html)
        sections = dom_tools.iter_semantic_sections()

        logger.info(
            f"[bold cyan]Sliding traversal across {len(sections)} semantic sections[/]"
        )

        merged_structure = ""
        subtree_summaries = []

        for idx, node in enumerate(sections):
            structured = dom_tools.serialize_node(node)
            serialized_str = str(structured)

            content_hash = self._hash_content(serialized_str)

            if self.enable_cache and content_hash in self.cache:
                logger.info(f"[dim]Using cached subtree {idx + 1}[/dim]")
                subtree_summary = self.cache[content_hash]
            else:
                logger.info(f"[cyan]Analyzing subtree {idx + 1}/{len(sections)}[/cyan]")

                prompt = f"""
You are given a structured HTML subtree.

Requirements:
- Preserve ALL structure and metadata.
- Maintain hierarchy.
- Do NOT discard any factual information.
- Produce a structured hierarchical summary.

Subtree:
{structured}
"""

                subtree_summary = self.subtree_agent.run(prompt)

                if self.enable_cache:
                    self.cache[content_hash] = subtree_summary

            subtree_summaries.append(subtree_summary)

        # ------------------------------------------------------
        # Iterative Merge
        # ------------------------------------------------------

        logger.info("[bold magenta]Merging subtree summaries iteratively...[/]")

        merged_structure = subtree_summaries[0]

        for next_summary in subtree_summaries[1:]:
            merge_prompt = f"""
Merge the following two structured summaries into a single unified
hierarchical representation.

Requirements:
- Do NOT remove any factual information.
- Preserve hierarchy.
- Combine overlapping sections logically.

Summary A:
{merged_structure}

Summary B:
{next_summary}
"""
            merged_structure = self.merge_agent.run(merge_prompt)

        # ------------------------------------------------------
        # Final Formatting
        # ------------------------------------------------------

        final_prompt = f"""
Convert the following unified structured representation into a
clean, readable hierarchical summary.

Requirements:
- Preserve all information.
- Keep structure clear.
- Maintain links, metadata, tables.

Structured Representation:
{merged_structure}
"""

        final_summary = self.manager.run(final_prompt)

        console.print(
            Panel(
                final_summary.strip(),
                title="[bold green]Scalable HTML Summary[/bold green]",
                border_style="green",
            )
        )

        return final_summary
