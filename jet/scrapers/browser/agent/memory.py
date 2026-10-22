"""
Agent Memory Module
Handles short-term working memory accumulation for multi-step browser tasks.
"""

from typing import Any, Dict, List

from jet.scrapers.browser.agent.tools import AgentDecision


class AgentMemory:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.found_items: List[
            str
        ] = []  # Specific list for things like job titles/links
        self.current_url_history: List[str] = []

    def add_step(self, step_index: int, decision: AgentDecision, result: str, url: str):
        """Record a step in memory."""
        self.steps.append(
            {
                "step": step_index,
                "tool": decision.tool,
                "thought": decision.thought,
                "result_summary": result[:200] if result else "",  # Keep summary short
                "url": url,
            }
        )

        if (
            url
            and url != "Unknown"
            and (not self.current_url_history or self.current_url_history[-1] != url)
        ):
            self.current_url_history.append(url)

    def add_found_item(self, item: str):
        """Add specific extracted items (e.g., job titles) to persistent memory."""
        if item and item not in self.found_items:
            self.found_items.append(item)

    def get_context_string(self, max_recent_steps: int = 5) -> str:
        """Generate a text representation of memory for RAG injection."""
        if not self.steps:
            return ""

        context_parts = ["# AGENT MEMORY & HISTORY"]

        # 1. Recent Steps Summary
        recent_steps = self.steps[-max_recent_steps:]
        if recent_steps:
            context_parts.append("## Recent Actions")
            for s in recent_steps:
                context_parts.append(
                    f"- Step {s['step']} [{s['tool']}]: {s['thought']}. "
                    f"Result: {s['result_summary']}"
                )

        # 2. Accumulated Findings
        if self.found_items:
            context_parts.append("## Accumulated Findings")
            # Show last 10 items to avoid token bloat
            items_display = self.found_items[-10:]
            context_parts.append(", ".join(items_display))

        # 3. Navigation History
        if self.current_url_history:
            context_parts.append("## Visited URLs")
            context_parts.append(" -> ".join(self.current_url_history[-3:]))

        return "\n".join(context_parts)

    def reset(self):
        """Clear memory for a new session."""
        self.steps.clear()
        self.found_items.clear()
        self.current_url_history.clear()
