"""
DOM Agent Logic
Encapsulates the logic for an LLM agent to analyze web page structure
using pattern recognition and density mapping.
"""

import json
from typing import Callable, Dict, Optional

from jet.libs.llama_cpp.usage.chat_stream import run_chat_stream_async
from jet.logger import logger
from jet.scrapers.browser.playwright.dom_analyzer import (
    analyze_patterns,
    analyze_structure,
)


class DOMAnalysisAgent:
    """
    An agent that uses LLM reasoning to interpret DOM structures.
    """

    def __init__(self, model: str = "qwen3.5-uncensored:2b"):
        self.model = model
        self.tools: Dict[str, Callable] = {
            "analyze_page_patterns": self._tool_analyze_patterns,
            "analyze_page_density": self._tool_analyze_density,
        }
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_page_patterns",
                    "description": "Identifies repeating UI patterns like lists, grids, or cards on a webpage.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The URL or HTML string to analyze.",
                            },
                            "match_type": {
                                "type": "string",
                                "enum": ["tag", "class", "both"],
                                "description": "How to match children elements.",
                            },
                        },
                        "required": ["source"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_page_density",
                    "description": "Identifies dense areas of the DOM to help locate main content or complex layouts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The URL or HTML string to analyze.",
                            },
                            "min_children": {
                                "type": "integer",
                                "description": "Minimum number of children to consider a node dense.",
                            },
                        },
                        "required": ["source"],
                    },
                },
            },
        ]

    async def _tool_analyze_patterns(
        self, source: str, match_type: str = "both"
    ) -> str:
        """Internal tool wrapper for pattern analysis."""
        logger.info(f"[Agent Tool] Analyzing patterns for: {source[:50]}...")
        try:
            results = await analyze_patterns(
                source=source, match_type=match_type, headless=True
            )
            # Return a truncated JSON string to save tokens in the context
            return json.dumps(results[:5], indent=2)
        except Exception as e:
            return f"Error analyzing patterns: {str(e)}"

    async def _tool_analyze_density(self, source: str, min_children: int = 5) -> str:
        """Internal tool wrapper for density analysis."""
        logger.info(f"[Agent Tool] Analyzing density for: {source[:50]}...")
        try:
            results = await analyze_structure(
                source=source, min_children=min_children, headless=True
            )
            return json.dumps(results[:5], indent=2)
        except Exception as e:
            return f"Error analyzing density: {str(e)}"

    async def reason_about_page(
        self,
        source: str,
        user_goal: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Main entry point for the agent. The LLM decides which analysis tool to use
        to achieve the user's goal.
        """
        if not system_prompt:
            system_prompt = (
                "You are a Web Structure Analysis Agent. Your goal is to help a scraper "
                "understand how to extract data from a webpage. You have access to tools "
                "that can analyze DOM patterns and density. Use these tools to provide "
                "specific CSS selectors or structural advice."
            )

        initial_prompt = (
            f"I need to achieve this goal: '{user_goal}'\n"
            f"The target source is: '{source}'\n"
            f"Please analyze the page structure and recommend the best extraction strategy."
        )

        result = await run_chat_stream_async(
            prompt_or_messages=initial_prompt,
            model=self.model,
            system_message=system_prompt,
            tools=self.tool_definitions,
            tool_registry=self.tools,
            max_tool_rounds=3,
        )

        return result.content


async def create_dom_agent(model: str = "qwen3.5-uncensored:2b") -> DOMAnalysisAgent:
    """Factory function to create a configured DOM Agent."""
    return DOMAnalysisAgent(model=model)
