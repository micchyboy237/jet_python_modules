# web_researcher/agents/sub_agents.py
"""
Specialized sub-agents for the web researcher system.
"""

import logging

from models.llama_cpp_model import LlamaCppModel
from smolagents import ToolCallingAgent, WebSearchTool
from tools.summarize import summarize_text
from tools.visit_webpage import visit_webpage

logger = logging.getLogger(__name__)


def create_search_agent(
    model: LlamaCppModel,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
) -> ToolCallingAgent:
    """
    Create a search agent that finds relevant URLs.

    This agent has a single responsibility: search for information
    and return URLs. It doesn't read content, keeping its context small.
    """
    logger.info("Creating search agent")

    return ToolCallingAgent(
        tools=[WebSearchTool()],
        model=model,
        max_steps=max_steps,
        name="search_agent",
        description="""Searches the web for information and returns relevant URLs.
        Use this when you need to find sources on a topic.
        Returns a list of URLs with brief descriptions.""",
        verbosity_level=1,
        # Limit context per step
        max_tokens_per_step=max_tokens_per_step,
        # Simple prompt to keep tokens low
        instructions="""
        You are a search specialist. Your only job is to find relevant web pages.
        1. Perform web searches for the given query
        2. Return a list of URLs that are likely to contain the answer
        3. Keep each result description brief (1-2 sentences)
        4. Return only the URLs, not content from the pages
        """,
    )


def create_extract_agent(
    model: LlamaCppModel,
    max_steps: int = 3,
    max_tokens_per_step: int = 512,
) -> ToolCallingAgent:
    """
    Create an extraction agent that reads and extracts information from webpages.

    This agent visits pages and extracts relevant snippets.
    """
    logger.info("Creating extract agent")

    return ToolCallingAgent(
        tools=[visit_webpage],
        model=model,
        max_steps=max_steps,
        name="extract_agent",
        description="""Visits webpages and extracts specific information.
        Use this to read content from URLs found by the search agent.
        Returns extracted text snippets relevant to the query.""",
        verbosity_level=1,
        max_tokens_per_step=max_tokens_per_step,
        instructions="""
        You are an information extraction specialist.
        1. Visit the provided URL
        2. Extract only the information relevant to the query
        3. Keep the extracted text concise
        4. Do not summarize, just extract the relevant parts
        """,
    )


def create_synthesize_agent(
    model: LlamaCppModel,
    max_steps: int = 2,
    max_tokens_per_step: int = 512,
) -> ToolCallingAgent:
    """
    Create a synthesis agent that compiles answers from extracted information.

    This agent combines information from multiple sources into a final answer.
    """
    logger.info("Creating synthesize agent")

    return ToolCallingAgent(
        tools=[summarize_text],
        model=model,
        max_steps=max_steps,
        name="synthesize_agent",
        description="""Synthesizes information from multiple sources into a coherent answer.
        Use this to combine extracted information and produce a final response.
        Returns a concise, well-structured answer.""",
        verbosity_level=1,
        max_tokens_per_step=max_tokens_per_step,
        instructions="""
        You are a synthesis specialist.
        1. Combine information from all provided sources
        2. Identify key facts and insights
        3. Create a coherent, well-structured answer
        4. Keep the answer concise but comprehensive
        5. Cite sources where appropriate
        """,
    )
