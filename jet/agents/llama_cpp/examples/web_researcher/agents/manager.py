# web_researcher/agents/manager.py
"""
Manager agent that orchestrates the web research workflow.
"""

import logging
from typing import Any, Dict, Optional

from agents.sub_agents import (
    create_extract_agent,
    create_search_agent,
    create_synthesize_agent,
)
from models.llama_cpp_model import LlamaCppModel
from smolagents import CodeAgent

logger = logging.getLogger(__name__)


class WebResearcherAgent:
    """
    Main web researcher agent that orchestrates specialized sub-agents.

    Architecture:
        Manager (CodeAgent) -> Search Agent -> Extract Agent -> Synthesize Agent

    Each sub-agent has a single responsibility, keeping context small.
    """

    def __init__(
        self,
        model: LlamaCppModel,
        search_agent: Any,
        extract_agent: Any,
        synthesize_agent: Any,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize the web researcher agent.

        Args:
            model: LlamaCppModel instance
            search_agent: Search sub-agent
            extract_agent: Extraction sub-agent
            synthesize_agent: Synthesis sub-agent
            max_steps: Maximum steps for the manager
            verbose: Enable verbose logging
        """
        self.model = model
        self.search_agent = search_agent
        self.extract_agent = extract_agent
        self.synthesize_agent = synthesize_agent
        self.verbose = verbose

        # Create manager agent with sub-agents
        self.manager = self._create_manager(max_steps)

        logger.info(
            f"WebResearcherAgent initialized with max_steps={max_steps}, "
            f"model={model.model_id}"
        )

    def _create_manager(self, max_steps: int) -> CodeAgent:
        """Create the manager CodeAgent with managed sub-agents."""
        return CodeAgent(
            tools=[],  # No direct tools, uses sub-agents
            model=self.model,
            managed_agents=[
                self.search_agent,
                self.extract_agent,
                self.synthesize_agent,
            ],
            max_steps=max_steps,
            verbosity_level=2 if self.verbose else 1,
            # Custom instructions for efficient orchestration
            instructions="""
            You are a research manager coordinating a team of specialists.

            Team Members:
            1. search_agent: Finds relevant web pages for a topic
            2. extract_agent: Reads web pages and extracts specific information
            3. synthesize_agent: Combines extracted information into a final answer

            Workflow guidelines:
            1. First, use search_agent to find relevant URLs for the query
            2. For each relevant URL, use extract_agent to get specific information
            3. Finally, use synthesize_agent to combine all information

            Efficiency tips:
            - Be specific in your queries to reduce token usage
            - Extract only what you need, not entire pages
            - If you have enough information, skip unnecessary extractions
            - Keep each step concise and focused

            Always provide the final answer using the final_answer() function.
            """,
            additional_authorized_imports=[],  # No imports needed for orchestration
        )

    def run(self, query: str, **kwargs) -> str:
        """
        Run the web research workflow.

        Args:
            query: The research query
            **kwargs: Additional arguments for the manager

        Returns:
            The final answer
        """
        logger.info(f"Running web researcher with query: {query[:100]}...")

        # Reset token stats for a clean run
        self.model.reset_token_stats()

        # Run the manager agent
        result = self.manager.run(query, **kwargs)

        # Log token usage
        stats = self.model.get_token_stats()
        logger.info(
            f"Research completed. Token stats: "
            f"Input={stats['total_input_tokens']}, "
            f"Output={stats['total_output_tokens']}, "
            f"Total={stats['total_tokens']}"
        )

        return result

    def get_token_stats(self) -> Dict[str, int]:
        """Get token usage statistics from the model."""
        return self.model.get_token_stats()

    def reset(self) -> None:
        """Reset the agent's memory and token stats."""
        self.model.reset_token_stats()
        # Reset sub-agents if they have reset methods
        for agent in [self.search_agent, self.extract_agent, self.synthesize_agent]:
            if hasattr(agent, "reset"):
                agent.reset()


def create_web_researcher(
    llm_model: Optional[str] = None,
    base_url: Optional[str] = None,
    max_context_length: int = 4096,
    temperature: float = 0.7,
    max_tokens: int = 512,
    search_agent_steps: int = 5,
    extract_agent_steps: int = 3,
    synthesize_agent_steps: int = 2,
    manager_steps: int = 10,
    verbose: bool = False,
) -> WebResearcherAgent:
    """
    Factory function to create a fully configured web researcher.

    Args:
        llm_model: Model ID for llama.cpp
        base_url: llama.cpp server URL
        max_context_length: Maximum context window size
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation
        search_agent_steps: Max steps for search agent
        extract_agent_steps: Max steps for extract agent
        synthesize_agent_steps: Max steps for synthesize agent
        manager_steps: Max steps for manager agent
        verbose: Enable verbose logging

    Returns:
        Configured WebResearcherAgent
    """
    # Create the shared model
    model = LlamaCppModel(
        model_id=llm_model,
        base_url=base_url,
        max_context_length=max_context_length,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Create sub-agents
    search_agent = create_search_agent(
        model,
        max_steps=search_agent_steps,
        max_tokens_per_step=max_tokens // 2,
    )

    extract_agent = create_extract_agent(
        model,
        max_steps=extract_agent_steps,
        max_tokens_per_step=max_tokens,
    )

    synthesize_agent = create_synthesize_agent(
        model,
        max_steps=synthesize_agent_steps,
        max_tokens_per_step=max_tokens,
    )

    # Create the manager
    return WebResearcherAgent(
        model=model,
        search_agent=search_agent,
        extract_agent=extract_agent,
        synthesize_agent=synthesize_agent,
        max_steps=manager_steps,
        verbose=verbose,
    )
