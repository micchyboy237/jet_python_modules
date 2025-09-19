from swarms import Agent
from typing import Callable, List, Dict, Any
from datetime import datetime
import ollama
import os


# Example 1: Basic Agent with LLM and System Prompt
def basic_agent_example() -> str:
    """Demonstrates basic Agent setup with ollama/llama3.2 and system prompt."""
    agent = Agent(
        model_name="ollama/llama3.2",
        system_prompt="You are a helpful financial analyst.",
        max_loops=1,
        verbose=True
    )
    response = agent.run("Generate a brief financial report summary.")
    return response


# Example 2: Agent with Tools and Tool Schema
def agent_with_tools_example() -> str:
    """Demonstrates Agent with custom tools and tool schema."""
    def sample_tool(query: str) -> str:
        return f"Tool processed: {query}"

    agent = Agent(
        model_name="ollama/llama3.2",
        tools=[sample_tool],
        tool_schema="json",
        execute_tool=True,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Use the tool to process 'financial data'.")
    return response


# Example 3: Interactive Agent with Streaming
def interactive_streaming_example() -> str:
    """Demonstrates interactive mode with streaming output."""
    agent = Agent(
        model_name="ollama/llama3.2",
        interactive=True,
        streaming_on=True,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Tell me a short story.")
    return response


# Example 4: Agent with Long-Term Memory and Document Ingestion
def memory_and_docs_example() -> str:
    """Demonstrates long-term memory and document ingestion."""
    agent = Agent(
        model_name="ollama/llama3.2",
        long_term_memory=None,  # Placeholder, assumes BaseVectorDatabase
        docs=["sample_doc.txt"],
        docs_folder="./docs",
        context_length=2048,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Summarize the content of sample_doc.txt.")
    return response


# Example 5: Agent with Dynamic Temperature and Sentiment Analysis
def dynamic_temperature_example() -> str:
    """Demonstrates dynamic temperature and sentiment analysis."""
    def sentiment_analyzer(text: str) -> float:
        return 0.8 if "positive" in text.lower() else 0.2

    agent = Agent(
        model_name="ollama/llama3.2",
        dynamic_temperature_enabled=True,
        sentiment_analyzer=sentiment_analyzer,
        sentiment_threshold=0.5,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Generate a positive financial outlook.")
    return response


# Example 6: Agent with Artifacts and Custom Output
def artifacts_example() -> str:
    """Demonstrates artifact generation and custom output type."""
    agent = Agent(
        model_name="ollama/llama3.2",
        artifacts_on=True,
        artifacts_output_path="./artifacts",
        artifacts_file_extension=".md",
        output_type="markdown",
        max_loops=1,
        verbose=True
    )
    response = agent.run("Generate a markdown report.")
    return response


# Example 7: Agent with Chain of Thoughts and Stopping Condition
def chain_of_thoughts_example() -> str:
    """Demonstrates chain of thoughts with stopping condition."""
    def stopping_condition(response: str) -> bool:
        return "complete" in response.lower()

    agent = Agent(
        model_name="ollama/llama3.2",
        chain_of_thoughts=True,
        stopping_condition=stopping_condition,
        max_loops=3,
        verbose=True
    )
    response = agent.run(
        "Explain your reasoning for a financial forecast. End with 'complete'.")
    return response
