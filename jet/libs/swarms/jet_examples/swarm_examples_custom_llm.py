from langchain_ollama import ChatOllama
from swarms import Agent
from typing import Any, List, Dict, Optional


# Custom ChatOllama class with __call__ method
class CustomChatOllama(ChatOllama):
    """Custom ChatOllama class with a __call__ method for Agent integration."""

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """
        Custom generation method for Agent compatibility.

        Args:
            prompt (str): The input prompt for generation.
            **kwargs: Additional arguments like max_tokens, temperature, etc.

        Returns:
            str: Generated text response.
        """
        # Prepare messages for LangChain's ChatOllama
        messages = [{"role": "user", "content": prompt}]

        # Invoke the model using LangChain's invoke method
        response = self.invoke(messages, **kwargs)

        # Extract content from response
        return response.content


# Example: Agent using CustomChatOllama
def custom_ollama_agent_example() -> str:
    """Demonstrates Agent using CustomChatOllama with basic configuration."""
    custom_llm = CustomChatOllama(model="llama3.2")
    agent = Agent(
        llm=custom_llm,
        system_prompt="You are a financial analyst assistant.",
        max_loops=1,
        verbose=True,
        temperature=0.7,
        max_tokens=100
    )
    response = agent.run("Generate a brief market summary.")
    return response


# Example: Agent with CustomChatOllama and Tools
def custom_ollama_tools_example() -> str:
    """Demonstrates Agent with CustomChatOllama and tool integration."""
    def sample_tool(query: str) -> str:
        return f"Processed: {query}"

    custom_llm = CustomChatOllama(model="llama3.2")
    agent = Agent(
        llm=custom_llm,
        system_prompt="Use tools to process financial queries.",
        tools=[sample_tool],
        tool_schema="json",
        execute_tool=True,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Process financial data with tool.")
    return response


# Example: Agent with CustomChatOllama and Streaming
def custom_ollama_streaming_example() -> str:
    """Demonstrates Agent with CustomChatOllama and streaming output."""
    custom_llm = CustomChatOllama(model="llama3.2")
    agent = Agent(
        llm=custom_llm,
        system_prompt="You are a storytelling assistant.",
        streaming_on=True,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Tell a short financial story.")
    return response
