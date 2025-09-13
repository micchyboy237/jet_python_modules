"""Implement a RAG agent using ChromaDB memory in AutoGen v0.4.

This module demonstrates how to create a Retrieval-Augmented Generation (RAG) agent in AutoGen v0.4 using `ChromaDBVectorMemory` for persistent storage. It replaces the v0.2 `Teachability` approach by integrating a memory store with an `AssistantAgent`, supporting tool usage and custom memory implementations.
"""

import os
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from autogen_core.memory import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig


def get_weather(city: str) -> str:
    return f"The weather in {city} is 72 degree and sunny."


chroma_user_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="preferences",
        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
        k=2,
        score_threshold=0.4,
    )
)

assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    tools=[get_weather],
    memory=[chroma_user_memory],
)
