"""Initialize an OpenAI chat completion client for use with AutoGen v0.4.

This module demonstrates how to create an instance of `OpenAIChatCompletionClient` using the v0.4 API, replacing the v0.2 `OpenAIWrapper`. It configures the client with a specific model and API key, enabling interaction with OpenAI's chat completion API for agent-based workflows.
"""

from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key="sk-xxx")
