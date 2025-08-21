"""Configure an OpenAI-compatible model client for AutoGen v0.4.

This module illustrates how to use `OpenAIChatCompletionClient` to connect to an OpenAI-compatible API in AutoGen v0.4. It specifies a custom model, base URL, API key, and model capabilities (e.g., vision, function calling) to enable compatibility with non-OpenAI APIs for agent interactions.
"""

from autogen_ext.models.openai import OpenAIChatCompletionClient

custom_model_client = OpenAIChatCompletionClient(
    model="custom-model-name",
    base_url="https://custom-model.com/reset/of/the/path",
    api_key="placeholder",
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
)