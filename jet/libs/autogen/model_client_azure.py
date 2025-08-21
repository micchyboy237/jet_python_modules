"""Initialize an Azure OpenAI chat completion client for AutoGen v0.4.

This module shows how to set up an `AzureOpenAIChatCompletionClient` in AutoGen v0.4, replacing the v0.2 `OpenAIWrapper` for Azure-hosted models. It configures the client with deployment details, endpoint, model, API version, and key for integration with Azure OpenAI services in agent workflows.
"""

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o",
    azure_endpoint="https://<your-endpoint>.openai.azure.com/",
    model="gpt-4o",
    api_version="2024-09-01-preview",
    api_key="sk-xxx",
)