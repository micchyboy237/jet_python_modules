"""Demonstrate caching for OpenAI model responses in AutoGen v0.4.

This module shows how to wrap an `OpenAIChatCompletionClient` with a `ChatCompletionCache` using a `DiskCacheStore` in AutoGen v0.4. It replaces the v0.2 caching mechanism (enabled via `cache_seed`) by explicitly configuring a disk-based cache to store and reuse model responses, improving efficiency for repeated queries.
"""

import asyncio
import tempfile
from autogen_core.models import UserMessage
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache


async def main():
    with tempfile.TemporaryDirectory() as tmpdirname:
        openai_model_client = OpenAIChatCompletionClient(model="gpt-4o")
        cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(tmpdirname))
        cache_client = ChatCompletionCache(openai_model_client, cache_store)
        response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
        print(response)
        response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
        print(response)
        await openai_model_client.close()

asyncio.run(main())
