# %% [markdown]
# # LongRAG example
#
# This LlamaPack implements LongRAG based on [this paper](https://arxiv.org/pdf/2406.15319).
#
# LongRAG retrieves large tokens at a time, with each retrieval unit being ~6k tokens long, consisting of entire documents or groups of documents. This contrasts the short retrieval units (100 word passages) of traditional RAG. LongRAG is advantageous because results can be achieved using only the top 4-8 retrieval units, and long-context LLMs can better understand the context of the documents because long retrieval units preserve their semantic integrity.

# %% [markdown]
# ## Setup

# %%
import json
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
import typing as t
from IPython.display import Markdown, display
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.packs.longrag import LongRAGPack
import os
import asyncio

# Custom
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager
from jet.logger import logger
from jet.file import save_json


# %%
# %pip install llama-index

# %%

# os.environ["OPENAI_API_KEY"] = "<Your API Key>"

# %% [markdown]
# ## Usage

# %% [markdown]
# Below shows the usage of `LongRAGPack` using the `gpt-4o` LLM, which is able to handle long context inputs.

# %%


class OllamaCallbackManager(CallbackManager):
    def on_event_start(
        self,
        *args,
        **kwargs: any,
    ):
        logger.log("OllamaCallbackManager on_event_start:", {
                   **args, **kwargs}, colors=["LOG", "INFO"])

    def on_event_end(
        self,
        *args,
        **kwargs: any,
    ):
        logger.log("OllamaCallbackManager on_event_end:", {
                   **args, **kwargs}, colors=["LOG", "INFO"])


async def main():
    data_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/llama_index/llama-index-packs/llama-index-packs-longrag/examples/data"
    logger.debug("Loading settings...")
    settings = {
        "llm_model": "llama3.1",
        "embedding_model": "mxbai-embed-large",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "base_url": "http://localhost:11434",
    }
    results = {
        "settings": settings,
        "queries": []
    }
    logger.log("Settings:", json.dumps(settings), colors=["GRAY", "DEBUG"])
    save_json(results, file_path="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/rag/generated/paul_graham/results.json")

    Settings.chunk_size = settings["chunk_size"]
    Settings.chunk_overlap = settings["chunk_overlap"]
    Settings.embed_model = OllamaEmbedding(
        model_name=settings["embedding_model"],
        base_url=settings["base_url"],
        callback_manager=OllamaCallbackManager(),
    )
    Settings.llm = Ollama(
        temperature=0,
        request_timeout=120.0,
        model=settings["llm_model"],
        base_url=settings["base_url"],
    )

    logger.debug("Loading LongRAGPack 1...")
    settings = {}
    pack = LongRAGPack(data_dir=data_dir, **settings)

    # %%

    query_str = (
        "How can Pittsburgh become a startup hub, and what are the two types of moderates?"
    )
    logger.debug("Loading query 1...")
    res = await pack.run(query_str)
    results["queries"].append({
        "settings": settings,
        "query": query_str,
        "response": res,
    })
    save_json(res, file_path="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/rag/generated/paul_graham/results.json")
    display(Markdown(str(res)))

    # %% [markdown]
    # Other parameters include `chunk_size`, `similarity_top_k`, and `small_chunk_size`.
    # - `chunk_size`: To demonstrate how different documents are grouped together, documents are split into nodes of `chunk_size` tokens, then re-grouped based on the relationships between the nodes. Because this does not affect the final answer, it can be disabled by setting `chunk_size` to None. The default size is 4096.
    # - `similarity_top_k`: Retrieves the top k large retrieval units. The default is 8, and based on the paper, the ideal range is 4-8.
    # - `small_chunk_size`: To compare similarities, each large retrieval unit is split into smaller child retrieval units of `small_chunk_size` tokens. The embeddings of these smaller retrieval units are compared to the query embeddings. The top k large parent retrieval units are chosen based on the maximum scores of their smaller child retrieval units. The default size is 512.

    # %%
    logger.debug("Loading LongRAGPack 2...")
    settings = {
        "chunk_size": None,
        "similarity_top_k": 4,
    }
    pack = LongRAGPack(data_dir=data_dir, **settings)
    query_str = (
        "How can Pittsburgh become a startup hub, and what are the two types of moderates?"
    )
    logger.debug("Loading query 2...")
    res = await pack.run(query_str)
    results["queries"].append({
        "settings": settings,
        "query": query_str,
        "response": res,
    })
    save_json(res, file_path="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/rag/generated/paul_graham/results.json")
    display(Markdown(str(res)))

    # %% [markdown]
    # ## Vector Storage
    #
    # The vector index can be extracted and be persisted to disk. A `LongRAGPack` can also be constructed given a vector index. Below is an example of persisting the index to disk.

    # %%
    logger.debug("Persisting VectorStoreIndex...")
    modules = pack.get_modules()
    index = t.cast(VectorStoreIndex, modules["index"])
    index.storage_context.persist(
        persist_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/rag/generated/paul_graham")

    # %% [markdown]
    # Below is an example of loading an index.

    # %%
    logger.debug("Loading VectorStoreIndex...")
    ctx = StorageContext.from_defaults(
        persist_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/rag/generated/paul_graham")
    index = load_index_from_storage(ctx)

    logger.debug("Loading LongRAGPack 3...")
    settings = {
        "chunk_size": None,
        "similarity_top_k": 4,
    }
    pack_from_idx = LongRAGPack(data_dir=data_dir, index=index)
    query_str = (
        "How can Pittsburgh become a startup hub, and what are the two types of moderates?"
    )
    logger.debug("Loading query 3...")
    res = await pack_from_idx.run(query_str)
    results["queries"].append({
        "settings": {
            "ctx": ctx
        },
        "query": query_str,
        "response": res,
    })
    save_json(res, file_path="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/rag/generated/paul_graham/results.json")
    display(Markdown(str(res)))


if __name__ == "__main__":
    asyncio.run(main())
