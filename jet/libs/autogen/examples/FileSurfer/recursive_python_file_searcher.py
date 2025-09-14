import os
import traceback
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.models import LLMMessage, AssistantMessage, UserMessage
from pydantic import BaseModel
from typing_extensions import Self


class RAGPythonFileSearcherConfig(BaseModel):
    """Configuration for RAGPythonFileSearcher agent"""
    name: str
    # must be a chat model (e.g., OllamaChatCompletionClient)
    model_client: ComponentModel
    base_path: str
    description: str | None = None


class RAGPythonFileSearcher(BaseChatAgent, Component[RAGPythonFileSearcherConfig]):
    """
    A Retrieval-Augmented Generation (RAG) agent for Python code search.
    Combines semantic search + keyword search, then uses an LLM to summarize results.
    """

    component_config_schema = RAGPythonFileSearcherConfig
    component_provider_override = (
        "autogen_ext.agents.file_surfer.rag_python_file_searcher.RAGPythonFileSearcher"
    )

    DEFAULT_DESCRIPTION = "An agent that uses RAG (semantic + keyword search + LLM summarization) to find relevant Python code."

    def __init__(
        self,
        name: str,
        model_client: ComponentModel,
        base_path: str,
        description: str = DEFAULT_DESCRIPTION,
    ) -> None:
        super().__init__(name, description)
        self._model_client = model_client
        self._base_path = base_path
        self._chat_history: List[LLMMessage] = []

        # Build vector index on init
        self._index, self._chunks = self._build_index()

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        for chat_message in messages:
            self._chat_history.append(chat_message.to_model_message())
        try:
            last_message = self._chat_history[-1]
            assert isinstance(last_message, UserMessage)
            query = last_message.content

            # Retrieve candidates
            semantic_results = await self._semantic_search(query, top_k=5)
            keyword_results = self._keyword_search(query)

            retrieved = semantic_results + \
                [r for r in keyword_results if r not in semantic_results]

            if retrieved:
                # Ask LLM to summarize
                prompt = (
                    f"You are a code search assistant.\n"
                    f"User query: {query}\n\n"
                    f"Here are retrieved code snippets:\n\n"
                    + "\n".join(retrieved)
                    + "\n\nPlease summarize which files are most relevant to the query and explain why."
                )

                llm_response = await self._model_client.complete(prompt)
                content = llm_response.output_text
            else:
                content = "No matches found."

            self._chat_history.append(AssistantMessage(
                content=content, source=self.name))
            return Response(chat_message=TextMessage(content=content, source=self.name))

        except BaseException:
            content = f"Search error:\n\n{traceback.format_exc()}"
            self._chat_history.append(AssistantMessage(
                content=content, source=self.name))
            return Response(chat_message=TextMessage(content=content, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._chat_history.clear()

    def _collect_python_chunks(self) -> List[Tuple[str, str]]:
        """Return list of (file_path, code_snippet)"""
        chunks: List[Tuple[str, str]] = []
        for root, _, files in os.walk(self._base_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            code = f.read()
                        # simple chunking by ~40 lines
                        lines = code.splitlines()
                        for i in range(0, len(lines), 40):
                            snippet = "\n".join(lines[i:i+40])
                            chunks.append((full_path, snippet))
                    except Exception:
                        continue
        return chunks

    def _build_index(self):
        """Build FAISS index of embeddings."""
        chunks = self._collect_python_chunks()
        if not chunks:
            return None, []

        # Generate embeddings via LLM client
        texts = [snippet for _, snippet in chunks]
        embeddings = self._model_client.embed(
            texts)  # must implement embedding API
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        return index, chunks

    async def _semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        if self._index is None:
            return []
        query_emb = self._model_client.embed([query])[0]
        D, I = self._index.search(
            np.array([query_emb]).astype("float32"), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self._chunks):
                file, snippet = self._chunks[idx]
                results.append(f"[Semantic] {file}\n{snippet[:200]}...\n")
        return results

    def _keyword_search(self, query: str) -> List[str]:
        matches: List[str] = []
        for root, _, files in os.walk(self._base_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if query.lower() in content.lower():
                            matches.append(f"[Keyword] {full_path}")
                    except Exception:
                        continue
        return matches

    def _to_config(self) -> RAGPythonFileSearcherConfig:
        return RAGPythonFileSearcherConfig(
            name=self.name,
            model_client=self._model_client.dump_component(),
            base_path=self._base_path,
            description=self.description,
        )

    @classmethod
    def _from_config(cls, config: RAGPythonFileSearcherConfig) -> Self:
        return cls(
            name=config.name,
            model_client=config.model_client,
            base_path=config.base_path,
            description=config.description or cls.DEFAULT_DESCRIPTION,
        )


if __name__ == "__main__":
    import asyncio
    from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient

    async def main():
        client = OllamaChatCompletionClient(model="llama3.2")
        searcher = RAGPythonFileSearcher(
            name="RAGPySearcher",
            model_client=client,
            base_path="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/swarms",
        )
        result = await searcher.run(task="web search agents with RAG")
        print(result.messages[-1].content)

    asyncio.run(main())
