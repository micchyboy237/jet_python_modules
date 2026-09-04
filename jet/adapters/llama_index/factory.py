import logging
import os
from typing import Any, Dict, List, Optional

import nest_asyncio
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like.base import OpenAILike
from llama_index.memory.mem0 import Mem0Memory
from llama_index.memory.mem0.utils import convert_messages_to_string, format_memory_json
from mem0 import Memory, MemoryClient
from mem0.configs.base import MemoryConfig

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

dispatcher = get_dispatcher(__name__)


def get_llama_cpp_llm() -> OpenAILike:
    """Reuse jet's llama_cpp config. OpenAILike skips llama_index's hardcoded
    OpenAI-model-name validation, so local/custom model names (e.g. Qwen3.5-4B) work."""
    logger.info(f"Building LLM client -> model={LLM_MODEL} base_url={LLM_BASE_URL}")
    return OpenAILike(
        model=LLM_MODEL,
        api_base=LLM_BASE_URL,
        api_key="not-needed",
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=10000,
        timeout=120.0,
        additional_kwargs={
            "stream_options": {"include_usage": True},
        },
        extra_body={
            "enable_thinking": False,
        },
    )


def get_llama_cpp_embed_model(**kwargs) -> OpenAILikeEmbedding:
    """Reuse jet's llama_cpp config for embeddings using OpenAILikeEmbedding
    to support custom/local model names without strict validation."""
    logger.info(
        f"Building embed model client -> model={EMBED_MODEL} base_url={EMBED_BASE_URL}"
    )
    return OpenAILikeEmbedding(
        model_name=EMBED_MODEL,
        api_base=EMBED_BASE_URL,
        api_key="not-needed",
        timeout=30.0,
        **kwargs,
    )


# ─── Reranker ───────────────────────────────────────────────────────────────


class LlamaCppRerank(BaseNodePostprocessor):
    """LlamaIndex node postprocessor that calls a llama.cpp ``/v1/rerank``
    endpoint (or any service that speaks the same wire format).

    llama.cpp rerank wire format (POST /v1/rerank):
        Request:  {"model": "...", "query": "...", "documents": [...], "top_n": N}
        Response: {"results": [{"index": 0, "relevance_score": 1.23}, ...]}

    This avoids pulling a multi-GB cross-encoder model into the Python
    process — the heavy lifting happens on your llama.cpp server.
    """

    base_url: str = Field(
        default="http://127.0.0.1:8082/v1",
        description="Base URL of the llama.cpp reranker server (include /v1).",
    )
    model: str = Field(
        default="bge-rerank:v2-m3",
        description="Model alias passed in the request body (must match --alias on server).",
    )
    top_n: int = Field(
        default=5,
        description="Number of top-scored nodes to return.",
    )
    timeout: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="If True, preserve the original retrieval score in node metadata.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "LlamaCppRerank"

    def _call_api(self, query: str, documents: List[str]) -> List[dict]:
        """Send a rerank request to the llama.cpp /v1/rerank endpoint.

        Returns the ``results`` list: [{"index": int, "relevance_score": float}, ...]
        """
        import httpx

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": self.top_n,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()["results"]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model,
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            documents = [
                node.node.get_content(metadata_mode=MetadataMode.ALL) for node in nodes
            ]
            scores = self._call_api(query_bundle.query_str, documents)

            # Map scores back by original index (llama.cpp returns sorted by score desc)
            for score_item in scores:
                original_index = score_item["index"]
                if self.keep_retrieval_score:
                    nodes[original_index].node.metadata["retrieval_score"] = nodes[
                        original_index
                    ].score
                nodes[original_index].score = float(score_item["relevance_score"])

            new_nodes = sorted(nodes, key=lambda x: -(x.score if x.score else 0))[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes


def get_llama_cpp_reranker(
    top_n: int = 5,
    timeout: float = 30.0,
    keep_retrieval_score: bool = False,
    **kwargs,
) -> LlamaCppRerank:
    """Create a LlamaCppRerank postprocessor using jet's llama_cpp config.

    Reads RERANK_MODEL and RERANK_BASE_URL from jet.adapters.llama_cpp.config
    (which in turn read from env vars LLAMA_CPP_RERANK_MODEL and LLAMA_CPP_RERANK_URL).

    Args:
        top_n: Number of top-scored nodes to return after reranking.
        timeout: HTTP timeout in seconds for the rerank request.
        keep_retrieval_score: If True, preserve the original retrieval score
            in each node's metadata under the key ``retrieval_score``.
        **kwargs: Additional keyword arguments forwarded to LlamaCppRerank.

    Returns:
        Configured LlamaCppRerank postprocessor ready for use in a query pipeline.

    Raises:
        ValueError: If RERANK_MODEL or RERANK_BASE_URL is not configured.

    Example:
        >>> from jet.adapters.llama_index.factory import get_llama_cpp_reranker
        >>> reranker = get_llama_cpp_reranker(top_n=5)
        >>> # Use in a query engine:
        >>> query_engine = index.as_query_engine(
        ...     similarity_top_k=20,
        ...     node_postprocessors=[reranker],
        ... )
    """
    if not RERANK_MODEL:
        raise ValueError(
            "RERANK_MODEL is not set. "
            "Set LLAMA_CPP_RERANK_MODEL env var (e.g. 'bge-rerank:v2-m3')."
        )
    if not RERANK_BASE_URL:
        raise ValueError(
            "RERANK_BASE_URL is not set. "
            "Set LLAMA_CPP_RERANK_URL env var (e.g. 'http://192.168.68.150:8082/v1')."
        )

    logger.info(
        f"Building reranker client -> model={RERANK_MODEL} base_url={RERANK_BASE_URL}"
    )
    return LlamaCppRerank(
        model=RERANK_MODEL,
        base_url=RERANK_BASE_URL,
        top_n=top_n,
        timeout=timeout,
        keep_retrieval_score=keep_retrieval_score,
        **kwargs,
    )


# ─── Mem0 Memory ────────────────────────────────────────────────────────────


def _build_mem0_config_dict(
    llm_model: str,
    llm_base_url: str,
    embed_model: str,
    embed_base_url: str,
    embed_dims: int,
    collection_name: str,
    storage_path: str,
) -> Dict[str, Any]:
    """Build mem0 configuration dictionary matching example patterns.

    Args:
        llm_model: Model name for LLM provider
        llm_base_url: Base URL for LLM service
        embed_model: Model name for embedding service
        embed_base_url: Base URL for embedding service
        embed_dims: Embedding dimensions (must match model output)
        collection_name: FAISS collection name
        storage_path: Path for FAISS vector store

    Returns:
        Configuration dictionary compatible with MemoryConfig
    """
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": "not-needed",
                "openai_base_url": llm_base_url,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embed_model,
                "api_key": "not-needed",
                "openai_base_url": embed_base_url,
                "embedding_dims": embed_dims,
            },
        },
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": embed_dims,
                "path": storage_path,
            },
        },
    }


def get_mem0_local_memory(
    user_id: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    collection_name: str = "mem0_memories",
    storage_path: Optional[str] = None,
    embed_dims: int = 768,
    search_msg_limit: int = 5,
) -> Mem0Memory:
    """Create local Mem0Memory using llama.cpp backend with FAISS vector store.

    This function creates a memory instance that runs entirely locally, using:
    - Local llama.cpp server for LLM inference
    - Local llama.cpp server for embeddings
    - FAISS for vector storage on disk

    Uses SafeMem0Memory internally to avoid a second SYSTEM ChatMessage
    being injected into chat history (see SafeMem0Memory docstring for why
    this matters with strict chat templates like llama.cpp's Jinja template).

    Args:
        user_id: Unique identifier for the user (required for filtering)
        agent_id: Optional agent identifier for multi-agent scenarios
        run_id: Optional run/session identifier
        collection_name: Name for the FAISS collection (default: "mem0_memories")
        storage_path: Path for FAISS storage (default: ~/.mem0/mem0_faiss_store)
        embed_dims: Embedding dimensions (default: 768 for nomic-embed-text)
        search_msg_limit: Number of recent messages to use in search context (default: 5)

    Returns:
        Configured SafeMem0Memory instance ready for use

    Raises:
        ValidationError: If context validation fails (no identifiers provided)
        Exception: If mem0 initialization fails

    Example:
        >>> memory = get_mem0_local_memory(user_id="student_123")
        >>> # Or with custom storage
        >>> memory = get_mem0_local_memory(
        ...     user_id="student_123",
        ...     collection_name="learning_memories",
        ...     storage_path="./my_memories",
        ...     embed_dims=768,
        ... )
    """
    logger.info(
        f"Creating local SafeMem0Memory for user_id={user_id}, "
        f"agent_id={agent_id}, run_id={run_id}"
    )

    context_dict = {"user_id": user_id}
    if agent_id is not None:
        context_dict["agent_id"] = agent_id
    if run_id is not None:
        context_dict["run_id"] = run_id
    logger.debug(f"Context: {context_dict}")

    if storage_path is None:
        home_dir = os.path.expanduser("~")
        storage_path = os.path.join(home_dir, ".mem0", "mem0_faiss_store")

    logger.info(
        f"Using storage path: {storage_path}, collection: {collection_name}, "
        f"embed_dims: {embed_dims}"
    )

    config_dict = _build_mem0_config_dict(
        llm_model=LLM_MODEL,
        llm_base_url=LLM_BASE_URL,
        embed_model=EMBED_MODEL,
        embed_base_url=EMBED_BASE_URL,
        embed_dims=embed_dims,
        collection_name=collection_name,
        storage_path=storage_path,
    )
    logger.debug(f"Mem0 config: {config_dict}")

    try:
        config = MemoryConfig(**config_dict)
        mem0_client = Memory(config)
        logger.info("Successfully created mem0 Memory client from config")

        memory = SafeMem0Memory.from_config(
            context=context_dict,
            config=config_dict,
            search_msg_limit=search_msg_limit,
        )
        logger.info(
            f"Local SafeMem0Memory created successfully with search_msg_limit={search_msg_limit}"
        )
        return memory

    except Exception as e:
        logger.error(f"Failed to create local SafeMem0Memory: {e}", exc_info=True)
        raise


def get_mem0_cloud_memory(
    user_id: str,
    api_key: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    host: Optional[str] = None,
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
    search_msg_limit: int = 5,
) -> Mem0Memory:
    """Create cloud-based Mem0Memory using MemoryClient API.

    This function creates a memory instance that connects to Mem0's cloud API,
    enabling persistent, scalable memory storage without local infrastructure.

    Uses SafeMem0Memory internally to avoid a second SYSTEM ChatMessage
    being injected into chat history (see SafeMem0Memory docstring for why
    this matters with strict chat templates like llama.cpp's Jinja template).

    Args:
        user_id: Unique identifier for the user (required for filtering)
        api_key: Mem0 API key for authentication
        agent_id: Optional agent identifier for multi-agent scenarios
        run_id: Optional run/session identifier
        host: Optional custom API host (uses Mem0 default if None)
        org_id: Optional organization ID for multi-tenant setups
        project_id: Optional project ID for project-scoped memories
        search_msg_limit: Number of recent messages to use in search context (default: 5)

    Returns:
        Configured SafeMem0Memory instance connected to cloud API

    Raises:
        ValidationError: If context validation fails (no identifiers provided)
        ValueError: If api_key is empty or None
        Exception: If MemoryClient initialization fails

    Example:
        >>> import os
        >>> memory = get_mem0_cloud_memory(
        ...     user_id="student_123",
        ...     api_key=os.getenv("MEM0_API_KEY"),
        ... )
        >>> # With full context
        >>> memory = get_mem0_cloud_memory(
        ...     user_id="student_123",
        ...     api_key=os.getenv("MEM0_API_KEY"),
        ...     agent_id="tutor_agent",
        ...     run_id="session_2024_01",
        ... )
    """
    if not api_key:
        raise ValueError(
            "api_key is required for cloud memory. Set MEM0_API_KEY environment variable."
        )

    logger.info(
        f"Creating cloud SafeMem0Memory for user_id={user_id}, "
        f"agent_id={agent_id}, run_id={run_id}"
    )

    context_dict = {"user_id": user_id}
    if agent_id is not None:
        context_dict["agent_id"] = agent_id
    if run_id is not None:
        context_dict["run_id"] = run_id
    logger.debug(f"Context: {context_dict}")

    try:
        client = MemoryClient(
            api_key=api_key,
            host=host,
        )
        logger.info("Successfully created MemoryClient")

        memory = SafeMem0Memory.from_client(
            context=context_dict,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
            search_msg_limit=search_msg_limit,
        )
        logger.info(
            f"Cloud SafeMem0Memory created successfully with search_msg_limit={search_msg_limit}"
        )
        return memory

    except Exception as e:
        logger.error(f"Failed to create cloud SafeMem0Memory: {e}", exc_info=True)
        raise


class SafeMem0Memory(Mem0Memory):
    """
    Drop-in replacement for Mem0Memory that avoids injecting a second
    SYSTEM ChatMessage into chat history.

    Why: AgentWorkflow/FunctionAgent already prepends `agent.system_prompt`
    as the FIRST system message before sending to the LLM. The stock
    Mem0Memory.get() *also* inserts its own system message (built from
    DEFAULT_INTRO_PREFERENCES) at index 0 of the history it returns,
    which becomes index 1 once the agent's prompt is prepended.

    Some chat templates (e.g. local llama.cpp Jinja templates) reject
    any system message that isn't the very first message, causing:
        "System message must be at the beginning" (HTTP 500)

    This override folds retrieved mem0 memories into the latest user
    message as plain context text instead of a new system message,
    so only ONE system message (the agent's own) ever exists.
    """

    def get(self, input: Optional[str] = None, **kwargs: Any) -> list[ChatMessage]:
        """Get chat history, injecting mem0 context into the last user
        message instead of adding a second SYSTEM message.

        Args:
            input: Optional current user input, used to build the mem0
                search query alongside recent chat history.
            **kwargs: Passed through to primary_memory.get().

        Returns:
            List[ChatMessage] with at most the original SYSTEM message(s)
            preserved (if any existed already in primary_memory), plus
            memory context folded into the trailing USER message.
        """
        import time

        t0 = time.time()
        logger.info("[SafeMem0Memory.get] fetching primary_memory history...")
        messages = self.primary_memory.get(input=input, **kwargs)
        logger.info(
            f"[SafeMem0Memory.get] primary_memory.get done in {time.time() - t0:.2f}s "
            f"({len(messages)} messages)"
        )

        search_input = convert_messages_to_string(
            messages, input, limit=self.search_msg_limit
        )

        flt = self.context.build_filter()

        t1 = time.time()
        logger.info(f"[SafeMem0Memory.get] calling mem0 search(filters={flt})...")
        result = self.search(query=search_input, filters=flt)
        logger.info(
            f"[SafeMem0Memory.get] mem0 search() done in {time.time() - t1:.2f}s "
            f"(first call may take longer — lazy model loads)"
        )

        search_results = result.get("results", [])
        logger.info(
            f"[SafeMem0Memory] Retrieved {len(search_results)} memory "
            f"result(s) for context filter={flt}"
        )

        if not search_results:
            logger.debug(
                "[SafeMem0Memory] No memory results found; returning history as-is."
            )
            return messages

        memory_text = "\n".join(format_memory_json(m) for m in search_results)
        context_block = (
            "Relevant context retrieved from memory "
            "(not from the current conversation):\n"
            f"{memory_text}\n"
        )

        if messages and messages[-1].role == MessageRole.USER:
            original = messages[-1].content or ""
            messages[-1] = ChatMessage(
                role=MessageRole.USER,
                content=f"{context_block}\n{original}",
            )
            logger.info(
                "[SafeMem0Memory] Injected memory context into last USER message "
                "(no extra system message added)."
            )
        else:
            messages.append(ChatMessage(role=MessageRole.USER, content=context_block))
            logger.info(
                "[SafeMem0Memory] No trailing USER message found — appended memory "
                "context as a new USER message instead."
            )

        return messages
