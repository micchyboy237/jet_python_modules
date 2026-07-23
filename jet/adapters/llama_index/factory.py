import logging
import os
from typing import Any, Dict, Optional

import nest_asyncio
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like.base import OpenAILike
from llama_index.memory.mem0 import Mem0Memory
from mem0 import Memory, MemoryClient
from mem0.configs.base import MemoryConfig

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_llama_cpp_llm(**kwargs) -> OpenAILike:
    """Reuse jet's llama_cpp config. OpenAILike skips llama_index's hardcoded
    OpenAI-model-name validation, so local/custom model names (e.g. Qwen3.5-4B) work."""
    logger.info(f"Building LLM client -> model={LLM_MODEL} base_url={LLM_BASE_URL}")
    settings = dict(
        model=LLM_MODEL,
        api_base=LLM_BASE_URL,
        api_key="not-needed",
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=10000,  # adjust to match your local model's actual context length
        timeout=120.0,
        additional_kwargs={
            "enable_thinking": False,
            "stream_options": {"include_usage": True},
        },
        **kwargs,
    )
    return OpenAILike(**settings)


def get_llama_cpp_embed_model(**kwargs) -> OpenAIEmbedding:
    """Reuse jet's llama_cpp config for embeddings, same pattern as factory.get_embedding_client()."""
    logger.info(
        f"Building embed model client -> model={EMBED_MODEL} base_url={EMBED_BASE_URL}"
    )
    settings = dict(model=EMBED_MODEL, api_base=EMBED_BASE_URL, timeout=30.0, **kwargs)
    return OpenAIEmbedding(**kwargs)


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

    Args:
        user_id: Unique identifier for the user (required for filtering)
        agent_id: Optional agent identifier for multi-agent scenarios
        run_id: Optional run/session identifier
        collection_name: Name for the FAISS collection (default: "mem0_memories")
        storage_path: Path for FAISS storage (default: ~/.mem0/mem0_faiss_store)
        embed_dims: Embedding dimensions (default: 768 for nomic-embed-text)
        search_msg_limit: Number of recent messages to use in search context (default: 5)

    Returns:
        Configured Mem0Memory instance ready for use

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
        f"Creating local Mem0Memory for user_id={user_id}, "
        f"agent_id={agent_id}, run_id={run_id}"
    )

    # Build context dict with at least one non-null field
    context_dict = {"user_id": user_id}
    if agent_id is not None:
        context_dict["agent_id"] = agent_id
    if run_id is not None:
        context_dict["run_id"] = run_id

    logger.debug(f"Context: {context_dict}")

    # Set default storage path if not provided
    if storage_path is None:
        home_dir = os.path.expanduser("~")
        storage_path = os.path.join(home_dir, ".mem0", "mem0_faiss_store")

    logger.info(
        f"Using storage path: {storage_path}, collection: {collection_name}, "
        f"embed_dims: {embed_dims}"
    )

    # Build configuration dictionary
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
        # Create mem0 Memory instance from config
        config = MemoryConfig(**config_dict)
        mem0_client = Memory(config)
        logger.info("Successfully created mem0 Memory client from config")

        # Create Mem0Memory wrapper with local client
        memory = Mem0Memory.from_config(
            context=context_dict,
            config=config_dict,
            search_msg_limit=search_msg_limit,
        )

        logger.info(
            f"Local Mem0Memory created successfully with search_msg_limit={search_msg_limit}"
        )
        return memory

    except Exception as e:
        logger.error(f"Failed to create local Mem0Memory: {e}", exc_info=True)
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
        Configured Mem0Memory instance connected to cloud API

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
        f"Creating cloud Mem0Memory for user_id={user_id}, "
        f"agent_id={agent_id}, run_id={run_id}"
    )

    # Build context dict with at least one non-null field
    context_dict = {"user_id": user_id}
    if agent_id is not None:
        context_dict["agent_id"] = agent_id
    if run_id is not None:
        context_dict["run_id"] = run_id

    logger.debug(f"Context: {context_dict}")

    try:
        # Create MemoryClient instance
        client = MemoryClient(
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
        )
        logger.info("Successfully created MemoryClient")

        # Create Mem0Memory wrapper with cloud client
        memory = Mem0Memory.from_client(
            context=context_dict,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
            search_msg_limit=search_msg_limit,
        )

        logger.info(
            f"Cloud Mem0Memory created successfully with search_msg_limit={search_msg_limit}"
        )
        return memory

    except Exception as e:
        logger.error(f"Failed to create cloud Mem0Memory: {e}", exc_info=True)
        raise
