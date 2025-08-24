# jet_python_modules/jet/llm/mlx/prompt_cache_manager.py
from typing import Any, Dict, List, Optional
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache


class PromptCacheManager:
    """Manages session-based prompt caches for MLX models."""

    def __init__(self):
        """Initialize an empty prompt cache dictionary."""
        self._cache: Dict[str, List[Any]] = {}

    def get_cache(self, session_id: str) -> Optional[List[Any]]:
        """
        Retrieve the prompt cache for a given session ID.

        Args:
            session_id (str): The session ID to look up.

        Returns:
            Optional[List[Any]]: The prompt cache if it exists, else None.
        """
        return self._cache.get(session_id)

    def set_cache(self, session_id: str, cache: List[Any]) -> None:
        """
        Store a prompt cache for a given session ID.

        Args:
            session_id (str): The session ID to associate with the cache.
            cache (List[Any]): The prompt cache to store.

        Raises:
            TypeError: If cache is not a list.
        """
        if not isinstance(cache, list):
            raise TypeError(
                f"Prompt cache must be a list, got {type(cache).__name__}")
        self._cache[session_id] = cache

    def clear_cache(self, session_id: str) -> None:
        """
        Remove the prompt cache for a given session ID.

        Args:
            session_id (str): The session ID whose cache should be removed.
        """
        self._cache.pop(session_id, None)

    def clear_all_caches(self) -> None:
        """Remove all prompt caches."""
        self._cache.clear()

    def initialize_cache(self, model: nn.Module, draft_model: Optional[nn.Module] = None, session_id: str = "", max_kv_size: Optional[int] = None) -> List[Any]:
        """
        Initialize a new prompt cache for a session using the model's make_prompt_cache.

        Args:
            model (nn.Module): The MLX model to create the cache for.
            draft_model (Optional[nn.Module]): The draft model, if any.
            session_id (str): The session ID to associate with the cache.
            max_kv_size (Optional[int]): Maximum size for RotatingKVCache, if applicable.

        Returns:
            List[Any]: The newly created prompt cache.
        """
        cache = make_prompt_cache(model, max_kv_size)
        if draft_model is not None:
            cache.extend(make_prompt_cache(draft_model, max_kv_size))
        if session_id:
            self.set_cache(session_id, cache)
        return cache
