from typing import Any, List, Optional
from autogen_core.memory import MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.memory._list_memory import ListMemory, ListMemoryConfig
from autogen_core.memory._base_memory import ChatCompletionContext, CancellationToken
from jet.llm.mlx.memory import MemoryManager


class MemoryList(ListMemory):
    """A list-based memory implementation with optional MemoryManager integration.

    This class extends ListMemory to add text-based filtering for queries and
    optional integration with MemoryManager for persistent storage and advanced
    querying. If a MemoryManager is provided, high-priority memories are persisted,
    and queries can be delegated for vector-based search.
    """

    def __init__(
        self,
        name: str | None = None,
        memory_contents: List[MemoryContent] | None = None,
        memory_manager: Optional[MemoryManager] = None
    ) -> None:
        """Initialize the MemoryList.

        Args:
            name: Optional identifier for this memory instance.
            memory_contents: Optional initial list of memory contents.
            memory_manager: Optional MemoryManager instance for persistent storage and advanced queries.
        """
        super().__init__(name=name, memory_contents=memory_contents)
        self._memory_manager = memory_manager

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add new content to memory and optionally persist to MemoryManager.

        Args:
            content: Memory content to store.
            cancellation_token: Optional token to cancel operation.
        """
        self._contents.append(content)
        if self._memory_manager and content.metadata and content.metadata.get("priority") == "high":
            await self._memory_manager.add(content, cancellation_token)

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query memories with text-based filtering or delegate to MemoryManager.

        If a MemoryManager is available and the query is a string, delegates to
        MemoryManager for vector-based search. Otherwise, filters local memories
        based on whether the query string is present in the content (case-insensitive).

        Args:
            query: The search query string or MemoryContent object.
            cancellation_token: Optional token to cancel operation.
            **kwargs: Additional parameters for MemoryManager query.

        Returns:
            MemoryQueryResult containing filtered or retrieved memories.
        """
        _ = cancellation_token
        if self._memory_manager and isinstance(query, str):
            return await self._memory_manager.query(query, cancellation_token, **kwargs)

        query_text = ""
        if isinstance(query, str):
            query_text = query.lower()
        elif isinstance(query, MemoryContent) and isinstance(query.content, str):
            query_text = query.content.lower()

        if not query_text:
            return MemoryQueryResult(results=self._contents)

        filtered_contents = [
            content for content in self._contents
            if isinstance(content.content, str) and query_text in content.content.lower()
        ]
        return MemoryQueryResult(results=filtered_contents)
