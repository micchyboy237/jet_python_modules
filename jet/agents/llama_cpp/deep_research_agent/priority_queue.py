"""Async-safe priority queue with URL deduplication for agentic search."""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class QueueItem:
    """Priority queue item. Lower score = higher priority (min-heap)."""

    sort_key: float
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)
    sub_query_id: str = field(compare=False, default="")
    parent_url: str = field(compare=False, default="")
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)


class AgentPriorityQueue:
    """Min-heap priority queue with O(1) URL deduplication."""

    def __init__(self, max_depth: int = 2):
        self._heap: list[QueueItem] = []
        self._seen_urls: set[str] = set()
        self._lock = asyncio.Lock()
        self.max_depth = max_depth

    async def push(self, item: QueueItem) -> bool:
        """Add item if URL not seen and depth within limit. Returns True if added."""
        if item.depth > self.max_depth:
            return False
        async with self._lock:
            if item.url in self._seen_urls:
                return False
            self._seen_urls.add(item.url)
            heapq.heappush(self._heap, item)
            return True

    async def push_many(self, items: list[QueueItem]) -> int:
        """Batch push. Returns count of items actually added."""
        added = 0
        for item in items:
            if await self.push(item):
                added += 1
        return added

    async def pop(self) -> QueueItem | None:
        """Pop highest-priority item. Returns None if empty."""
        async with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    async def is_empty(self) -> bool:
        async with self._lock:
            return len(self._heap) == 0

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def seen_count(self) -> int:
        return len(self._seen_urls)
