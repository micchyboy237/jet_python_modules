from collections import OrderedDict
from typing import ItemsView, Any


class LRUCache:
    def __init__(self, max_size: int = 5) -> None:
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size: int = max_size

    def get(self, key: str) -> Any:
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
        return self.cache.get(key)

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest item

        self.cache[key] = value

    def clear(self) -> None:
        """Clear all items in the cache."""
        self.cache.clear()

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self.cache)

    def items(self) -> ItemsView[str, Any]:
        """Return a view of the cache's items (key-value pairs)."""
        return self.cache.items()

    def __iter__(self):
        """Allow iteration over the cache keys."""
        return iter(self.cache)
