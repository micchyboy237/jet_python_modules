import unittest
from lru_cache import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_put_and_get(self):
        cache = LRUCache(max_size=2)
        cache.put("A", 1)
        cache.put("B", "text")
        self.assertEqual(cache.get("A"), 1)
        self.assertEqual(cache.get("B"), "text")
        self.assertIsNone(cache.get("C"))

    def test_eviction(self):
        cache = LRUCache(max_size=2)
        cache.put("A", 1)
        cache.put("B", 2.5)
        cache.put("C", [1, 2, 3])  # "A" should be evicted
        self.assertIsNone(cache.get("A"))
        self.assertEqual(cache.get("B"), 2.5)
        self.assertEqual(cache.get("C"), [1, 2, 3])

    def test_recently_used(self):
        cache = LRUCache(max_size=2)
        cache.put("A", {"key": "value"})
        cache.put("B", (1, 2))
        cache.get("A")  # "A" becomes most recently used
        cache.put("C", {3, 4})  # "B" should be evicted
        self.assertEqual(cache.get("A"), {"key": "value"})
        self.assertIsNone(cache.get("B"))
        self.assertEqual(cache.get("C"), {3, 4})

    def test_clear(self):
        cache = LRUCache(max_size=2)
        cache.put("A", True)
        cache.put("B", None)
        cache.clear()
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
