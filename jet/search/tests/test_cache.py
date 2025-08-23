import unittest
import tempfile
import os
import time
from search import Cache


class TestCache(unittest.TestCase):
    def setUp(self):
        # Create a temporary file to avoid overwriting the main cache file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache = Cache(
            cache_duration=2, cache_file=self.temp_file.name)  # Use custom parameters

    def tearDown(self):
        # Remove the temporary file and reset the cache
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_set_and_get_cache(self):
        self.cache.set("test_key", [{"value": "test_data"}])
        result = self.cache.get("test_key")
        self.assertIsNotNone(result)
        self.assertEqual(result, [{"value": "test_data"}])

    def test_cache_expiry(self):
        self.cache.set("expiring_key", [{"value": "test_data"}])
        time.sleep(3)  # Wait for cache to expire
        result = self.cache.get("expiring_key")
        self.assertIsNone(result)  # Expired cache should return None

    def test_clear_cache(self):
        self.cache.set("test_key", [{"value": "test_data"}])
        self.cache.clear()
        self.assertIsNone(self.cache.get("test_key"))

    def test_clean_expired(self):
        self.cache.set("key1", [{"value": "data1"}])
        # Ensure key1 is expired by waiting longer than CACHE_DURATION
        time.sleep(3)
        self.cache.set("key2", [{"value": "data2"}])
        self.cache.clean_expired()
        # key1 should be expired and cleaned
        self.assertIsNone(self.cache.get("key1"))
        # key2 should still be available
        self.assertIsNotNone(self.cache.get("key2"))

    def test_get_clears_expired_key(self):
        self.cache.set("expired_key", [{"value": "expired_data"}])
        time.sleep(2)
        result = self.cache.get("expired_key")
        self.assertIsNone(result)
        self.assertNotIn("expired_key", self.cache.cache)


if __name__ == "__main__":
    unittest.main()
