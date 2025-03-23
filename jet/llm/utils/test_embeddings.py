import unittest
import os
import time
from unittest.mock import patch

from jet.data.utils import hash_text
from jet.llm.models import DEFAULT_SF_EMBED_MODEL
from jet.llm.utils.embeddings import get_embedding_function, initialize_embed_function
from jet.cache.joblib.utils import CACHE_TTL, save_cache, load_cache, ttl_cache


class TestEmbeddingCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup runs once before all tests."""
        cls.model_name = DEFAULT_SF_EMBED_MODEL
        cls.batch_size = 16
        cls.embed_func = get_embedding_function(cls.model_name, cls.batch_size)

        # Create a test cache directory
        cls.cache_dir = "cache"
        os.makedirs(cls.cache_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Cleanup runs once after all tests."""
        if os.path.exists(cls.cache_dir):
            for file in os.listdir(cls.cache_dir):
                os.remove(os.path.join(cls.cache_dir, file))
            os.rmdir(cls.cache_dir)

    def setUp(self):
        """Runs before each test - clears TTL cache."""
        ttl_cache.clear()

    def test_single_text_caching(self):
        """Test if a single text embedding is cached correctly."""
        text = "Hello world"
        text_hash = hash_text(text)

        # ✅ Correct the patch path
        with patch("jet.llm.utils.embeddings.initialize_embed_function") as mock_func:
            mock_func.return_value = lambda x: [
                0.1, 0.2, 0.3]  # Fake embedding

            # First call should compute & cache
            embedding = self.embed_func(text)
            self.assertEqual(embedding, [0.1, 0.2, 0.3])

            # Check if cached in TTL cache
            self.assertIn(text_hash, ttl_cache)

            # Second call should fetch from cache
            cached_embedding = self.embed_func(text)
            self.assertEqual(embedding, cached_embedding)

    def test_list_text_caching(self):
        """Test if a list of texts is cached correctly, storing each entry separately."""
        texts = ["Hello", "World"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]

        with patch("jet.llm.utils.embeddings.initialize_embed_function") as mock_func:
            mock_func.return_value = lambda x: expected_embeddings if isinstance(x, list) else [
                0.1, 0.2]

            # First call should compute & cache both
            embeddings = self.embed_func(texts)
            self.assertEqual(embeddings, expected_embeddings)

            # Second call should fetch from cache
            cached_embeddings = self.embed_func(texts)
            self.assertEqual(embeddings, cached_embeddings)

    def test_persistent_cache_storage(self):
        """Test if embeddings are stored & loaded from persistent cache."""
        text = "Persistent test"
        text_hash = hash_text(text)
        file_path = f"cache/{text_hash}.pkl"
        embedding = [0.5, 0.6, 0.7]

        # Save to persistent cache
        save_cache(file_path, embedding)
        self.assertTrue(os.path.exists(file_path))

        # Load from persistent cache
        loaded_embedding = load_cache(file_path)
        self.assertEqual(embedding, loaded_embedding)

    def test_ttl_cache_expiry(self):
        """Test if TTL cache expires after the set TTL."""
        text = "Expiring text"
        text_hash = hash_text(text)
        embedding = [0.9, 1.0, 1.1]

        # Store in TTL cache
        ttl_cache[text_hash] = embedding
        self.assertIn(text_hash, ttl_cache)

        # ✅ Force TTL cache to expire before checking
        time.sleep(1.5)  # Ensure this is slightly longer than the TTL
        expired_hashes = ttl_cache.expire()  # Explicitly call expiration

        self.assertNotIn(text_hash, expired_hashes,
                         "TTL cache entry should have expired.")

    def test_mixed_cache_hit_and_miss(self):
        """Test handling of a batch where some items are cached and some are not."""
        texts = ["Cached text", "New text"]
        cached_text_hash = hash_text(texts[0])
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]

        # Pre-cache first text
        ttl_cache[cached_text_hash] = expected_embeddings[0]

        with patch("jet.llm.utils.embeddings.initialize_embed_function") as mock_func:
            mock_func.return_value = lambda x: expected_embeddings[1:] if isinstance(x, list) else [
                0.3, 0.4]

            embeddings = self.embed_func(texts)

            self.assertEqual(embeddings, expected_embeddings)

            # Ensure both are now cached
            self.assertIn(hash_text(texts[1]), ttl_cache)

    def test_invalid_input(self):
        """Test that invalid input types raise errors."""
        with self.assertRaises(TypeError):
            self.embed_func(12345)  # Invalid type (not str or list of str)


if __name__ == "__main__":
    unittest.main()
