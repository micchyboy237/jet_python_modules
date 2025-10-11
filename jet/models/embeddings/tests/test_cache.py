import pytest
import time
from pathlib import Path
from jet.models.embeddings.cache import EmbeddingCache

@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Fixture to provide a temporary cache directory."""
    return tmp_path / "cache"

@pytest.fixture
def memory_cache(temp_cache_dir: Path) -> EmbeddingCache:
    """Fixture for memory-based cache."""
    return EmbeddingCache(backend="memory", cache_dir=temp_cache_dir, max_size=3)

@pytest.fixture
def file_cache(temp_cache_dir: Path) -> EmbeddingCache:
    """Fixture for file-based cache."""
    return EmbeddingCache(backend="file", cache_dir=temp_cache_dir, max_size=3)

@pytest.fixture
def sqlite_cache(temp_cache_dir: Path) -> EmbeddingCache:
    """Fixture for SQLite-based cache."""
    cache = EmbeddingCache(backend="sqlite", cache_dir=temp_cache_dir, max_size=3)
    yield cache
    cache.close()

class TestCacheHitMiss:
    """Tests for cache hit and miss scenarios."""

    @pytest.mark.parametrize("backend", ["memory", "file", "sqlite"])
    def test_cache_hit(self, backend: str, temp_cache_dir: Path, request: pytest.FixtureRequest) -> None:
        """Given a cached embedding, when retrieving it, then return the cached value."""
        # Given
        cache = request.getfixturevalue(f"{backend}_cache")
        input_text = "Hello, world!"
        embedding = [[1.0, 2.0, 3.0]]
        expected = embedding
        cache_key = cache._generate_key(input_text)
        cache.set(cache_key, embedding)

        # When
        result = cache.get(cache_key)

        # Then
        assert result == expected, f"Expected cached embedding {expected}, got {result}"

    @pytest.mark.parametrize("backend", ["memory", "file", "sqlite"])
    def test_cache_miss(self, backend: str, temp_cache_dir: Path, request: pytest.FixtureRequest) -> None:
        """Given an empty cache, when retrieving a non-existent key, then return None."""
        # Given
        cache = request.getfixturevalue(f"{backend}_cache")
        input_text = "Non-existent text"
        cache_key = cache._generate_key(input_text)
        expected = None

        # When
        result = cache.get(cache_key)

        # Then
        assert result == expected, f"Expected None for cache miss, got {result}"

class TestCacheEviction:
    """Tests for LRU eviction behavior."""

    @pytest.mark.parametrize("backend", ["memory", "file"])
    def test_lru_eviction(self, backend: str, temp_cache_dir: Path, request: pytest.FixtureRequest) -> None:
        """Given a full cache, when adding a new item, then evict the oldest entry."""
        # Given
        cache = request.getfixturevalue(f"{backend}_cache")
        texts = ["text1", "text2", "text3", "text4"]
        embeddings = [[float(i)] for i in range(1, 5)]
        expected = embeddings[1:]  # Expect oldest (text1) to be evicted
        for text, emb in zip(texts[:3], embeddings[:3]):
            cache.set(cache._generate_key(text), [emb])

        # When
        cache.set(cache._generate_key(texts[3]), [embeddings[3]])
        result = [cache.get(cache._generate_key(text)) for text in texts[1:]]

        # Then
        assert all(r == [e] for r, e in zip(result, expected)), f"Expected {expected}, got {result}"
        assert cache.get(cache._generate_key(texts[0])) is None, "Oldest entry should be evicted"

class TestCacheTTL:
    """Tests for TTL expiration behavior."""

    @pytest.mark.parametrize("backend", ["memory", "file", "sqlite"])
    def test_ttl_expiration(self, backend: str, temp_cache_dir: Path, request: pytest.FixtureRequest) -> None:
        """Given a cache with TTL, when time exceeds TTL, then return None."""
        # Given
        cache = request.getfixturevalue(f"{backend}_cache")
        cache.ttl = 1  # 1 second TTL
        input_text = "Expiring text"
        embedding = [[1.0, 2.0]]
        cache_key = cache._generate_key(input_text)
        cache.set(cache_key, embedding)
        expected = None

        # When
        time.sleep(1.1)  # Wait past TTL
        result = cache.get(cache_key)

        # Then
        assert result == expected, f"Expected None after TTL, got {result}"

class TestCachePersistence:
    """Tests for file and SQLite cache persistence."""

    def test_file_cache_persistence(self, temp_cache_dir: Path) -> None:
        """Given a file cache, when reloading, then preserve cached data."""
        # Given
        cache1 = EmbeddingCache(backend="file", cache_dir=temp_cache_dir)
        input_text = "Persistent text"
        embedding = [[1.0, 2.0, 3.0]]
        cache_key = cache1._generate_key(input_text)
        cache1.set(cache_key, embedding)
        expected = embedding

        # When
        cache2 = EmbeddingCache(backend="file", cache_dir=temp_cache_dir)
        result = cache2.get(cache_key)

        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_sqlite_cache_persistence(self, temp_cache_dir: Path) -> None:
        """Given a SQLite cache, when reloading, then preserve cached data."""
        # Given
        cache1 = EmbeddingCache(backend="sqlite", cache_dir=temp_cache_dir)
        input_text = "Persistent text"
        embedding = [[1.0, 2.0, 3.0]]
        cache_key = cache1._generate_key(input_text)
        cache1.set(cache_key, embedding)
        cache1.close()
        expected = embedding

        # When
        cache2 = EmbeddingCache(backend="sqlite", cache_dir=temp_cache_dir)
        result = cache2.get(cache_key)
        cache2.close()

        # Then
        assert result == expected, f"Expected {expected}, got {result}"

class TestCacheNormalization:
    """Tests for text normalization in cache keys."""

    @pytest.mark.parametrize("backend", ["memory", "file", "sqlite"])
    def test_text_normalization(self, backend: str, temp_cache_dir: Path, request: pytest.FixtureRequest) -> None:
        """Given texts with varying case/whitespace, when caching, then use same key."""
        # Given
        cache = request.getfixturevalue(f"{backend}_cache")
        texts = ["Hello World", "hello world ", "  HELLO WORLD"]
        embedding = [[1.0, 2.0]]
        cache_key = cache._generate_key(texts[0])
        cache.set(cache_key, embedding)
        expected = embedding

        # When
        results = [cache.get(cache._generate_key(text)) for text in texts]

        # Then
        assert all(result == expected for result in results), f"Expected {expected} for all normalized texts, got {results}"

class TestCacheOverwrite:
    """Tests for cache overwrite behavior."""
    @pytest.mark.parametrize("backend", ["memory", "file", "sqlite"])
    def test_overwrite_clears_cache(self, backend: str, temp_cache_dir: Path, request: pytest.FixtureRequest) -> None:
        """Given an existing cache, when initializing with overwrite=True, then clear existing data."""
        # Given: Create a cache with data
        cache1 = EmbeddingCache(backend=backend, cache_dir=temp_cache_dir)
        input_text = "Test text"
        embedding = [[1.0, 2.0, 3.0]]
        cache_key = cache1._generate_key(input_text)
        cache1.set(cache_key, embedding)
        cache1.close()

        # When: Initialize a new cache with overwrite=True
        cache2 = EmbeddingCache(backend=backend, cache_dir=temp_cache_dir, overwrite=True)
        
        # Then: Expect no data in the cache
        expected = None
        result = cache2.get(cache_key)
        assert result == expected, f"Expected None after overwrite, got {result}"
        cache2.close()

@pytest.fixture(autouse=True)
def cleanup_caches(temp_cache_dir: Path):
    """Cleanup SQLite connections and files after tests."""
    yield
    for db_file in temp_cache_dir.glob("*.db"):
        db_file.unlink()
    for cache_file in temp_cache_dir.glob("*.pkl"):
        cache_file.unlink()