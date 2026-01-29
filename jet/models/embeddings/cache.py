import hashlib
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union, Literal, List, Dict
import zlib
import logging

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_FILE = "embedding_cache.pkl"
CACHE_DIR = Path.home() / ".cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / CACHE_FILE

MEMORY_CACHE_MAX_SIZE = 10000
MEMORY_CACHE_PRUNE_RATIO = 0.8


class EmbeddingCache:
    """Modular cache for embeddings with LRU, TTL, and multi-backend support.

    Supports memory, file, and SQLite backends for caching embeddings with least-recently-used (LRU)
    eviction and time-to-live (TTL) expiration. Thread-safe with a lock for concurrent access.

    For file backend, embeddings are saved to `embedding_cache.pkl` in the specified `cache_dir`.
    For SQLite backend (default), embeddings are saved to `embedding_cache.db` in the specified `cache_dir`.
    The `cache_dir` can be customized to any valid directory path; it is created if it does not exist.
    When `overwrite` is True, existing cache data is cleared (file deleted, table dropped, or memory cache reset).

    Args:
        backend (Literal["memory", "file", "sqlite"]): Cache storage type (default: "sqlite").
        max_size (int): Maximum number of cache entries before LRU eviction (default: 10000).
        ttl (Optional[int]): Time-to-live in seconds for cache entries; None for no expiration.
        namespace (str): Prefix for cache keys to avoid collisions (default: "").
        cache_dir (Path): Directory for file or SQLite cache storage (default: ~/.cache/embeddings).
        overwrite (bool): If True, clear existing cache data on initialization (default: False).

    Raises:
        ValueError: If `cache_dir` is not a valid directory path for file or SQLite backends.

    Examples:
        # SQLite-based cache (default, saves to ~/.cache/embeddings/embedding_cache.db)
        cache = EmbeddingCache(max_size=100, ttl=7200, overwrite=True)
        cache.set("key1", [[1.0, 2.0], [3.0, 4.0]])
        embeddings = cache.get("key1")  # Returns [[1.0, 2.0], [3.0, 4.0]]
        cache.close()  # Closes SQLite connection

        # File-based cache (saves to ./custom_cache/embedding_cache.pkl)
        from pathlib import Path
        cache = EmbeddingCache(backend="file", cache_dir=Path("./custom_cache"), max_size=100, overwrite=True)
        cache.set("key2", [[5.0, 6.0]])
        embeddings = cache.get("key2")  # Returns [[5.0, 6.0]]
        cache.close()  # No-op for file backend

        # Memory-based cache (resets to empty cache if overwrite=True)
        cache = EmbeddingCache(backend="memory", max_size=100, ttl=3600, overwrite=True)
        cache.set("key3", [[7.0, 8.0]])
        embeddings = cache.get("key3")  # Returns [[7.0, 8.0]]
        cache.close()  # No-op for memory backend
    """

    def __init__(
        self,
        backend: Literal["memory", "file", "sqlite"] = "sqlite",
        max_size: int = MEMORY_CACHE_MAX_SIZE,
        ttl: Optional[int] = None,
        namespace: str = "",
        cache_dir: Path = CACHE_DIR,
        overwrite: bool = False,
    ) -> None:
        self.backend: Literal["memory", "file", "sqlite"] = backend
        self.max_size: int = max_size
        self.ttl: Optional[int] = ttl
        self.namespace: str = namespace
        self.cache_dir: Path = cache_dir

        # Validate cache_dir for file and sqlite backends
        if backend in ("file", "sqlite"):
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                if not self.cache_dir.is_dir():
                    raise ValueError(
                        f"cache_dir '{self.cache_dir}' is not a valid directory"
                    )
            except OSError as e:
                raise ValueError(
                    f"Failed to create or access cache_dir '{self.cache_dir}': {e}"
                )

        if backend == "memory":
            self._store: OrderedDict[str, tuple[List[List[float]], float]] = (
                OrderedDict()
            )
            if overwrite:
                self._store.clear()
                logger.debug("Memory cache cleared due to overwrite=True")
        elif backend == "file":
            self._store: OrderedDict[str, tuple[List[List[float]], float]] = (
                OrderedDict()
            )
            self._cache_file: Path = self.cache_dir / CACHE_FILE
            if overwrite and self._cache_file.exists():
                try:
                    self._cache_file.unlink()
                    logger.debug(f"Deleted existing cache file: {self._cache_file}")
                except OSError as e:
                    logger.error(
                        f"Failed to delete cache file '{self._cache_file}': {e}"
                    )
            self._load_file()
        elif backend == "sqlite":
            self._db_path: Path = self.cache_dir / "embedding_cache.db"
            if overwrite and self._db_path.exists():
                try:
                    self._db_path.unlink()
                    logger.debug(f"Deleted existing SQLite database: {self._db_path}")
                except OSError as e:
                    logger.error(
                        f"Failed to delete SQLite database '{self._db_path}': {e}"
                    )
            self._init_sqlite()
        self._lock: threading.Lock = threading.Lock()

    def _normalize_text(self, text: Union[str, List[str]]) -> str:
        """Normalize text for consistent hashing (lowercase, strip whitespace)."""
        if isinstance(text, str):
            return text.lower().strip()
        return "".join(sorted(t.lower().strip() for t in text))

    def _generate_key(self, text: Union[str, List[str]]) -> str:
        """Generate hash-based key with namespace."""
        norm_text: str = self._normalize_text(text)
        text_hash: str = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{text_hash}"

    def _load_file(self) -> None:
        """Load from pickle+zlib for file backend only."""
        if self.backend != "file":
            return
        try:
            if self._cache_file.exists():
                with open(self._cache_file, "rb") as f:
                    compressed: bytes = f.read()
                    data: bytes = zlib.decompress(compressed)
                    cache_data: Dict[str, tuple[List[List[float]], float]] = (
                        pickle.loads(data)
                    )
                    self._store = OrderedDict(
                        sorted(cache_data.items(), key=lambda x: x[0])
                    )
                logger.debug(f"Loaded file cache with {len(self._store)} entries.")
        except Exception as e:
            logger.debug(f"Failed to load file cache: {e}")
            self._store = OrderedDict()

    def _save_file(self) -> None:
        """Save to pickle+zlib with LRU prune for file backend only."""
        if self.backend != "file":
            return
        try:
            if len(self._store) > self.max_size:
                while len(self._store) > self.max_size:
                    self._store.popitem(last=False)
            data: bytes = pickle.dumps(dict(self._store))
            compressed: bytes = zlib.compress(data, level=6)
            with open(self._cache_file, "wb") as f:
                f.write(compressed)
            logger.debug(f"Saved file cache with {len(self._store)} entries.")
        except Exception as e:
            logger.error(f"Failed to save file cache: {e}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite DB for persistent, queryable cache."""
        self._conn: sqlite3.Connection = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                embedding BLOB,
                timestamp REAL
            )
        """)
        self._conn.commit()

    def get(self, key: str) -> Optional[List[List[float]]]:
        """Retrieve with TTL check."""
        with self._lock:
            if self.backend == "memory":
                if key in self._store:
                    if self.ttl and self._store[key][1] < time.time() - self.ttl:
                        del self._store[key]
                        return None
                    self._store.move_to_end(key)
                    return self._store[key][0]
            elif self.backend == "file":
                self._load_file()
                if key in self._store:
                    if self.ttl and self._store[key][1] < time.time() - self.ttl:
                        del self._store[key]
                        self._save_file()
                        return None
                    self._store.move_to_end(key)
                    return self._store[key][0]
            elif self.backend == "sqlite":
                cur: sqlite3.Cursor = self._conn.cursor()
                cur.execute(
                    "SELECT embedding FROM embeddings WHERE key = ? AND (timestamp > ? OR ? IS NULL)",
                    (key, time.time() - self.ttl if self.ttl else 0, self.ttl),
                )
                row: Optional[tuple[bytes]] = cur.fetchone()
                if row:
                    return pickle.loads(row[0])
                return None
            return None

    def set(self, key: str, embedding: List[List[float]]) -> None:
        """Store embedding."""
        with self._lock:
            serialized: bytes = pickle.dumps(embedding)
            ts: float = time.time()
            if self.backend == "memory":
                self._store[key] = (embedding, ts)
                if len(self._store) > self.max_size:
                    self._store.popitem(last=False)
            elif self.backend == "file":
                self._store[key] = (embedding, ts)
                self._save_file()
            elif self.backend == "sqlite":
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings (key, embedding, timestamp) VALUES (?, ?, ?)",
                    (key, serialized, ts),
                )
                self._conn.execute(
                    "DELETE FROM embeddings WHERE key NOT IN (SELECT key FROM embeddings ORDER BY timestamp DESC LIMIT ?)",
                    (self.max_size,),
                )
                self._conn.commit()

    def reset(self) -> None:
        """
        Completely clear/reset the entire cache (all entries removed).
        Works for all backends: memory, file, sqlite.
        """
        with self._lock:
            if self.backend == "memory":
                self._store.clear()
                logger.debug("Memory cache cleared")

            elif self.backend == "file":
                self._store.clear()
                if self._cache_file.exists():
                    try:
                        self._cache_file.unlink()
                        logger.debug(f"Deleted cache file: {self._cache_file}")
                    except OSError as e:
                        logger.error(
                            f"Failed to delete cache file '{self._cache_file}': {e}"
                        )
                logger.debug("File cache reset")

            elif self.backend == "sqlite":
                try:
                    self._conn.execute("DROP TABLE IF EXISTS embeddings")
                    self._conn.execute("""
                        CREATE TABLE embeddings (
                            key TEXT PRIMARY KEY,
                            embedding BLOB,
                            timestamp REAL
                        )
                    """)
                    self._conn.commit()
                    logger.debug("SQLite cache table reset")
                except sqlite3.Error as e:
                    logger.error(f"Failed to reset SQLite table: {e}")
                    self._conn.execute("DELETE FROM embeddings")
                    self._conn.commit()
                    logger.debug("SQLite cache cleared (fallback)")

    def close(self) -> None:
        """Cleanup."""
        if self.backend == "sqlite":
            self._conn.close()
