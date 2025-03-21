import hashlib
import os
import pickle
from typing import Optional

from jet.data.utils import generate_unique_hash

CACHE_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/.cache/heuristics"
CACHE_FILE = "ngrams_cache.pkl"  # Name of the cache file


class CacheManager:
    _instance = None  # Singleton instance

    def __new__(cls, cache_dir=CACHE_DIR, cache_file=CACHE_FILE):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance.cache_dir = cache_dir
            cls._instance.cache_file = cache_file
            cls._instance.cache = cls._instance.load_cache()
        return cls._instance

    def _get_cache_path(self):
        """Return the full path of the cache file."""
        return os.path.join(self.cache_dir, self.cache_file)

    def get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the given file."""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_cache(self) -> dict:
        """Load the cache file if exists, otherwise return an empty dict."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                return pickle.load(cache_file)
        return {}

    def save_cache(self, data: dict) -> None:
        """Save the cache to a file."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_path()
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        self.cache = data  # Update in-memory cache

    def is_cache_valid(self, cache_data: Optional[dict] = None, file_path: Optional[str] = None) -> bool:
        """Check if the cache is valid by comparing file hashes."""
        try:
            cache_path = file_path or self._get_cache_path()
            cache_data = cache_data or self.cache
            current_file_hash = self.get_file_hash(cache_path)
            return cache_data.get("file_hash") == current_file_hash
        except FileNotFoundError:
            return False

    def update_cache(self, ngrams: list, file_path: Optional[str] = None) -> dict:
        """Regenerate the cache with new data."""
        cache_path = file_path or self._get_cache_path()
        try:
            current_file_hash = self.get_file_hash(cache_path)
        except FileNotFoundError:
            current_file_hash = generate_unique_hash()

        cache_data = {
            "file_hash": current_file_hash,
            "common_texts_ngrams": ngrams
        }
        self.save_cache(cache_data)
        return cache_data

    def invalidate_cache(self) -> None:
        """Clears the cache and removes the cache file."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
        self.cache = {}
