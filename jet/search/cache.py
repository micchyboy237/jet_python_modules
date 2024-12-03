import os
import json
import threading
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from .config import CACHE_DIR, CACHE_DURATION, CACHE_FILE


class Cache:
    _lock = threading.Lock()

    def __init__(self, cache_dir: str = CACHE_DIR, cache_duration: int = CACHE_DURATION, cache_file: str = CACHE_FILE):
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        self.cache_file = cache_file
        self.cache_file_path = os.path.join(cache_dir, self.cache_file)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        self.cache = {}  # Initialize cache attribute
        self.load_cache()

    def _hash_key(self, key: str) -> str:
        """Generate a SHA-256 hash of the key for consistent access."""
        return hashlib.sha256(key.encode()).hexdigest()

    def load_cache(self):
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r") as file:
                    self.cache = json.load(file)
            except json.JSONDecodeError:
                # If file is empty or corrupt, initialize an empty cache
                self.cache = {}

    def save_cache(self):
        with open(self.cache_file_path, "w") as file:
            json.dump(self.cache, file, indent=2, ensure_ascii=False)

    def get(self, key: str) -> Optional[List[Dict]]:
        hashed_key = self._hash_key(key)
        data = self.cache.get(hashed_key)
        if data:
            if time.time() - data["timestamp"] < self.cache_duration:
                return data["data"]
            else:
                del self.cache[hashed_key]
                self.save_cache()
        return None

    def set(self, key: str, data: List[Dict]):
        hashed_key = self._hash_key(key)
        self.cache[hashed_key] = {
            "data": data,
            "timestamp": time.time()
        }
        self.save_cache()

    def clear(self):
        self.cache = {}
        self.save_cache()

    def clean_expired(self):
        current_time = time.time()
        with self._lock:
            keys_to_delete = [
                key for key, value in self.cache.items()
                if current_time - value["timestamp"] >= self.cache_duration
            ]
            # Delete expired entries
            for key in keys_to_delete:
                del self.cache[key]
            # Save updated cache state
            self.save_cache()
