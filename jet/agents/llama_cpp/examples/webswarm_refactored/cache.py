import hashlib
import sqlite3
import time

from .config import CACHE_DB


class DedupCache:
    def __init__(self):
        self.conn = sqlite3.connect(CACHE_DB)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, ts REAL)"
        )

    def is_seen(self, query: str, url: str = "") -> bool:
        h = hashlib.sha256(f"{query}|{url}".encode()).hexdigest()
        return (
            self.conn.execute("SELECT 1 FROM seen WHERE hash=?", (h,)).fetchone()
            is not None
        )

    def mark_seen(self, query: str, url: str = ""):
        h = hashlib.sha256(f"{query}|{url}".encode()).hexdigest()
        self.conn.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (h, time.time()))
        self.conn.commit()
