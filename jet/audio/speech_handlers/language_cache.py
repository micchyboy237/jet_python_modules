"""
Language Cache
==============
Persistent language preference storage under ~/.cache/live_speech_translator/.

Supports values: "auto" (default), "en", "ja"
Thread-safe reads and writes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal

LanguageCode = Literal["auto", "en", "ja"]
SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("auto", "en", "ja")
DEFAULT_LANGUAGE: LanguageCode = "auto"

_CACHE_DIR = Path.home() / ".cache" / "live_speech_translator"
_CACHE_FILE = _CACHE_DIR / "language.json"


def _ensure_cache_dir() -> None:
    """Create the cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_language() -> LanguageCode:
    """
    Load the cached language preference.
    Returns DEFAULT_LANGUAGE if the file doesn't exist or is invalid.
    """
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            lang = data.get("language", DEFAULT_LANGUAGE)
            if lang in SUPPORTED_LANGUAGES:
                return lang  # type: ignore[return-value]
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return DEFAULT_LANGUAGE


def save_language(language: LanguageCode) -> None:
    """
    Save the language preference to the cache file.
    Silently ignores invalid values.
    """
    if language not in SUPPORTED_LANGUAGES:
        return
    _ensure_cache_dir()
    data = {"language": language}
    _CACHE_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


class LanguageStore:
    """
    Thread-safe in-memory + persistent language store.
    Use this to share language state between components.
    """

    def __init__(self, initial: LanguageCode | None = None) -> None:
        self._lock = threading.Lock()
        self._language: LanguageCode = (
            initial if initial in SUPPORTED_LANGUAGES else load_language()
        )

    @property
    def language(self) -> LanguageCode:
        with self._lock:
            return self._language

    def set_language(self, language: LanguageCode) -> None:
        """Set the language and persist to cache."""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Must be one of {SUPPORTED_LANGUAGES}"
            )
        with self._lock:
            self._language = language
        save_language(language)

    def cycle_language(self) -> LanguageCode:
        """
        Cycle to the next supported language and return it.
        Useful for a single-button toggle: auto → en → ja → auto.
        """
        with self._lock:
            idx = SUPPORTED_LANGUAGES.index(self._language)
            next_idx = (idx + 1) % len(SUPPORTED_LANGUAGES)
            self._language = SUPPORTED_LANGUAGES[next_idx]
        save_language(self._language)
        return self._language
