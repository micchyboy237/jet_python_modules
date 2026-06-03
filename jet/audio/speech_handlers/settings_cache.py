"""
Settings Cache
==============
Persistent application settings storage under ~/.cache/live_speech_translator/.

Stored values:
- language: "auto" (default), "en", "ja"
- global_reset_on_clear: bool (default False) — whether to trigger a global
  reset when the clear button is pressed

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
DEFAULT_GLOBAL_RESET_ON_CLEAR: bool = False

_CACHE_DIR = Path.home() / ".cache" / "live_speech_translator"
_CACHE_FILE = _CACHE_DIR / "settings.json"


def _ensure_cache_dir() -> None:
    """Create the cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_settings() -> dict:
    """
    Load all settings from the cache file.
    Returns an empty dict if the file doesn't exist or is invalid.
    """
    try:
        if _CACHE_FILE.exists():
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_settings(data: dict) -> None:
    """Persist all settings to the cache file."""
    _ensure_cache_dir()
    _CACHE_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_language() -> LanguageCode:
    """
    Load the cached language preference.
    Returns DEFAULT_LANGUAGE if not found or invalid.
    """
    data = _load_settings()
    lang = data.get("language", DEFAULT_LANGUAGE)
    if lang in SUPPORTED_LANGUAGES:
        return lang
    return DEFAULT_LANGUAGE


def save_language(language: LanguageCode) -> None:
    """
    Save the language preference to the cache file.
    Preserves other existing settings.
    """
    if language not in SUPPORTED_LANGUAGES:
        return
    data = _load_settings()
    data["language"] = language
    _save_settings(data)


def load_global_reset_on_clear() -> bool:
    """Load the global-reset-on-clear checkbox preference."""
    data = _load_settings()
    return data.get("global_reset_on_clear", DEFAULT_GLOBAL_RESET_ON_CLEAR)


def save_global_reset_on_clear(value: bool) -> None:
    """Save the global-reset-on-clear checkbox preference."""
    data = _load_settings()
    data["global_reset_on_clear"] = bool(value)
    _save_settings(data)


class AppSettingsStore:
    """
    Thread-safe in-memory + persistent settings store.
    Use this to share settings state between components.

    Stores:
    - language: current transcription language
    - global_reset_on_clear: whether to call /global/reset on clear
    """

    def __init__(
        self,
        initial_language: LanguageCode | None = None,
        initial_global_reset: bool | None = None,
    ) -> None:
        self._lock = threading.Lock()

        # Load existing settings once
        data = _load_settings()

        self._language: LanguageCode = (
            initial_language
            if initial_language in SUPPORTED_LANGUAGES
            else data.get("language", DEFAULT_LANGUAGE)
        )
        if self._language not in SUPPORTED_LANGUAGES:
            self._language = DEFAULT_LANGUAGE

        self._global_reset_on_clear: bool = (
            initial_global_reset
            if initial_global_reset is not None
            else data.get("global_reset_on_clear", DEFAULT_GLOBAL_RESET_ON_CLEAR)
        )

    # ---- Language ----

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

    # ---- Global Reset on Clear ----

    @property
    def global_reset_on_clear(self) -> bool:
        with self._lock:
            return self._global_reset_on_clear

    def set_global_reset_on_clear(self, value: bool) -> None:
        """Set the global-reset-on-clear flag and persist to cache."""
        with self._lock:
            self._global_reset_on_clear = bool(value)
        save_global_reset_on_clear(self._global_reset_on_clear)
