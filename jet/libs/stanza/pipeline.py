import stanza
from threading import Lock
from typing import Dict, Optional

class StanzaPipelineCache:
    """Singleton cache for Stanza pipelines to avoid repeated initialization."""
    _instance: Optional['StanzaPipelineCache'] = None
    _lock: Lock = Lock()
    _pipelines: Dict[str, stanza.Pipeline] = {}

    def __new__(cls) -> 'StanzaPipelineCache':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StanzaPipelineCache, cls).__new__(cls)
            return cls._instance

    def get_pipeline(self, lang: str = "en", processors: str = "tokenize,pos,lemma,depparse,ner", use_gpu: bool = True) -> stanza.Pipeline:
        """Retrieve or create a Stanza pipeline with the given configuration."""
        key = f"{lang}:{processors}:{use_gpu}"
        with self._lock:
            if key not in self._pipelines:
                stanza.download(lang, processors=processors, verbose=False)
                self._pipelines[key] = stanza.Pipeline(
                    lang=lang, processors=processors, use_gpu=use_gpu, verbose=False
                )
            return self._pipelines[key]

    def clear_cache(self) -> None:
        """Clear all cached pipelines (for testing or resource management)."""
        with self._lock:
            self._pipelines.clear()