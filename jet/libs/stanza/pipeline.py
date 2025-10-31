import stanza
from threading import Lock
from typing import Optional, Tuple
from jet.logger import logger

class StanzaPipelineCache:
    """Singleton cache for a single Stanza pipeline to avoid repeated initialization."""
    _instance: Optional['StanzaPipelineCache'] = None
    _lock: Lock = Lock()
    _pipeline: Optional[stanza.Pipeline] = None
    _config: Optional[Tuple[str, str, bool]] = None

    def __new__(cls) -> 'StanzaPipelineCache':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StanzaPipelineCache, cls).__new__(cls)
            return cls._instance

    def get_pipeline(self, lang: str = "en", processors: str = "tokenize,pos,lemma,depparse,ner", use_gpu: bool = True, verbose: bool = False, **kwargs) -> stanza.Pipeline:
        """Retrieve or create a single Stanza pipeline, replacing any existing one."""
        # Include verbose and sorted kwargs in config for accurate cache key
        config = (lang, processors, use_gpu, verbose, tuple(sorted(kwargs.items())))
        with self._lock:
            if self._pipeline is None or self._config != config:
                logger.info(f"Creating new Stanza pipeline for config: {config}")
                stanza.download(lang, processors=processors, verbose=verbose)
                self._pipeline = stanza.Pipeline(
                    lang=lang, processors=processors, use_gpu=use_gpu, verbose=verbose, **kwargs
                )
                self._config = config
            else:
                logger.debug(f"Reusing cached Stanza pipeline for config: {config}")
            return self._pipeline

    def clear_cache(self) -> None:
        """Clear the cached pipeline (for testing or resource management)."""
        with self._lock:
            self._pipeline = None
            self._config = None