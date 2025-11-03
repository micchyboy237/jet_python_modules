import stanza
from threading import Lock
from typing import Optional, Tuple, Any, Generator
from contextlib import contextmanager
from stanza.server import CoreNLPClient

from jet.logger import logger


class StanzaPipelineCache:
    """Singleton cache for a single Stanza pipeline to avoid repeated initialization."""
    _instance: Optional['StanzaPipelineCache'] = None
    _lock: Lock = Lock()
    _pipeline: Optional[stanza.Pipeline] = None
    _config: Optional[Tuple[str, str, bool, bool, Tuple[Tuple[str, Any], ...]]] = None

    def __new__(cls, *args, **kwargs) -> 'StanzaPipelineCache':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StanzaPipelineCache, cls).__new__(cls)
            return cls._instance

    def __init__(self, lang: str = "en", processors: str = "tokenize,mwt,pos,lemma,depparse,ner,sentiment,constituency",
                 use_gpu: bool = True, verbose: bool = False, **kwargs):
        """Optionally initialize pipeline on instantiation."""
        config = (lang, processors, use_gpu, verbose, tuple(sorted(kwargs.items())))
        if self._pipeline is None or self._config != config:
            self._init_pipeline(lang, processors, use_gpu, verbose, **kwargs)

    def _init_pipeline(self, lang: str, processors: str, use_gpu: bool, verbose: bool, **kwargs):
        """Internal initializer for pipeline setup."""
        logger.info(f"Creating new Stanza pipeline for lang={lang}, processors={processors}")
        stanza.download(lang, processors=processors, verbose=verbose)
        self._pipeline = stanza.Pipeline(
            lang=lang, processors=processors, use_gpu=use_gpu, verbose=verbose, **kwargs
        )
        self._config = (lang, processors, use_gpu, verbose, tuple(sorted(kwargs.items())))

    def get_pipeline(self, lang: Optional[str] = None, processors: Optional[str] = None,
                    use_gpu: Optional[bool] = None, verbose: Optional[bool] = None, **kwargs) -> stanza.Pipeline:
        """Retrieve or create a single Stanza pipeline, replacing any existing one if config differs."""
        with self._lock:
            # Use existing config as fallback
            if self._config:
                current_lang, current_processors, current_use_gpu, current_verbose, current_kwargs = self._config
            else:
                current_lang, current_processors, current_use_gpu, current_verbose, current_kwargs = (
                    "en", "tokenize,mwt,pos,lemma,depparse,ner,sentiment,constituency", True, False, tuple()
                )

            merged_lang = lang or current_lang
            merged_processors = processors or current_processors
            merged_use_gpu = use_gpu if use_gpu is not None else current_use_gpu
            merged_verbose = verbose if verbose is not None else current_verbose
            merged_kwargs = dict(current_kwargs)
            merged_kwargs.update(kwargs)

            config = (
                merged_lang,
                merged_processors,
                merged_use_gpu,
                merged_verbose,
                tuple(sorted(merged_kwargs.items())),
            )

            if self._pipeline is None or self._config != config:
                self._init_pipeline(merged_lang, merged_processors, merged_use_gpu, merged_verbose, **merged_kwargs)

            return self._pipeline


    def clear_cache(self) -> None:
        """Clear the cached pipeline (for testing or resource management)."""
        with self._lock:
            self._pipeline = None
            self._config = None

    def extract_sentences(self, text: str) -> list[str]:
        """Extract POS tags or token strings."""
        from jet.wordnet.sentence import split_sentences
        sentences = split_sentences(text)
        return sentences

    def extract_pos(self, text: str) -> list[str]:
        """Extract POS tags or token strings."""
        sentences = self.extract_sentences(text)
        results = []
        for sentence in sentences:
            doc = self._pipeline(sentence)
            results.extend([sent.tokens_string() for sent in doc.sentences])
        return results

    def extract_entities(self, text: str) -> list[dict]:
        """Extract named entities."""
        sentences = self.extract_sentences(text)
        results = []
        for sentence in sentences:
            doc = self._pipeline(sentence)
            results.extend([
                {"text": ent.text, "type": ent.type, "start_char": ent.start_char, "end_char": ent.end_char}
                for ent in doc.ents
            ])
        return results

    def extract_dependencies(self, text: str) -> list[str]:
        """Extract dependency relations."""
        sentences = self.extract_sentences(text)
        results = []
        for sentence in sentences:
            doc = self._pipeline(sentence)
            results.extend([sent.dependencies_string() for sent in doc.sentences])
        return results

    def extract_constituencies(self, text: str) -> list[str]:
        """Extract POS tags or token strings."""
        sentences = self.extract_sentences(text)
        results = []
        for sentence in sentences:
            doc = self._pipeline(sentence)
            results.extend([str(sent.constituency) for sent in doc.sentences])
        return results

    @classmethod
    @contextmanager
    def corenlp_client(cls, **client_kwargs) -> Generator[CoreNLPClient, None, None]:
        """Thread-safe context manager for a shared CoreNLPClient instance."""
        with cls._lock:
            if not hasattr(cls, '_corenlp_client') or cls._corenlp_client is None:
                logger.info("Starting shared CoreNLPClient")
                cls._corenlp_client = CoreNLPClient(preload=False, **client_kwargs)
                cls._corenlp_client.start()
            client = cls._corenlp_client
        try:
            yield client
        finally:
            # Do not stop here; optional manual stop via clear_corenlp()
            pass

    @classmethod
    def clear_corenlp(cls) -> None:
        """Manually stop and clear the shared CoreNLPClient (call at script end if needed)."""
        with cls._lock:
            if hasattr(cls, '_corenlp_client') and cls._corenlp_client is not None:
                logger.info("Stopping shared CoreNLPClient")
                cls._corenlp_client.stop()
                cls._corenlp_client = None

    def extract_scenes(self, text: str) -> list[str]:
        """Extract scene graphs using shared CoreNLPClient."""
        sentences = self.extract_sentences(text)
        results = []
        with self.corenlp_client() as client:  # Uses class-level shared client
            for sentence in sentences:
                scenegraph = client.scenegraph(sentence)
                results.append(scenegraph)
        return results
