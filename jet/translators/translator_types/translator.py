from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Literal, Sequence
from typing_extensions import NotRequired, TypedDict

import ctranslate2
from threading import RLock

from jet.logger import logger

Device = Literal["cpu", "cuda", "auto"]
BatchType = Literal["examples", "tokens"]

# ----------------------------------------------------------------------
# Result containers – keep as TypedDict (never instantiated)
# ----------------------------------------------------------------------
class ExecutionStats(TypedDict):
    num_tokens: int
    num_examples: int
    total_time_in_ms: float

class TranslationResult(TypedDict):
    hypotheses: list[list[str]]
    scores: NotRequired[list[float]]
    attention: NotRequired[list[list[list[float]]]]

class ScoringResult(TypedDict):
    tokens: list[str]
    tokens_score: list[float]

# ----------------------------------------------------------------------
# Options → real dataclass with defaults
# ----------------------------------------------------------------------
@dataclass
class TranslationOptions:
    beam_size: int = 2
    patience: float = 1.0
    num_hypotheses: int = 1
    length_penalty: float = 1.0
    coverage_penalty: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    prefix_bias_beta: float = 0.0
    max_input_length: int = 1024
    max_decoding_length: int = 256
    min_decoding_length: int = 1
    sampling_topk: int = 1
    sampling_topp: float = 1.0
    sampling_temperature: float = 1.0
    return_scores: bool = False
    return_attention: bool = False
    return_alternatives: bool = False
    min_alternative_expansion_prob: float = 0.0
    return_logits_vocab: bool = False
    disable_unk: bool = False
    suppress_sequences: Sequence[Sequence[str]] | None = None
    end_token: str | Sequence[str] | None = None
    return_end_token: bool = False
    use_vmap: bool = False
    replace_unknowns: bool = False
    callback: Callable[[str], bool] | None = None

    # batch-related options
    max_batch_size: int = 0
    batch_type: BatchType = "examples"
    asynchronous: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return only non-default values for ctranslate2."""
        defaults = {k: v.default for k, v in self.__dataclass_fields__.items()}
        return {k: v for k, v in self.__dict__.items() if v != defaults.get(k)}

# ----------------------------------------------------------------------
# Enhanced ctranslate2.Translator with instance-level caching and logging
# ----------------------------------------------------------------------
class Translator(ctranslate2.Translator):
    """Enhanced ctranslate2.Translator with instance caching per (model_path, device).

    This ensures only one instance is created and reused for the same configuration.
    Thread-safe and supports multiple distinct models/configs concurrently.
    """

    _cache: dict[tuple[str, str], "Translator"] = {}
    _cache_lock = RLock()
    _tokenizer_cache: dict[str, Any] = {}  # tokenizer_name -> tokenizer

    def __new__(cls, model_path: str, device: Device = "cpu", **kwargs) -> "Translator":
        key = (model_path, device)
        with cls._cache_lock:
            if key not in cls._cache:
                logger.info(f"Creating new Translator instance for model_path='{model_path}' on device='{device}'")
                instance = super(ctranslate2.Translator, cls).__new__(cls)
                cls._cache[key] = instance
            else:
                logger.debug(f"Reusing cached Translator instance for model_path='{model_path}' on device='{device}'")
            return cls._cache[key]

    def __init__(self, model_path: str, device: Device = "cpu", **kwargs):
        key = (model_path, device)
        with self._cache_lock:
            if hasattr(self, "_initialized"):
                return  # Already initialized (cached instance)
            logger.info(f"Loading model from '{model_path}' on device '{device}'")
            super().__init__(model_path, device=device, **kwargs)
            self._initialized = True  # Mark as initialized

    @classmethod
    def get_tokenizer(cls, tokenizer_name: str) -> Any:
        """Cached tokenizer loading – separate from Translator cache."""
        with cls._cache_lock:
            if tokenizer_name not in cls._tokenizer_cache:
                logger.info(f"Loading new tokenizer: '{tokenizer_name}'")
                from transformers import AutoTokenizer
                cls._tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
            else:
                logger.debug(f"Reusing cached tokenizer: '{tokenizer_name}'")
            return cls._tokenizer_cache[tokenizer_name]

    def translate_batch(  # type: ignore[override]
        self,
        source: Sequence[Sequence[str]],
        target_prefix: Sequence[Sequence[str]] | None = None,
        *,
        options: TranslationOptions | None = None,
        **kwargs: Any,
    ) -> list[TranslationResult]:
        opts = (options.as_dict() if options else {}) | kwargs
        print(f"opts:\n{json.dumps(opts, indent=2)}")
        return super().translate_batch(source, target_prefix=target_prefix, **opts)