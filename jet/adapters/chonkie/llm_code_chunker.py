"""
Custom CodeChunker that uses a local LLM for fast, constrained language detection.
Avoids the performance bottleneck of CodeChunker(language="auto") which relies
on Magika or trial-parsing all grammars.
"""

from typing import Literal, Optional, Union

from jet.adapters.llama_cpp.tasks import answer_multiple_choice

from chonkie.chunker.code import CodeChunker
from chonkie.logger import get_logger
from chonkie.tokenizer import TokenizerProtocol

logger = get_logger(__name__)

# Cache available languages to avoid repeated imports/calls
_AVAILABLE_LANGUAGES: Optional[list[str]] = None


# In llm_code_chunker.py, replace _get_available_languages() with:
_TOP_WEB_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "c",
    "cpp",
    "csharp",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "bash",
    "shell",
    "html",
    "css",
    "scss",
    "json",
    "yaml",
    "xml",
    "markdown",
    "sql",
    "dockerfile",
    "lua",
    "r",
    "julia",
    "dart",
    "elixir",
}


def _get_available_languages() -> list[str]:
    global _AVAILABLE_LANGUAGES
    if _AVAILABLE_LANGUAGES is None:
        try:
            from tree_sitter_language_pack import downloaded_languages

            all_langs = set(downloaded_languages())
            # Prioritize common web languages; include others as fallback
            prioritized = sorted(_TOP_WEB_LANGUAGES & all_langs)
            remaining = sorted(all_langs - _TOP_WEB_LANGUAGES)
            _AVAILABLE_LANGUAGES = prioritized + remaining
            logger.info(
                f"Cached {len(_AVAILABLE_LANGUAGES)} languages "
                f"({len(prioritized)} prioritized)"
            )
        except ImportError:
            logger.error("tree_sitter_language_pack not installed")
            _AVAILABLE_LANGUAGES = []
    return _AVAILABLE_LANGUAGES


class LLMDetectedCodeChunker(CodeChunker):
    """
    CodeChunker variant that uses constrained local LLM generation
    for language detection instead of Magika or brute-force trial parsing.

    Benefits over language='auto':
    - Deterministic output via logit_bias (no hallucinated languages)
    - Single-token generation (~10-50ms on GTX 1660)
    - No Magika model loading overhead
    - No trial-parsing through 100+ grammars
    """

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        chunk_size: int = 2048,
        language: Literal["auto"] | str = "auto",
        include_nodes: bool = False,
        llm_model: Optional[str] = None,
        max_detection_chars: int = 500,
    ) -> None:
        # Always initialize parent with explicit language to prevent
        # parent's own auto-detection from running
        resolved_language = language if language != "auto" else "python"
        super().__init__(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            language=resolved_language,
            include_nodes=include_nodes,
        )
        self._llm_model = llm_model
        self._max_detection_chars = max_detection_chars
        self._use_llm_detection = language == "auto"

        if self._use_llm_detection:
            langs = _get_available_languages()
            logger.info(
                f"LLMDetectedCodeChunker initialized with {len(langs)} language choices"
            )

    def _detect_language(self, text: str) -> str:
        """
        Override parent's _detect_language to use constrained LLM selection.
        Falls back to parent's method if LLM detection fails.
        """
        if not self._use_llm_detection:
            return self.language

        languages = _get_available_languages()
        if not languages:
            logger.warning("No tree-sitter languages available, falling back to parent")
            return super()._detect_language(text)

        # Truncate to save tokens — first N chars are usually sufficient
        snippet = text[: self._max_detection_chars].strip()
        if not snippet:
            logger.warning("Empty code snippet after truncation")
            return super()._detect_language(text)

        question = (
            "Identify the programming language of this code snippet. "
            "Choose exactly one language from the provided options.\n\n"
            f"```\n{snippet}\n```"
        )

        logger.debug(
            f"LLM language detection: {len(snippet)} chars, {len(languages)} choices"
        )

        result = answer_multiple_choice(
            question=question,
            choices=languages,
            model=self._llm_model,
            max_tokens=1,
            temperature=0.0,
        )

        if result["is_valid"] and result["answer"]:
            detected = result["answer"]
            logger.info(f"LLM detected language: '{detected}'")
            return detected

        logger.warning(
            f"LLM detection failed ({result.get('error', 'invalid')}), "
            f"falling back to parent auto-detection"
        )
        return super()._detect_language(text)

    def __repr__(self) -> str:
        return (
            f"LLMDetectedCodeChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"llm_detection={self._use_llm_detection}, "
            f"model={self._llm_model})"
        )
