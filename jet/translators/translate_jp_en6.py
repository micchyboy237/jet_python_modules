#!/usr/bin/env python3
"""
Robust Japanese to English translator using the translators library.

Prioritizes high-quality backends for Japanese→English while falling back
to stable alternatives if one fails (e.g., rate limits, network issues).

Uses a priority list based on current library status (as of Dec 2025):
- DeepL: Highest natural quality for Japanese (stable but slower).
- Google: Reliable and fast.
- Bing: Default, very stable.
- Papago: Good support for Asian languages.
- Caiyun: High quality for professional text (slower).

Run ts.preaccelerate_and_speedtest() once manually to cache sessions for speed.
"""

from __future__ import annotations

from typing import Literal, Sequence, Dict
import re
import unicodedata

import translators as ts
from rich.console import Console
from rich.table import Table

console = Console()

TranslatorBackend = Literal["deepl", "google", "bing", "papago", "caiyun"]

# Priority order for Japanese→English quality + stability
DEFAULT_PRIORITY: Sequence[TranslatorBackend] = (
    "deepl",   # Best natural Japanese translations
    "google",  # Reliable fallback
    "bing",    # Library default, very stable
    "papago",  # Strong Asian language support
    "caiyun",  # Professional/technical quality
)

# Known common OCR/variant fixes for Japanese text (safe, targeted)
COMMON_SUBSTITUTIONS: Dict[str, str] = {
    "架": "下",      # 水面架 → 水面下 (common OCR error)
    "知列": "激し",   # 知列な → 激しい
    # Add more safe pairs here if new patterns emerge
}

def preprocess_japanese_text(text: str, *, aggressive: bool = False) -> str:
    """
    Safe preprocessing for Japanese text to fix common input issues.

    Parameters
    ----------
    text: str
        Raw input text.
    aggressive: bool
        If True, apply targeted character substitutions for known OCR errors.
        Default False to avoid unintended changes on clean text.

    Returns
    -------
    str
        Normalized and optionally corrected text.
    """
    # Always-safe normalizations
    text = unicodedata.normalize("NFKC", text)  # Compatibility decomposition (fullwidth ↔ halfwidth, etc.)
    text = re.sub(r"\s+", " ", text)            # Collapse any whitespace sequences
    text = text.strip()

    if aggressive:
        for wrong, correct in COMMON_SUBSTITUTIONS.items():
            text = text.replace(wrong, correct)

    return text


def translate_jp_to_en(
    text: str,
    *,
    from_language: str = "ja",
    to_language: str = "en",
    priority: Sequence[TranslatorBackend] | None = None,
    preprocess: bool = False,
) -> str:
    """
    Translate Japanese text to English with automatic fallback on failure.

    Parameters
    ----------
    text: str
        The Japanese source text to translate.
    from_language: str
        Source language code (default 'ja').
    to_language: str
        Target language code (default 'en').
    priority: Sequence[TranslatorBackend] | None
        Optional custom priority list of backends. Uses DEFAULT_PRIORITY if None.
    preprocess: bool
        If True, run safe (and aggressive if needed) preprocessing on the input text.

    Returns
    -------
    str
        The translated English text.

    Raises
    ------
    RuntimeError
        If all backends fail.
    """
    processed_text = preprocess_japanese_text(text, aggressive=preprocess) if preprocess else text

    backends = priority or DEFAULT_PRIORITY

    last_exception: Exception | None = None
    for backend in backends:
        try:
            translated = ts.translate_text(
                processed_text,
                translator=backend,
                from_language=from_language,
                to_language=to_language,
            )
            console.print(f"[green]✓ Success with '{backend}'[/]")
            return translated
        except Exception as e:
            console.print(f"[red]✗ '{backend}' failed: {e}[/]")
            last_exception = e

    raise RuntimeError("All translation backends failed.") from last_exception


if __name__ == "__main__":
    # Example source – correct Spy x Family opening narration
    source_text = "世界各国が水面架で知列な情報戦を繰り広げる時代に、にらみ合う2つの国、東のオスタニア、西のウェスタリス、戦"

    console.print("\n[bold cyan]Translating Japanese → English using all prioritized backends[/]")
    console.print(f"[dim]Source text:[/] {source_text}\n")

    translations: Dict[str, str] = {}

    for backend in DEFAULT_PRIORITY:
        try:
            translated = ts.translate_text(
                source_text,
                translator=backend,
                from_language="ja",
                to_language="en",
            )
            translations[backend] = translated
            console.print(f"[green]✓ {backend.capitalize():<7}[/] {translated}")
        except Exception as e:
            translations[backend] = f"[red]Failed: {str(e)}[/]"
            console.print(f"[red]✗ {backend.capitalize():<7}[/] {str(e)}")

    # Display comparison table
    table = Table(title="Japanese → English Translation Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Backend", style="cyan", width=10)
    table.add_column("Translation", justify="left")

    for backend in DEFAULT_PRIORITY:
        table.add_row(backend.capitalize(), translations.get(backend, "[red]N/A[/]"))

    console.print("\n")
    console.print(table)
    console.print("\n")

    # Optional: Use the robust fallback function
    # try:
    #     best_result = translate_jp_to_en(source_text, preprocess=False)
    #     console.print("[bold green]Recommended translation (first success):[/]")
    #     console.print(best_result)
    # except RuntimeError as err:
    #     console.print(f"[bold red]All backends failed: {err}[/]")