# translation_service.py
from __future__ import annotations

from typing import Optional

from googletrans import Translator
import asyncio

# Keep popular language codes – feel free to extend
LanguageCode = str  # googletrans accepts any string, Literal is too restrictive for real use


class TranslationService:
    """
    Fixed version – does NOT close the internal httpx client.
    Works reliably with both sync and async calls when the instance is shared.
    """

    def __init__(self) -> None:
        # One persistent Translator (its internal AsyncClient stays open)
        self.translator = Translator()

    async def atranslate(
        self,
        text: str,
        dest: LanguageCode = "en",
        src: Optional[LanguageCode] = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # ←←← NOTE: NO "async with" → client is never closed
        result = await self.translator.translate(
            text, src=src or "auto", dest=dest
        )

        if not result or not getattr(result, "text", None):
            raise RuntimeError("Google Translate returned no result")

        return result.text.strip()

    def translate(
        self,
        text: str,
        dest: LanguageCode = "en",
        src: Optional[LanguageCode] = None,
    ) -> str:
        return asyncio.run(self.atranslate(text, dest=dest, src=src))


# Module-level shared service (import this everywhere)
_translation_service = TranslationService()


# Public API – unchanged signatures
def translate_text(
    text: str,
    dest: LanguageCode = "en",
    src: Optional[LanguageCode] = None,
) -> str:
    return _translation_service.translate(text, dest=dest, src=src)


async def atranslate_text(
    text: str,
    dest: LanguageCode = "en",
    src: Optional[LanguageCode] = None,
) -> str:
    return await _translation_service.atranslate(text, dest=dest, src=src)