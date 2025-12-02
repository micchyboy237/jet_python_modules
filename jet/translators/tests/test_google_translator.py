# tests/test_translation.py
import pytest
from jet.translators.google_translator import translate_text, atranslate_text


@pytest.mark.asyncio
async def test_atranslate_text_japanese_to_english():
    # Given
    text = "お元気ですか？"
    expected_words = ["how", "you"]

    # When
    result = await atranslate_text(text, dest="en")

    # Then
    assert isinstance(result, str)
    assert any(word in result.lower() for word in expected_words)


def test_translate_text_english_to_spanish_sync():
    # Given
    text = "See you later, alligator"
    expected = "hasta luego"

    # When
    result = translate_text(text, dest="es")

    # Then
    assert expected in result.lower()


def test_translate_text_auto_detect_french():
    # Given
    text = "C'est la vie"

    # When
    result = translate_text(text, dest="ja")

    # Then
    assert "これが人生" in result or "それが人生" in result