import asyncio
from googletrans import Translator

async def translate_text():
    # Translator supports async context manager in 4.0.0-rc1
    async with Translator() as translator:
        # JA → EN
        result = await translator.translate("こんにちは", src="ja", dest="en")
        print(result.text)  # Hello / Hello there / Hi (varies)

        # EN → JA
        result = await translator.translate("Good morning", src="en", dest="ja")
        print(result.text)

        # Auto-detect JA → EN
        result = await translator.translate("今日はいい天気ですね。", dest="en")
        print(result.text)

asyncio.run(translate_text())
