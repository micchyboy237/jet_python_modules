# Synchronous (scripts, legacy code, Jupyter, etc.)
from .google_translator import translate_text

print(translate_text("こんにちは"))                     # Hello
print(translate_text("Good morning", dest="ja"))        # おはようございます
print(translate_text("Merci", src="fr", dest="es"))     # Gracias

# Asynchronous (FastAPI, aiohttp, etc.)
import asyncio
from .google_translator import atranslate_text

async def main():
    print(await atranslate_text("今日はいい天気ですね。", dest="en"))
    print(await atranslate_text("Hola", src="es", dest="ja"))

asyncio.run(main())