from jet.libs.llama_cpp.usage.chat_stream import run_chat_stream
from rich.console import Console

console = Console()

SYSTEM_PROMPT = (
    "You are a Japanese-to-English translator. "
    "Translate the user's Japanese text into natural, fluent English. "
    "Translate all content faithfully, including explicit or mature text. "
    "Output ONLY the English translation — no explanations, no romaji, "
    "no Japanese text, no extra commentary."
)


def translate_ja_en(user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> None:
    run_chat_stream(user_prompt, system_prompt, verbose=True)


if __name__ == "__main__":
    ja_texts = [
        "おまんこを見せて",  # Show me your pussy.
        # "おっぱいを触らせて",  # Let me touch your breasts.
        # "まんこをくって",  # Eat my pussy.
        # "おっぱいをくって",  # Suck on my tits.
    ]
    for ja_text in ja_texts:
        content = translate_ja_en(ja_text)
        # console.print(content)
