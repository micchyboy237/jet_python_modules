import asyncio
from typing import List

import httpx
from transformers import AutoTokenizer

# System prompt and instruction (unchanged)
system_prompt = (
    "You are a highly skilled professional Japanese-to-English translator. "
    "Translate the given Japanese text into natural, accurate, and fluent English. "
    "Preserve the original meaning, tone, and cultural nuances. "
    "Only add an explicit subject in English when it is clearly specified in the Japanese sentence. "
    "For technical terms and proper nouns, use standard English equivalents when they exist, "
    "or keep them in romanized form if no common translation is available. "
    "After translating, review your output to ensure it is grammatically correct and reads naturally. "
    "Take a deep breath and produce the best possible translation.\n\n"
)

instruct = (
    "Translate the following Japanese text to English.\n"
    "When translating, please use the following hints:\n"
    "[writing_style: casual]"
)

initial_messages = [
    {"role": "user", "content": system_prompt + instruct},
    {"role": "assistant", "content": "OK"}
]

message_list = [
    "私も、日本人の聴衆に本当に初めて話すので、少し緊張していました。ホワイトハウスを去ってから間違いなく初めてです。",
    "そして、とても優秀な通訳がいました。もし皆さんが日本で英語でスピーチをしたことがあれば、日本語で言うとずっと時間がかかることが分かると思います。",
    "そこで、私は自分が知っている一番短いジョークを話して、場を和ませようと思いました。",
    "それは私が知っている最高のジョークではありませんでしたが、一番短いジョークで、何年か前の知事選のキャンペーンから残っていたものです。",
    "それで私はジョークを話し、通訳がそのジョークを伝えました。すると聴衆は大笑いしました。",
    "人生でこれほど良い反応をもらったことはありませんでした。",
    "だからスピーチを早く終わらせて、通訳に聞きたくて仕方ありませんでした。",
    "「私のジョークをどうやって伝えたんですか？」",
    "彼はとても曖昧で、どう伝えたか教えてくれませんでした。",
    "私がしつこく聞くと、ついに頭を下げてこう言いました。",
    "「『カーター大統領が面白い話をしてくれました。みんな、笑ってください』と伝えました。」"
]

tokenizer = AutoTokenizer.from_pretrained("webbigdata/gemma-2-2b-jpn-it-translate")


async def translate_sentence(
    client: httpx.AsyncClient,
    messages: List[dict],
    sentence: str,
) -> str:
    """Send a single sentence to the local LLM and return the assistant response."""
    messages.append({"role": "user", "content": sentence})

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    payload = {
        "prompt": prompt,
        "n_predict": 1200,
    }

    response = await client.post(
        "http://shawn-pc.local:8080/completion",
        json=payload,
        timeout=60.0,  # Prevent hanging indefinitely
    )

    if response.status_code != 200:
        raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")

    content = response.json().get("content", "").strip()
    messages.append({"role": "assistant", "content": content})

    return content


async def main() -> None:
    messages = initial_messages.copy()

    async with httpx.AsyncClient() as client:
        for sentence in message_list:
            print("user: " + sentence)

            try:
                assistant_response = await translate_sentence(client, messages, sentence)
                print("assistant: " + assistant_response)
            except Exception as e:
                print(f"Error during translation: {e}")
                continue

            # Keep context window manageable: initial 2 + last 6 exchanges (12 messages total)
            if len(messages) > 12:
                messages = initial_messages + messages[-10:]  # last 5 full exchanges (10 messages)


if __name__ == "__main__":
    asyncio.run(main())