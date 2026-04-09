import os

from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",  # Dummy key — llama.cpp ignores it
)

messages = [
    {
        "role": "user",
        "content": "Write a 2 sentence short story about a curious robot.",
    },
]

stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M",  # must match loaded model name or ignore if server uses --model-alias
    messages=messages,
    stream=True,
    # ── Most important speed / quality knobs ─────────────────────────────
    temperature=0.7,  # 0.0–1.0; lower = faster + more deterministic
    top_p=0.95,  # nucleus sampling — very common & fast combo with temp
    max_tokens=180,  # safety cap — prevents very long generations
    # ── Nice to have ──────────────────────────────────────────────────────
    stream_options={"include_usage": True},  # shows tokens/s at end
    # top_k=40,                # optional — can be faster than pure top_p on some models
    # presence_penalty=0.1,    # tiny diversity boost — usually not needed for speed
    # ── llama.cpp specific (works if your server supports extra_body) ─────
    # extra_body={"mirostat": 2, "mirostat_tau": 5.0}   # uncomment if you like Mirostat
)

for part in stream:
    if part.choices:
        delta = part.choices[0].delta
        content = delta.content or ""
        logger.teal(content, flush=True, end="")

    # Optional: show final usage stats (tokens/s)
    if hasattr(part, "usage") and part.usage:
        logger.info(f"\nUsage → {part.usage}")
