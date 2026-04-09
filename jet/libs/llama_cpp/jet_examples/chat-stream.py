import os
import time

from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)

messages = [
    {
        "role": "user",
        "content": "Write a 2 sentence short story about a curious robot.",
    },
]

start_time = time.time()

stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
    model="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M",
    messages=messages,
    stream=True,
    temperature=0.7,
    top_p=0.95,
    max_tokens=180,
    stream_options={"include_usage": True},
)

for part in stream:
    if part.choices and part.choices[0].delta.content:
        logger.teal(part.choices[0].delta.content, flush=True, end="")

    usage = getattr(part, "usage", None)
    if usage is not None:
        logger.info("\n\n=== Completion Details ===")
        logger.info(f"Prompt tokens : {usage.prompt_tokens}")
        logger.info(f"Completion tokens : {usage.completion_tokens}")
        logger.info(f"Total tokens : {usage.total_tokens}")

        # Calculate overall speed using only standard Python time + official OpenAI usage
        # (no llama.cpp-specific 'timings' field is used)
        elapsed = time.time() - start_time
        if usage.completion_tokens > 0:
            overall = usage.completion_tokens / elapsed
            logger.info(f"Overall speed : {overall:.2f} tokens/s (wall-clock)")
