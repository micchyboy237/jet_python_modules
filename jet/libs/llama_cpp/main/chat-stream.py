import argparse
import os
import time
from typing import Optional

from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

DEFAULT_PROMPT = "Write a 2 sentence short story about a curious robot."


def main():
    parser = argparse.ArgumentParser(
        description="Stream chat completion from llama.cpp OpenAI API-compatible endpoint"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=DEFAULT_PROMPT,
        help="User input prompt for the chat model (default: a robot story example)",
    )
    args = parser.parse_args()

    user_prompt = args.prompt

    client = OpenAI(
        base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
        api_key="sk-1234",
    )

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    logger.log("User prompt: ", user_prompt, colors=["GRAY", "DEBUG"])

    # High precision timer
    start_time = time.perf_counter()

    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    token_count = 0

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
        now = time.perf_counter()

        # Handle streamed tokens
        if part.choices and part.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = now  # TTFT capture

            last_token_time = now
            token_count += 1

            # Avoid flush=True per token (performance hit)
            logger.teal(part.choices[0].delta.content, flush=True, end="")

        # Handle usage (final chunk)
        usage = getattr(part, "usage", None)
        if usage is not None:
            end_time = time.perf_counter()

            logger.info("\n\n=== Completion Details ===")
            logger.info(f"Prompt tokens     : {usage.prompt_tokens}")
            logger.info(f"Completion tokens : {usage.completion_tokens}")
            logger.info(f"Total tokens      : {usage.total_tokens}")

            total_elapsed = end_time - start_time

            # TTFT
            if first_token_time is not None:
                ttft = first_token_time - start_time
                logger.info(f"TTFT              : {ttft:.3f}s")

            # Generation duration
            if first_token_time and last_token_time:
                generation_duration = last_token_time - first_token_time

                if usage.completion_tokens > 0 and generation_duration > 0:
                    decode_speed = usage.completion_tokens / generation_duration
                    logger.info(
                        f"Decode speed      : {decode_speed:.2f} tokens/s (generation only)"
                    )

            # End-to-end speed (your original metric)
            if usage.completion_tokens > 0:
                overall_speed = usage.completion_tokens / total_elapsed
                logger.info(
                    f"Overall speed     : {overall_speed:.2f} tokens/s (wall-clock)"
                )

            logger.info(f"Total latency     : {total_elapsed:.3f}s")


if __name__ == "__main__":
    main()
