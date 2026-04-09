import argparse
import os

from jet.libs.llama_cpp.performance_tracker import (
    PerformanceMetrics,
    PerformanceTracker,
)
from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk


def log_metrics(metrics: PerformanceMetrics) -> None:
    logger.info("\n\n=== Completion Details (llama.cpp aligned) ===")

    logger.info(f"Prompt tokens     : {metrics.prompt_tokens}")
    logger.info(f"Completion tokens : {metrics.completion_tokens}")
    logger.info(f"Total tokens      : {metrics.total_tokens}")

    if metrics.ttft is not None:
        logger.info(f"TTFT              : {metrics.ttft:.3f}s")

    if metrics.prompt_eval_speed is not None:
        logger.info(
            f"Prompt eval speed : {metrics.prompt_eval_speed:.2f} tokens/s (approx)"
        )

    if metrics.decode_speed is not None:
        logger.info(f"Decode speed      : {metrics.decode_speed:.2f} tokens/s (eval)")

    logger.info(f"Total latency     : {metrics.total_latency:.3f}s")

    # Optional: keep but clearly marked as non-standard
    if metrics.end_to_end_throughput is not None:
        logger.info(
            f"End-to-end throughput : {metrics.end_to_end_throughput:.2f} tokens/s (non-standard)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream chat completion from llama.cpp OpenAI API-compatible endpoint"
    )
    parser.add_argument("prompt", type=str, help="User input prompt for the chat model")
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

    tracker = PerformanceTracker()

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
            tracker.mark_token()
            logger.teal(part.choices[0].delta.content, flush=True, end="")

        usage = getattr(part, "usage", None)
        if usage is not None:
            metrics = tracker.finalize(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            log_metrics(metrics)


if __name__ == "__main__":
    main()
