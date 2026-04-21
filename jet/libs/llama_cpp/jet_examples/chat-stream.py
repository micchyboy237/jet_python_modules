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
            f"End-to-end throughput : {metrics.end_to_end_throughput:.2f} tokens/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream chat completion from llama.cpp OpenAI API-compatible endpoint"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Write a 2 sentence short story about a curious robot.",
        help="User input prompt for the chat model (default: %(default)s)",
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

    tracker = PerformanceTracker()

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model="Qwen/Qwen3.5-2B",
        messages=messages,
        max_tokens=32768,
        temperature=1.0,
        top_p=1.0,
        presence_penalty=2.0,
        extra_body={
            "top_k": 20,
        },
        stream=True,
    )

    for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            # Check for reasoning_content first
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                tracker.mark_token()
                logger.orange(delta.reasoning_content, flush=True, end="")
            # Then check for regular content
            elif hasattr(delta, "content") and delta.content:
                tracker.mark_token()
                logger.teal(delta.content, flush=True, end="")

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
