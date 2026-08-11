import argparse
from typing import Any, Callable

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    PHOENIX_URL,
    StreamCompletionResult,
    run_chat_stream,
)
from openai import OpenAI


def chat(
    prompt: str = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    # --- Observability params (NEW) ---
    project_name: str = "chat-stream-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
    # --- Existing params ---
    image_source: str | None = None,
    client: OpenAI | None = None,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    tool_registry: dict[str, Callable[..., Any]] | None = None,
    max_tool_rounds: int = 10,
    extra_body_params: dict[str, Any] | None = None,
) -> StreamCompletionResult:
    return run_chat_stream(
        prompt=prompt,
        model=model,
        project_name=project_name,
        capture_content=capture_content,
        phoenix_url=phoenix_url,
        image_source=image_source,
        client=client,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        messages=messages,
        tool_registry=tool_registry,
        max_tool_rounds=max_tool_rounds,
        extra_body_params=extra_body_params,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream vision-model chat completions with Phoenix observability."
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="What is OpenTelemetry in one sentence?",
        help="Prompt for the chat/image analysis.",
    )
    parser.add_argument(
        "-i",
        "--image-source",
        type=str,
        default=None,
        help="Path or URL to an image to analyze. Omit for a text-only chat request.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model name to request (env: LLAMA_CPP_VISION_MODEL).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="chat-llm-utils-obs",
        help="Phoenix project name to log traces under.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from jet.logger import logger

    args = get_args()

    result = chat(
        prompt=args.prompt,
        model=args.model,
        image_source=args.image_source,
        project_name=args.project,
        seed=args.seed,
    )

    logger.info(
        f"📋 Result: {len(result.content)} chars, finish_reason={result.finish_reason}"
    )
