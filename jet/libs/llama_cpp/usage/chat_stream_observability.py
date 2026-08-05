import argparse
import os

# ============================================================
# OPENLLMETRY SETUP - Must run BEFORE creating the OpenAI client
# Configure via env vars, NOT px.Client (removed in Phoenix v8+)
# ============================================================
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006/v1/traces")
os.environ.setdefault("OTEL_SERVICE_NAME", "llama-cpp-chat-stream")

from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

OpenAIInstrumentor().instrument()

# ============================================================
# STANDARD OPENAI CLIENT SETUP
# ============================================================
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",  # llama.cpp doesn't validate key, but SDK requires it
)


def run_chat_stream(
    user_prompt: str,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    verbose: bool = False,
    **kwargs,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if verbose:
        if system_prompt:
            print(f"[SYSTEM] {system_prompt}")
        print(f"[USER] {user_prompt}")

    create_kwargs = dict(
        model=os.getenv("LLAMA_CPP_LLM_MODEL", "not-needed"),
        messages=messages,
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        stream_options={"include_usage": True},
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            },
        },
        stream=True,
        **kwargs,
    )

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        **create_kwargs
    )

    content = ""
    for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            # Handle reasoning/thinking content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                content += delta.reasoning_content
                if verbose:
                    print(delta.reasoning_content, end="", flush=True)

            # Handle regular assistant content
            elif hasattr(delta, "content") and delta.content:
                content += delta.content
                if verbose:
                    print(delta.content, end="", flush=True)

        # Token usage is automatically captured by OpenLLMetry
        usage = getattr(part, "usage", None)
        if usage is not None and verbose:
            print(
                f"\n[TOKENS] prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"total={usage.total_tokens}"
            )

    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream chat completion from llama.cpp with Phoenix observability"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Write a 2 sentence short story about a curious robot.",
        help="User input prompt for the chat model (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="Optional system prompt for the chat model",
    )
    args = parser.parse_args()

    run_chat_stream(args.prompt, args.system, verbose=True)
