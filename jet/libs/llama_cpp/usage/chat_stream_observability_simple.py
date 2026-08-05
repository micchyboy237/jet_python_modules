import base64
import logging
import os

from openai import OpenAI
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from phoenix.otel import register

# ── 1. Logging setup (for traceability while developing) ───────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("llm_client")

# ── 2. Read env vars ─────────────────────────────────────────────────────
VISION_MODEL = os.environ["LLAMA_CPP_VISION_MODEL"]
VLLM_URL = os.environ["LLAMA_CPP_VISION_URL"]
PHOENIX_URL = os.environ["LLM_OBS_PHOENIX_URL"]

# Phoenix (and OTel in general) needs message content capture explicitly
# enabled, otherwise spans only show metadata (model, tokens, latency) —
# no actual prompt/response text. Set this BEFORE instrument() runs.
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

# ── 3. Wire up OpenTelemetry → Phoenix ──────────────────────────────────
tracer_provider = register(
    project_name="vision-chat-client",
    endpoint=f"{PHOENIX_URL}/v1/traces",
    # batch=True is fine for long-running apps; for short scripts you may
    # want batch=False so spans flush immediately before the process exits.
)
log.info("Tracer provider registered → %s/v1/traces", PHOENIX_URL)

# ── 4. Instrument the OpenAI SDK (auto-patches chat.completions.create) ──
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
log.info("OpenAI SDK instrumented")

# ── 5. Point the OpenAI client at your local llama.cpp/vLLM server ───────
client = OpenAI(
    base_url=VLLM_URL,
    api_key="not-needed",  # local server, but SDK requires a non-empty string
)


def encode_image_to_data_url(path: str) -> str:
    """Read a local image file and return it as a base64 data URL."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = path.rsplit(".", 1)[-1].lower()
    return f"data:image/{ext};base64,{b64}"


def stream_vision_chat(
    prompt: str,
    image_path: str | None = None,
    *,
    enable_thinking: bool = False,
) -> str:
    """
    Streams a chat completion from the local vision model.
    Every call is auto-traced and shipped to Phoenix by the instrumentor.
    """
    content = [{"type": "text", "text": prompt}]
    if image_path:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(image_path)},
            }
        )

    log.info("Sending request to model=%s (image=%s)", VISION_MODEL, bool(image_path))

    stream = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        stream=True,
        stream_options={"include_usage": True},  # needed for token counts in the span
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            },
        },
    )

    collected = []
    in_think_block = False
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta:
                # Print reasoning_content wrapped in <think>...</think> block,
                # but only open <think> at the start and close at the end of the contiguous reasoning_content area.
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not in_think_block:
                        print("<think>", end="", flush=True)
                        in_think_block = True
                    print(delta.reasoning_content, end="", flush=True)
                    collected.append(delta.reasoning_content)
                elif in_think_block:
                    print("</think>", end="", flush=True)
                    in_think_block = False

                # Print normal content, always outside <think>
                if hasattr(delta, "content") and delta.content:
                    print(delta.content, end="", flush=True)
                    collected.append(delta.content)
        if in_think_block:
            print("</think>", end="", flush=True)
            in_think_block = False
    except Exception:
        log.exception("Streaming failed")
        raise
    finally:
        print()  # newline after stream ends

    full_response = "".join(collected)
    log.info("Stream complete, %d chars received", len(full_response))
    return full_response


if __name__ == "__main__":
    result = stream_vision_chat(
        prompt="Describe what's happening in this image.",
        image_path=None,  # e.g. "photo.jpg" to test vision
    )
