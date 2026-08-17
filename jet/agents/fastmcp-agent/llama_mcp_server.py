"""
FastMCP server that exposes a local llama.cpp `llama-server` instance as MCP tools.

Prerequisites:
    pip install -r requirements.txt

Run llama-server first (on your Windows box, GTX 1660), e.g.:
    llama-server -hf unsloth/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 8080 --n-gpu-layers 28

Then run this script:
    python llama_mcp_server.py

Register it with your MCP client (Claude Desktop / Claude Code) pointing at this script,
or connect to it with demo_client.py.

Note on streaming output: this server is normally spawned over stdio by an MCP
client, and stdout carries the MCP JSON-RPC protocol. So streamed chunks are
flushed to STDERR (not stdout) as they arrive — safe to watch in a terminal,
but never mixed into the protocol stream. The full reply is still returned as
the tool's result over MCP once generation finishes.
"""

import json
import logging
import os
import sys
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config — loaded from a .env file (if present) and/or real environment
# variables, so you don't have to edit code when the server moves (e.g.
# Windows LAN IP vs 127.0.0.1). Real environment variables always win over
# .env, so `export LLAMA_CPP_LLM_URL=...` still overrides the file.
# ---------------------------------------------------------------------------
load_dotenv()

LLM_MODEL = os.environ.get("LLAMA_CPP_LLM_MODEL", "qwen3.5-uncensored:2b")
LLAMA_SERVER_URL = os.environ.get("LLAMA_CPP_LLM_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLAMA_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Logging — every call and response is logged for traceability.
# StreamHandler defaults to stderr, which is what keeps this safe alongside
# an stdio-based MCP transport.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llama_mcp_server")

mcp = FastMCP("Local Llama.cpp Bridge")


def _build_payload(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    min_p: Optional[float],
    top_k: Optional[int],
    repeat_penalty: Optional[float],
    extra_params: Optional[dict[str, Any]],
    stream: bool,
) -> dict[str, Any]:
    """Assemble the JSON body sent to llama-server's /v1/chat/completions."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if LLM_MODEL:
        payload["model"] = LLM_MODEL

    # Only include sampling params the caller actually set, so unset ones fall
    # back to llama-server's own defaults instead of being overridden with None.
    optional_sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
    }
    for key, value in optional_sampling_params.items():
        if value is not None:
            payload[key] = value

    # llama-server / vLLM-style passthrough for chat template kwargs (e.g.
    # disabling a model's "thinking" mode). This is a plain top-level field
    # in the raw JSON body — no "extra_body" wrapper needed, that's an
    # OpenAI-SDK-only convention with no meaning to llama-server itself.
    payload.setdefault("chat_template_kwargs", {"enable_thinking": False})

    if extra_params:
        payload.update(extra_params)

    return payload


@mcp.tool
async def ask_local_llm(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    extra_params: Optional[dict[str, Any]] = None,
    stream: bool = True,
) -> str:
    """
    Send a prompt to the locally-hosted llama.cpp model and return its reply.

    Args:
        prompt: The user's question or instruction for the local model.
        system_prompt: Optional system-level instruction to steer the model's behavior.
        max_tokens: Maximum number of tokens to generate in the reply.
        temperature: Sampling temperature (higher = more random). llama-server default ~0.8.
        top_p: Nucleus sampling cutoff (0-1).
        min_p: Minimum-probability sampling cutoff (0-1), llama.cpp-specific.
        top_k: Only sample from the top K tokens by probability.
        repeat_penalty: Penalty applied to repeated tokens (1.0 = disabled).
        extra_params: Any additional llama-server sampling params not listed above
            (e.g. {"dry_multiplier": 0.5, "grammar": "..."}). Merged directly into
            the request body, so any key llama-server accepts is valid here.
        stream: If True (default), tokens are requested as a stream and printed
            to stderr as they arrive. The full reply is still returned as one
            string once generation completes, either way.

    Returns:
        The model's full text response, or an error message if the request failed.
    """
    logger.info(
        "ask_local_llm called | prompt_len=%d max_tokens=%d stream=%s",
        len(prompt),
        max_tokens,
        stream,
    )

    payload = _build_payload(
        prompt,
        system_prompt,
        max_tokens,
        temperature,
        top_p,
        min_p,
        top_k,
        repeat_penalty,
        extra_params,
        stream,
    )
    logger.info(
        "Sampling params: %s", {k: v for k, v in payload.items() if k != "messages"}
    )

    try:
        if stream:
            return await _stream_completion(payload)
        return await _single_shot_completion(payload)
    except httpx.TimeoutException:
        logger.error(
            "Request to llama-server timed out after %.0fs", REQUEST_TIMEOUT_SECONDS
        )
        return "Error: the local model took too long to respond (timeout)."
    except httpx.HTTPStatusError as exc:
        logger.error(
            "llama-server returned HTTP %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        return f"Error: llama-server returned HTTP {exc.response.status_code}."
    except httpx.RequestError as exc:
        logger.error("Could not reach llama-server at %s: %s", LLAMA_SERVER_URL, exc)
        return (
            f"Error: could not reach llama-server at {LLAMA_SERVER_URL}. Is it running?"
        )


async def _single_shot_completion(payload: dict[str, Any]) -> str:
    """Non-streaming request: wait for the full response, then return it."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    reply = data["choices"][0]["message"]["content"]
    logger.info("ask_local_llm success (non-streaming) | reply_len=%d", len(reply))
    return reply


async def _stream_completion(payload: dict[str, Any]) -> str:
    """
    Streaming request: parse llama-server's SSE chunks, print each piece of
    content to stderr as it arrives (flushed immediately, no newline, so it
    reads naturally like the model "typing"), and return the assembled reply.
    """
    reply_parts: list[str] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        async with client.stream(
            "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed SSE chunk: %r", data_str)
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    print(content, end="", flush=True, file=sys.stderr)
                    reply_parts.append(content)

    print(file=sys.stderr)  # final newline after the streamed output
    reply = "".join(reply_parts)
    logger.info("ask_local_llm success (streaming) | reply_len=%d", len(reply))
    return reply


@mcp.tool
async def check_llama_server_health() -> str:
    """Check whether the local llama-server is reachable and report basic status."""
    logger.info("check_llama_server_health called")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/health")
            response.raise_for_status()
        logger.info("llama-server is healthy")
        return f"llama-server at {LLAMA_SERVER_URL} is up and healthy."
    except httpx.RequestError as exc:
        logger.error("Health check failed: %s", exc)
        return f"llama-server at {LLAMA_SERVER_URL} is NOT reachable: {exc}"


if __name__ == "__main__":
    logger.info("Starting FastMCP bridge, target llama-server: %s", LLAMA_SERVER_URL)
    mcp.run()
