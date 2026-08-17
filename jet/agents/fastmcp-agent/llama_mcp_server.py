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
import time
import uuid
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DOTENV_PATH = os.path.join(_SCRIPT_DIR, ".env")
_ENV_BEFORE_LOAD = set(os.environ)
_DOTENV_LOADED = os.path.isfile(_DOTENV_PATH) and load_dotenv(_DOTENV_PATH)

_DEFAULTS = {
    "LLAMA_CPP_LLM_MODEL": "qwen3.5-uncensored:2b",
    "LLAMA_CPP_LLM_URL": "http://127.0.0.1:8080",
    "LLAMA_TIMEOUT": "120",
}


def _env(name: str) -> tuple[str, str]:
    """Return (value, source), distinguishing a real shell env var from one
    that only appeared after loading .env, from a hardcoded default."""
    if name in _ENV_BEFORE_LOAD:
        return os.environ[name], "shell env var"
    if name in os.environ:
        return os.environ[name], f".env file ({_DOTENV_PATH})"
    return _DEFAULTS[name], "default"


_llm_model_val, _llm_model_src = _env("LLAMA_CPP_LLM_MODEL")
_llama_url_val, _llama_url_src = _env("LLAMA_CPP_LLM_URL")
_timeout_val, _timeout_src = _env("LLAMA_TIMEOUT")

LLM_MODEL = _llm_model_val

# Normalize base URL: strip trailing slashes and accidental /v1 suffix
# to prevent double-path bugs like /v1/v1/chat/completions
LLAMA_SERVER_URL = _llama_url_val.rstrip("/")
if LLAMA_SERVER_URL.endswith("/v1"):
    logging.getLogger("llama_mcp_server").warning(
        "LLAMA_CPP_LLM_URL ends with /v1 — stripping to avoid double path suffix. "
        "Set base URL only (e.g. http://host:port)."
    )
    LLAMA_SERVER_URL = LLAMA_SERVER_URL[:-3].rstrip("/")

REQUEST_TIMEOUT_SECONDS = float(_timeout_val)

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
    request_id = uuid.uuid4().hex[:8]
    endpoint = f"{LLAMA_SERVER_URL}/v1/chat/completions"
    started_at = time.monotonic()

    logger.info(
        "[%s] ask_local_llm start | prompt_len=%d max_tokens=%d stream=%s target=%s",
        request_id,
        len(prompt),
        max_tokens,
        stream,
        endpoint,
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
        "[%s] Sampling params: %s",
        request_id,
        {k: v for k, v in payload.items() if k != "messages"},
    )

    try:
        if stream:
            reply = await _stream_completion(payload, request_id)
        else:
            reply = await _single_shot_completion(payload, request_id)

        elapsed = time.monotonic() - started_at
        logger.info(
            "[%s] ask_local_llm done | elapsed=%.2fs reply_len=%d",
            request_id,
            elapsed,
            len(reply),
        )
        return reply

    except httpx.TimeoutException:
        elapsed = time.monotonic() - started_at
        logger.error(
            "[%s] Timed out after %.2fs (limit=%.0fs) calling %s",
            request_id,
            elapsed,
            REQUEST_TIMEOUT_SECONDS,
            endpoint,
        )
        return "Error: the local model took too long to respond (timeout)."

    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - started_at
        logger.error(
            "[%s] HTTP %s from %s after %.2fs | body=%s",
            request_id,
            exc.response.status_code,
            endpoint,
            elapsed,
            exc.response.text,
        )
        return f"Error: llama-server returned HTTP {exc.response.status_code}."

    except httpx.RequestError as exc:
        elapsed = time.monotonic() - started_at
        logger.error(
            "[%s] Connection failed after %.2fs | target=%s error_type=%s error=%s",
            request_id,
            elapsed,
            endpoint,
            type(exc).__name__,
            exc,
        )
        return (
            f"Error: could not reach llama-server at {LLAMA_SERVER_URL}. Is it running?"
        )


async def _single_shot_completion(payload: dict[str, Any], request_id: str) -> str:
    """Non-streaming request: wait for the full response, then return it."""
    endpoint = f"{LLAMA_SERVER_URL}/v1/chat/completions"
    logger.info("[%s] Sending non-streaming request to %s", request_id, endpoint)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(endpoint, json=payload)
        logger.info(
            "[%s] Received HTTP %s from llama-server", request_id, response.status_code
        )
        response.raise_for_status()
        data = response.json()

    reply = data["choices"][0]["message"]["content"]
    usage = data.get("usage")
    if usage:
        logger.info("[%s] Token usage: %s", request_id, usage)

    logger.info(
        "[%s] Non-streaming completion parsed | reply_len=%d", request_id, len(reply)
    )
    return reply


async def _stream_completion(payload: dict[str, Any], request_id: str) -> str:
    """
    Streaming request: parse llama-server's SSE chunks, print each piece of
    content to stderr as it arrives (flushed immediately, no newline, so it
    reads naturally like the model "typing"), and return the assembled reply.
    """
    endpoint = f"{LLAMA_SERVER_URL}/v1/chat/completions"
    logger.info("[%s] Opening streaming request to %s", request_id, endpoint)

    reply_parts: list[str] = []
    chunk_count = 0
    started_at = time.monotonic()
    first_chunk_at: Optional[float] = None

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        async with client.stream("POST", endpoint, json=payload) as response:
            logger.info(
                "[%s] Stream connection opened | HTTP %s",
                request_id,
                response.status_code,
            )
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :].strip()
                if data_str == "[DONE]":
                    logger.info("[%s] Received [DONE] sentinel", request_id)
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(
                        "[%s] Skipping malformed SSE chunk: %r", request_id, data_str
                    )
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                        logger.info(
                            "[%s] First token received | time_to_first_token=%.2fs",
                            request_id,
                            first_chunk_at - started_at,
                        )
                    chunk_count += 1
                    print(content, end="", flush=True, file=sys.stderr)
                    reply_parts.append(content)

    print(file=sys.stderr)
    reply = "".join(reply_parts)
    logger.info(
        "[%s] Stream complete | chunks=%d reply_len=%d",
        request_id,
        chunk_count,
        len(reply),
    )
    return reply


@mcp.tool
async def check_llama_server_health() -> str:
    """Check whether the local llama-server is reachable and report basic status."""
    request_id = uuid.uuid4().hex[:8]
    endpoint = f"{LLAMA_SERVER_URL}/health"
    logger.info(
        "[%s] check_llama_server_health start | target=%s", request_id, endpoint
    )
    started_at = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(endpoint)
            response.raise_for_status()

        elapsed = time.monotonic() - started_at
        logger.info("[%s] llama-server healthy | elapsed=%.2fs", request_id, elapsed)
        return f"llama-server at {LLAMA_SERVER_URL} is up and healthy."

    except httpx.RequestError as exc:
        elapsed = time.monotonic() - started_at
        logger.error(
            "[%s] Health check failed after %.2fs | error_type=%s error=%s",
            request_id,
            elapsed,
            type(exc).__name__,
            exc,
        )
        return f"llama-server at {LLAMA_SERVER_URL} is NOT reachable: {exc}"


if __name__ == "__main__":
    logger.info(
        "Process | pid=%d cwd=%s script=%s python=%s",
        os.getpid(),
        os.getcwd(),
        os.path.abspath(__file__),
        sys.executable,
    )
    logger.info(
        ".env file: %s",
        f"found and loaded from {_DOTENV_PATH}"
        if _DOTENV_LOADED
        else f"not found at {_DOTENV_PATH} (using shell env vars / defaults only)",
    )
    logger.info("Config | LLAMA_CPP_LLM_MODEL=%s (%s)", LLM_MODEL, _llm_model_src)
    logger.info("Config | LLAMA_CPP_LLM_URL=%s (%s)", LLAMA_SERVER_URL, _llama_url_src)
    logger.info(
        "Config | LLAMA_TIMEOUT=%.0fs (%s)", REQUEST_TIMEOUT_SECONDS, _timeout_src
    )
    logger.info(
        "Starting FastMCP bridge | server_name='Local Llama.cpp Bridge' target=%s",
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
    )
    mcp.run()
