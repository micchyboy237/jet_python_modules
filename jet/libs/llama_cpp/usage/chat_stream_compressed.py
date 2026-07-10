import argparse
import os

from headroom import compress
from headroom.transforms import CodeAwareCompressor
from jet.libs.llama_cpp.utils.performance_tracker import PerformanceTracker, log_metrics
from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)

compressor = CodeAwareCompressor()


def compress_messages(messages: list[dict]) -> list[dict]:
    print("=== ORIGINAL MESSAGES ===")
    print(f"Total messages: {len(messages)}")
    # Rough token estimate (not exact)
    original_tokens = sum(len(str(m).split()) * 1.3 for m in messages)  # Very rough
    print(f"Rough original token estimate: ~{int(original_tokens)}")

    # =============================================
    # Compress with Headroom (triggers CodeAwareCompressor)
    # =============================================
    print("\n=== COMPRESSING WITH HEADROOM ===")

    result = compress(
        messages,
        model="gpt-4o",  # Helps with token counting / model-specific tweaks
        target_ratio=0.5,
        protect_recent=0,
    )
    # result = compressor.compress(messages, language="python")

    print(f"Tokens before: {result.tokens_before}")
    print(f"Tokens after:  {result.tokens_after}")
    print(
        f"Tokens saved:  {result.tokens_saved} ({result.compression_ratio:.1%} compression)"
    )
    print(f"Transforms applied: {result.transforms_applied}")

    # View compressed messages
    print("\n=== COMPRESSED MESSAGES ===")
    compressed_messages = result.messages
    for msg in compressed_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:500] + (
            "..." if len(str(msg.get("content", ""))) > 500 else ""
        )
        print(f"[{role.upper()}] {content}")

    return compressed_messages


def run_chat_stream(
    user_prompt: str, system_prompt: str | None = None, verbose: bool = False
):
    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    messages = SAMPLE_MESSAGES

    if system_prompt:
        if verbose:
            logger.log("System prompt: ", system_prompt, colors=["PURPLE", "DEBUG"])
    if verbose:
        logger.log("User prompt: ", user_prompt, colors=["GRAY", "DEBUG"])

    # Compress messages to reduce token usage
    messages = compress_messages(messages)

    tracker = PerformanceTracker()

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model="Qwen/Qwen3.5-2B",
        messages=messages,
        max_tokens=1024,
        temperature=1.0,
        top_p=1.0,
        presence_penalty=2.0,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
        stream=True,
    )

    content = ""
    for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            # Check for reasoning_content first
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                content += delta.reasoning_content
                tracker.mark_token()
                if verbose:
                    logger.orange(delta.reasoning_content, flush=True, end="")
            # Then check for regular content
            elif hasattr(delta, "content") and delta.content:
                content += delta.content
                tracker.mark_token()
                if verbose:
                    logger.teal(delta.content, flush=True, end="")

        usage = getattr(part, "usage", None)
        if usage is not None:
            metrics = tracker.finalize(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            if verbose:
                log_metrics(metrics)

    return content


# =============================================
# Sample Messages: User prompt asking about code
# =============================================
SAMPLE_MESSAGES = [
    {
        "role": "system",
        "content": "You are an expert Python performance engineer. Analyze code and suggest improvements.",
    },
    {
        "role": "user",
        "content": "Explain this function from utils.py and suggest performance improvements. Focus on the main processing logic.",
    },
    {
        "role": "tool",
        "content": """# utils.py - Large data processor
import time
from typing import List, Dict, Any

def process_large_list(data: List[Dict[str, Any]]) -> Dict[str, Dict]:
    '''Process list of items with heavy computation. Handles 10k+ entries.'''
    result = {}
    start = time.time()
    
    for item in data:  # Potential bottleneck with large N
        key = item.get('id')
        if not key:
            continue
            
        # Nested processing
        value = item.get('value', 0) * 2
        for sub in item.get('subs', []):
            value += sub.get('score', 0) * 1.5
            
        # Additional heavy ops
        if 'metadata' in item:
            for k, v in item['metadata'].items():
                if isinstance(v, (int, float)):
                    value += v
        
        result[key] = {
            'processed': value,
            'processing_time': time.time() - start,
            'items_processed': len(data)
        }
    
    print(f"Total time: {time.time() - start:.2f}s")
    return result

# Unused legacy helper (should be removed)
def old_helper(x: int) -> int:
    return x ** 2 + 42

# Example usage (commented)
# data = [{'id': i, 'value': i*10, 'subs': [...]} for i in range(10000)]
# result = process_large_list(data)
""",
        "tool_call_id": "call_process_code_123",
    },
]

if __name__ == "__main__":
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
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default=None,
        help="Optional system prompt for the chat model",
    )
    args = parser.parse_args()

    user_prompt = args.prompt
    system_prompt = args.system

    run_chat_stream(user_prompt, system_prompt, verbose=True)
