"""
Demo: Custom Decorator Stacking & Composition
Covers: Stacking multiple custom decorators, combining jet-telemetry with
        application-specific decorators, decorator order matters, and
        building reusable decorator chains.
"""

import asyncio
import time
from functools import wraps
from typing import Optional

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL, PHOENIX_BASE_URL
from jet_telemetry import (
    chain,
    get_trace_url,
    hash_prompt,
    initialize_telemetry,
    llm,
    redact,
    tool,
)
from openai import AsyncOpenAI
from opentelemetry import trace as otel_trace

initialize_telemetry(service_name="decorator-stacking-demo", endpoint=PHOENIX_BASE_URL)

llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="sk-local")


# ─── Custom Decorators ────────────────────────────────────────────────────────


def performance_monitor(func=None, *, threshold_ms: float = 500):
    """
    Tracks execution time and flags slow operations.
    Can be stacked with other decorators.
    """

    def decorator(f):
        tracer = otel_trace.get_tracer(__name__)

        is_async = asyncio.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                span_name = f"{f.__name__}_performance"

                with tracer.start_as_current_span(span_name) as span:
                    try:
                        result = await f(*args, **kwargs)
                        elapsed_ms = (time.time() - start_time) * 1000

                        span.set_attribute(
                            "performance.duration_ms", round(elapsed_ms, 2)
                        )
                        span.set_attribute(
                            "performance.slow", elapsed_ms > threshold_ms
                        )

                        if elapsed_ms > threshold_ms:
                            trace_url = get_trace_url(PHOENIX_BASE_URL)
                            span.set_attribute("performance.trace_url", trace_url or "")
                            print(
                                f"⚠️  Slow operation '{f.__name__}': {elapsed_ms:.0f}ms"
                            )

                        return result

                    except Exception as e:
                        elapsed_ms = (time.time() - start_time) * 1000
                        span.set_attribute(
                            "performance.duration_ms", round(elapsed_ms, 2)
                        )
                        span.record_exception(e)
                        raise

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                span_name = f"{f.__name__}_performance"

                with tracer.start_as_current_span(span_name) as span:
                    try:
                        result = f(*args, **kwargs)
                        elapsed_ms = (time.time() - start_time) * 1000

                        span.set_attribute(
                            "performance.duration_ms", round(elapsed_ms, 2)
                        )
                        span.set_attribute(
                            "performance.slow", elapsed_ms > threshold_ms
                        )

                        return result

                    except Exception as e:
                        elapsed_ms = (time.time() - start_time) * 1000
                        span.set_attribute(
                            "performance.duration_ms", round(elapsed_ms, 2)
                        )
                        span.record_exception(e)
                        raise

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def prompt_tracker(func=None):
    """
    Hashes and tracks string arguments that look like prompts.
    Useful for prompt versioning and deduplication.
    """

    def decorator(f):
        is_async = asyncio.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                # Get current span instead of creating new one
                span = otel_trace.get_current_span()

                # Track prompt hashes from kwargs
                for key, value in kwargs.items():
                    if isinstance(value, str) and len(value) > 20:
                        prompt_hash = hash_prompt(value)
                        preview = redact(value[:80])

                        if span.is_recording():
                            span.set_attribute(f"prompt.{key}.hash", prompt_hash)
                            span.set_attribute(f"prompt.{key}.preview", preview)

                return await f(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                span = otel_trace.get_current_span()

                for key, value in kwargs.items():
                    if isinstance(value, str) and len(value) > 20:
                        prompt_hash = hash_prompt(value)
                        preview = redact(value[:80])

                        if span.is_recording():
                            span.set_attribute(f"prompt.{key}.hash", prompt_hash)
                            span.set_attribute(f"prompt.{key}.preview", preview)

                return f(*args, **kwargs)

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def retry_with_tracking(func=None, *, max_retries: int = 3, backoff_ms: int = 100):
    """
    Adds retry logic with span tracking for each attempt.
    """

    def decorator(f):
        tracer = otel_trace.get_tracer(__name__)

        is_async = asyncio.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(1, max_retries + 1):
                    with tracer.start_as_current_span(
                        f"{f.__name__}_attempt_{attempt}"
                    ) as span:
                        span.set_attribute("retry.attempt", attempt)
                        span.set_attribute("retry.max_retries", max_retries)

                        try:
                            result = await f(*args, **kwargs)
                            span.set_attribute("retry.success", True)
                            return result

                        except Exception as e:
                            last_exception = e
                            span.set_attribute("retry.success", False)
                            span.set_attribute("retry.error", str(e))
                            span.record_exception(e)

                            if attempt < max_retries:
                                wait_time = backoff_ms * attempt / 1000
                                span.set_attribute("retry.wait_seconds", wait_time)
                                print(
                                    f"🔄 Retry {attempt}/{max_retries} for '{f.__name__}' "
                                    f"after {wait_time:.1f}s"
                                )
                                await asyncio.sleep(wait_time)

                raise last_exception

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                import time as time_module

                last_exception = None

                for attempt in range(1, max_retries + 1):
                    with tracer.start_as_current_span(
                        f"{f.__name__}_attempt_{attempt}"
                    ) as span:
                        span.set_attribute("retry.attempt", attempt)
                        span.set_attribute("retry.max_retries", max_retries)

                        try:
                            result = f(*args, **kwargs)
                            span.set_attribute("retry.success", True)
                            return result

                        except Exception as e:
                            last_exception = e
                            span.set_attribute("retry.success", False)
                            span.set_attribute("retry.error", str(e))
                            span.record_exception(e)

                            if attempt < max_retries:
                                wait_time = backoff_ms * attempt / 1000
                                span.set_attribute("retry.wait_seconds", wait_time)
                                print(
                                    f"🔄 Retry {attempt}/{max_retries} for '{f.__name__}' "
                                    f"after {wait_time:.1f}s"
                                )
                                time_module.sleep(wait_time)

                raise last_exception

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def cache_simulator(func=None, *, ttl_seconds: int = 60):
    """
    Simulates cache behavior with hit/miss tracking.
    In production, replace with actual caching logic.
    """
    _cache = {}

    def decorator(f):
        tracer = otel_trace.get_tracer(__name__)

        is_async = asyncio.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                cache_key = f"{f.__name__}:{str(args)}:{str(kwargs)}"

                with tracer.start_as_current_span(f"{f.__name__}_cache_check") as span:
                    if cache_key in _cache:
                        cached_time, cached_result = _cache[cache_key]
                        if time.time() - cached_time < ttl_seconds:
                            span.set_attribute("cache.hit", True)
                            span.set_attribute("cache.ttl_seconds", ttl_seconds)
                            print(f"💾 Cache HIT for '{f.__name__}'")
                            return cached_result

                    span.set_attribute("cache.hit", False)
                    span.set_attribute("cache.ttl_seconds", ttl_seconds)
                    print(f"🔍 Cache MISS for '{f.__name__}'")

                    result = await f(*args, **kwargs)
                    _cache[cache_key] = (time.time(), result)
                    return result

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                cache_key = f"{f.__name__}:{str(args)}:{str(kwargs)}"

                with tracer.start_as_current_span(f"{f.__name__}_cache_check") as span:
                    if cache_key in _cache:
                        cached_time, cached_result = _cache[cache_key]
                        if time.time() - cached_time < ttl_seconds:
                            span.set_attribute("cache.hit", True)
                            span.set_attribute("cache.ttl_seconds", ttl_seconds)
                            print(f"💾 Cache HIT for '{f.__name__}'")
                            return cached_result

                    span.set_attribute("cache.hit", False)
                    span.set_attribute("cache.ttl_seconds", ttl_seconds)
                    print(f"🔍 Cache MISS for '{f.__name__}'")

                    result = f(*args, **kwargs)
                    _cache[cache_key] = (time.time(), result)
                    return result

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ─── Stacked Decorator Examples ───────────────────────────────────────────────


@tool(description="Simulated database lookup with full observability stack")
@performance_monitor(threshold_ms=200)
@cache_simulator(ttl_seconds=30)
@retry_with_tracking(max_retries=2, backoff_ms=50)
async def search_database(query: str) -> list[dict]:
    """
    Demonstrates stacking: tool → performance → cache → retry

    Order matters! The outermost decorator (tool) executes last.
    Execution flow: tool → performance → cache → retry → function
    """
    await asyncio.sleep(0.15)
    return [
        {"id": 1, "title": f"Result for '{query}'", "score": 0.95},
        {"id": 2, "title": f"Related to '{query}'", "score": 0.87},
    ]


@llm(model_name=LLM_MODEL)
@performance_monitor(threshold_ms=1000)
@prompt_tracker
async def generate_with_tracking(messages: list, context: Optional[str] = None) -> str:
    """
    Demonstrates stacking: llm → performance → prompt_tracker

    Tracks prompt hashes AND performance while maintaining LLM semantic conventions.
    """
    full_context = context or "No additional context provided."

    response_messages = [
        {"role": "system", "content": f"Context: {full_context}"},
        *messages,
    ]

    print("\n🤖 LLM Response: ", end="", flush=True)  # ✅ Added
    collected_content = []
    stream = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=response_messages,
        temperature=0.1,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                continue
            if delta.content:
                collected_content.append(delta.content)
                print(delta.content, end="", flush=True)  # ✅ Added

    print("\n", flush=True)  # ✅ Added
    return "".join(collected_content)


@chain(name="enhanced-rag-with-stacking")
@performance_monitor(threshold_ms=2000)
async def enhanced_rag_pipeline(query: str) -> dict:
    """
    Demonstrates stacking at the chain level.

    The entire pipeline is monitored for performance.
    """
    print(f"\n🔍 Processing query: {query}")

    # Step 1: Search with full observability stack
    results = await search_database(query)
    context = "\n".join([r["title"] for r in results])

    # Step 2: Generate with prompt tracking
    answer = await generate_with_tracking(
        messages=[{"role": "user", "content": query}],
        context=context,
    )

    trace_url = get_trace_url(PHOENIX_BASE_URL)

    return {
        "query": query,
        "answer": answer,
        "sources": len(results),
        "trace_url": trace_url,
    }


# ─── Advanced: Custom Composite Decorator ─────────────────────────────────────


def fully_observed_llm(
    func=None,
    *,
    model_name: str = LLM_MODEL,
    perf_threshold_ms: float = 1000,
    cache_ttl: int = 120,
):
    """
    Composite decorator that bundles common LLM observability patterns.

    This is a 'decorator factory' that combines multiple decorators
    into a single reusable decorator.
    """

    def decorator(f):
        # Apply decorators from inside out
        decorated = llm(model_name=model_name)(f)
        decorated = prompt_tracker(decorated)
        decorated = cache_simulator(ttl_seconds=cache_ttl)(decorated)
        decorated = performance_monitor(threshold_ms=perf_threshold_ms)(decorated)
        return decorated

    if func is not None:
        return decorator(func)
    return decorator


@fully_observed_llm(model_name=LLM_MODEL, perf_threshold_ms=800, cache_ttl=60)
async def smart_generate(prompt: str, temperature: float = 0.7) -> str:
    """
    Uses composite decorator for full observability with one line.

    Equivalent to:
    @llm(model_name=LLM_MODEL)
    @prompt_tracker
    @cache_simulator(ttl_seconds=60)
    @performance_monitor(threshold_ms=800)
    """
    messages = [{"role": "user", "content": prompt}]

    print("\n🤖 Smart Generate: ", end="", flush=True)  # ✅ Added
    collected_content = []
    stream = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                continue
            if delta.content:
                collected_content.append(delta.content)
                print(delta.content, end="", flush=True)  # ✅ Added

    print("\n", flush=True)  # ✅ Added
    return "".join(collected_content)


# ─── Main Demo ────────────────────────────────────────────────────────────────


async def main():
    print("=" * 80)
    print("🎯 Demo: Custom Decorator Stacking & Composition")
    print("=" * 80)

    # Example 1: Tool with full stack
    print("\n" + "=" * 80)
    print("📦 Example 1: Tool with retry, cache, performance monitoring")
    print("=" * 80)
    result1 = await search_database("AI safety principles")
    print(f"✅ Results: {len(result1)} items found")

    # Call again to demonstrate cache hit
    print("\nCalling again to test cache...")
    result1_cached = await search_database("AI safety principles")
    print(f"✅ Cached results: {len(result1_cached)} items")

    # Example 2: LLM with prompt tracking
    print("\n" + "=" * 80)
    print("🤖 Example 2: LLM with prompt tracking & performance monitoring")
    print("=" * 80)
    answer2 = await generate_with_tracking(
        messages=[{"role": "user", "content": "Explain quantum computing briefly"}],
        context="Quantum computing uses qubits instead of classical bits.",
    )
    print(f"✅ Answer generated ({len(answer2)} chars)")

    # Example 3: Full pipeline
    print("\n" + "=" * 80)
    print("🔗 Example 3: Enhanced RAG pipeline with stacked decorators")
    print("=" * 80)
    result3 = await enhanced_rag_pipeline("What is machine learning?")
    print(f"✅ Pipeline complete")
    print(f"   Sources: {result3['sources']}")
    print(f"   Answer length: {len(result3['answer'])} chars")
    if url := result3.get("trace_url"):
        print(f"🔍 View complete trace: {url}")

    # Example 4: Composite decorator
    print("\n" + "=" * 80)
    print("🎁 Example 4: Composite decorator (fully_observed_llm)")
    print("=" * 80)
    answer4 = await smart_generate("What are neural networks?")
    print(f"✅ Smart generate complete ({len(answer4)} chars)")

    # Call again to test cache
    print("\nCalling again to test cache...")
    answer4_cached = await smart_generate("What are neural networks?")
    print(f"✅ Cached answer ({len(answer4_cached)} chars)")

    print("\n" + "=" * 80)
    print("✨ Demo complete! Check Phoenix UI for detailed traces.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
