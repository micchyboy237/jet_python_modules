"""
Demo: Custom Decorator Stacking & Composition
Covers: Stacking multiple custom decorators, combining jet-telemetry with
application-specific decorators, decorator order matters, and
building reusable decorator chains.
Span Hierarchy:
📦 decorator-stacking-demo (CHAIN)
│
├── 📦 Example 1: Tool Stacking (CHAIN)
│   │
│   └── 🛠️ search_database (TOOL)
│       ├── attr: tool.name = "search_database"
│       ├── attr: perf.search_database.duration_ms = 150.2
│       ├── attr: cache.hit = False (first call) / True (second call)
│       └── attr: retry.attempts_taken = 1
│
├── 📦 Example 2: LLM Tracking (CHAIN)
│   │
│   └── 🤖 generate_with_tracking (LLM)
│       ├── attr: llm.model_name = "llama-3.2-3b-instruct"
│       ├── attr: perf.generate_with_tracking.duration_ms = 1200.5
│       └── attr: prompt.context.hash = "a1b2c3d4..."
│
├── 📦 Example 3: RAG Pipeline (CHAIN)
│   │
│   └── 📦 enhanced-rag-with-stacking (CHAIN)
│       ├── attr: perf.enhanced-rag-with-stacking.duration_ms = 2500.0
│       │
│       ├── 🛠️ search_database (TOOL) [Nested]
│       └── 🤖 generate_with_tracking (LLM) [Nested]
│
└── 📦 Example 4: Composite Decorator (CHAIN)
│
└── 🤖 smart_generate (LLM)
├── attr: llm.model_name = "llama-3.2-3b-instruct"
├── attr: cache.hit = False
└── attr: perf.smart_generate.duration_ms = 900.1
"""

import asyncio
import shutil
import time
from functools import wraps
from pathlib import Path
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
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace

# Setup Output Directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

initialize_telemetry(service_name="decorator-stacking-demo", endpoint=PHOENIX_BASE_URL)
llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="sk-local")


def log_trace_download(trace_url: str | None):
    """Helper to log trace URL and trigger JSONL export if possible."""
    if not trace_url:
        return

    print(f"🔍 View complete trace: {trace_url}")

    try:
        from jet_telemetry import export_spans_to_jsonl

        if "/redirects/traces/" in trace_url:
            trace_id = trace_url.split("/redirects/traces/")[-1]
            project_name = "decorator-stacking-demo"

            jsonl_path = export_spans_to_jsonl(
                project_name=project_name,
                trace_id=trace_id,
                output_path=OUTPUT_DIR / f"{trace_id}.jsonl",
                phoenix_base_url=PHOENIX_BASE_URL,
                wait_for_flush=True,
            )
            if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
                print(f"📥 Exported JSONL: {jsonl_path.resolve()}")
    except Exception as e:
        print(f"⚠️ Could not auto-export JSONL: {e}")


def performance_monitor(func=None, *, threshold_ms: float = 500):
    """
    Attaches performance metrics to the CURRENT active span instead of creating a new one.
    """

    def decorator(f):
        is_async = asyncio.iscoroutinefunction(f)
        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await f(*args, **kwargs)
                    elapsed_ms = (time.time() - start_time) * 1000
                    _attach_perf_metrics(elapsed_ms, threshold_ms, f.__name__)
                    return result
                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    _attach_perf_metrics(elapsed_ms, threshold_ms, f.__name__, error=e)
                    raise

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = f(*args, **kwargs)
                    elapsed_ms = (time.time() - start_time) * 1000
                    _attach_perf_metrics(elapsed_ms, threshold_ms, f.__name__)
                    return result
                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    _attach_perf_metrics(elapsed_ms, threshold_ms, f.__name__, error=e)
                    raise

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def _attach_perf_metrics(
    elapsed_ms: float, threshold_ms: float, name: str, error: Exception = None
):
    """Helper to attach metrics to the current active span."""
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute(f"perf.{name}.duration_ms", round(elapsed_ms, 2))
        span.set_attribute(f"perf.{name}.slow", elapsed_ms > threshold_ms)
        if elapsed_ms > threshold_ms:
            print(f"⚠️  Slow operation '{name}': {elapsed_ms:.0f}ms")
        if error:
            span.record_exception(error)


def prompt_tracker(func=None):
    """
    Attaches prompt hashes to the CURRENT active span.
    """

    def decorator(f):
        is_async = asyncio.iscoroutinefunction(f)
        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                span = otel_trace.get_current_span()
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
    Handles retries internally but only records final success/failure on the CURRENT span.
    """

    def decorator(f):
        is_async = asyncio.iscoroutinefunction(f)
        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, max_retries + 1):
                    try:
                        result = await f(*args, **kwargs)
                        span = otel_trace.get_current_span()
                        if span.is_recording():
                            span.set_attribute("retry.attempts_taken", attempt)
                            span.set_attribute("retry.success", True)
                        return result
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            wait_time = backoff_ms * attempt / 1000
                            print(
                                f"🔄 Retry {attempt}/{max_retries} for '{f.__name__}' after {wait_time:.1f}s"
                            )
                            await asyncio.sleep(wait_time)
                span = otel_trace.get_current_span()
                if span.is_recording():
                    span.set_attribute("retry.attempts_taken", max_retries)
                    span.set_attribute("retry.success", False)
                    span.record_exception(last_exception)
                raise last_exception

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                import time as time_module

                last_exception = None
                for attempt in range(1, max_retries + 1):
                    try:
                        result = f(*args, **kwargs)
                        span = otel_trace.get_current_span()
                        if span.is_recording():
                            span.set_attribute("retry.attempts_taken", attempt)
                            span.set_attribute("retry.success", True)
                        return result
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            wait_time = backoff_ms * attempt / 1000
                            print(
                                f"🔄 Retry {attempt}/{max_retries} for '{f.__name__}' after {wait_time:.1f}s"
                            )
                            time_module.sleep(wait_time)
                span = otel_trace.get_current_span()
                if span.is_recording():
                    span.set_attribute("retry.attempts_taken", max_retries)
                    span.set_attribute("retry.success", False)
                    span.record_exception(last_exception)
                raise last_exception

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def cache_simulator(func=None, *, ttl_seconds: int = 60):
    """
    Simulates cache behavior and attaches hit/miss stats to the CURRENT span.
    """
    _cache = {}

    def decorator(f):
        is_async = asyncio.iscoroutinefunction(f)
        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                cache_key = f"{f.__name__}:{str(args)}:{str(kwargs)}"
                span = otel_trace.get_current_span()
                if cache_key in _cache:
                    cached_time, cached_result = _cache[cache_key]
                    if time.time() - cached_time < ttl_seconds:
                        if span.is_recording():
                            span.set_attribute("cache.hit", True)
                            span.set_attribute("cache.ttl_seconds", ttl_seconds)
                        print(f"💾 Cache HIT for '{f.__name__}'")
                        return cached_result
                if span.is_recording():
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
                span = otel_trace.get_current_span()
                if cache_key in _cache:
                    cached_time, cached_result = _cache[cache_key]
                    if time.time() - cached_time < ttl_seconds:
                        if span.is_recording():
                            span.set_attribute("cache.hit", True)
                            span.set_attribute("cache.ttl_seconds", ttl_seconds)
                        print(f"💾 Cache HIT for '{f.__name__}'")
                        return cached_result
                if span.is_recording():
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


@tool(description="Simulated database lookup with full observability stack")
@performance_monitor(threshold_ms=200)
@cache_simulator(ttl_seconds=30)
@retry_with_tracking(max_retries=2, backoff_ms=50)
async def search_database(query: str) -> list[dict]:
    """
    Demonstrates stacking: All utilities now inject attributes into this single TOOL span.
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
    Demonstrates stacking: Prompt tracking and perf metrics attach to this LLM span.
    """
    full_context = context or "No additional context provided."
    response_messages = [
        {"role": "system", "content": f"Context: {full_context}"},
        *messages,
    ]
    print("\n🤖 LLM Response: ", end="", flush=True)
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
                print(delta.content, end="", flush=True)
    print("\n", flush=True)
    return "".join(collected_content)


@chain(name="enhanced-rag-with-stacking")
@performance_monitor(threshold_ms=2000)
async def enhanced_rag_pipeline(query: str) -> dict:
    """
    Demonstrates stacking at the chain level.
    """
    print(f"\n🔍 Processing query: {query}")
    results = await search_database(query)
    context = "\n".join([r["title"] for r in results])
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


def fully_observed_llm(
    func=None,
    *,
    model_name: str = LLM_MODEL,
    perf_threshold_ms: float = 1000,
    cache_ttl: int = 120,
):
    """
    Composite decorator that bundles common LLM observability patterns.
    """

    def decorator(f):
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
    """
    messages = [{"role": "user", "content": prompt}]
    print("\n🤖 Smart Generate: ", end="", flush=True)
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
                print(delta.content, end="", flush=True)
    print("\n", flush=True)
    return "".join(collected_content)


@chain(name="Example 1: Tool Stacking")
async def run_example_1():
    print("\n" + "=" * 80)
    print("📦 Example 1: Tool with retry, cache, performance monitoring")
    print("=" * 80)
    result1 = await search_database("AI safety principles")
    print(f"✅ Results: {len(result1)} items found")
    print("\nCalling again to test cache...")
    result1_cached = await search_database("AI safety principles")
    print(f"✅ Cached results: {len(result1_cached)} items")


@chain(name="Example 2: LLM Tracking")
async def run_example_2():
    print("\n" + "=" * 80)
    print("🤖 Example 2: LLM with prompt tracking & performance monitoring")
    print("=" * 80)
    answer2 = await generate_with_tracking(
        messages=[{"role": "user", "content": "Explain quantum computing briefly"}],
        context="Quantum computing uses qubits instead of classical bits.",
    )
    print(f"✅ Answer generated ({len(answer2)} chars)")


@chain(name="Example 3: RAG Pipeline")
async def run_example_3():
    print("\n" + "=" * 80)
    print("🔗 Example 3: Enhanced RAG pipeline with stacked decorators")
    print("=" * 80)
    result3 = await enhanced_rag_pipeline("What is machine learning?")
    print(f"✅ Pipeline complete")
    print(f"   Sources: {result3['sources']}")
    print(f"   Answer length: {len(result3['answer'])} chars")
    if url := result3.get("trace_url"):
        log_trace_download(url)


@chain(name="Example 4: Composite Decorator")
async def run_example_4():
    print("\n" + "=" * 80)
    print("🎁 Example 4: Composite decorator (fully_observed_llm)")
    print("=" * 80)
    answer4 = await smart_generate("What are neural networks?")
    print(f"✅ Smart generate complete ({len(answer4)} chars)")
    print("\nCalling again to test cache...")
    answer4_cached = await smart_generate("What are neural networks?")
    print(f"✅ Cached answer ({len(answer4_cached)} chars)")


@chain(name="decorator-stacking-demo")
async def run_demo():
    """
    Root chain to unify all examples under a single trace ID.
    Manually sets meaningful input/output to avoid empty '{}' values.
    """
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute(SpanAttributes.INPUT_VALUE, "Demo Execution Start")
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
    print("=" * 80)
    print("🎯 Demo: Custom Decorator Stacking & Composition")
    print("=" * 80)
    await run_example_1()
    await run_example_2()
    await run_example_3()
    await run_example_4()
    print("\n" + "=" * 80)
    print("✨ Demo complete! Check Phoenix UI for detailed traces.")
    print("=" * 80)
    if span.is_recording():
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, "Demo Completed Successfully")
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")

    log_trace_download(get_trace_url(PHOENIX_BASE_URL))


if __name__ == "__main__":
    asyncio.run(run_demo())
