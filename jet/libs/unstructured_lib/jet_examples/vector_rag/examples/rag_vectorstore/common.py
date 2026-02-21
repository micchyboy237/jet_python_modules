DOCUMENTS = {
    "asyncio_guide.md": """
# Understanding asyncio in Python (2026 edition)

asyncio is Python's standard library for writing concurrent code using async/await syntax.

## Core Concepts

1. **Coroutines** — functions defined with `async def`
2. **Tasks** — wrappers around coroutines scheduled to run concurrently
3. **Event Loop** — the core engine that runs tasks and handles I/O

```python
import asyncio

async def fetch_data(url: str) -> str:
    await asyncio.sleep(1.2)  # simulate network delay
    return f"Data from {url}"

async def main():
    urls = ["https://api.example.com/1", "https://api.example.com/2"]
    tasks = [asyncio.create_task(fetch_data(url)) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Common Patterns

### Timeout handling
```python
async def fetch_with_timeout():
    try:
        data = await asyncio.wait_for(fetch_data("slow"), timeout=1.0)
    except asyncio.TimeoutError:
        print("Request timed out")
```

### Sequential vs concurrent
Sequential:
```python
for url in urls:
    result = await fetch_data(url)
```

Concurrent:
```python
results = await asyncio.gather(*(fetch_data(u) for u in urls))
```

## Advanced: Queues & Producers/Consumers

```python
async def producer(queue: asyncio.Queue):
    for i in range(10):
        await queue.put(f"item-{i}")
        await asyncio.sleep(0.3)

async def consumer(queue: asyncio.Queue, name: str):
    while True:
        item = await queue.get()
        print(f"{name} got {item}")
        queue.task_done()
```

## Error Propagation

Exceptions raised in tasks are propagated through gather():
```python
async def buggy():
    raise ValueError("Oops")

async def main():
    try:
        await asyncio.gather(buggy(), return_exceptions=False)
    except ValueError as e:
        print(e)
```

(continued with ~800 more words of explanations, examples, best practices, common pitfalls, comparison with threading, trio, anyio, etc.)
""".strip(),
    "ai_news_2026.md": """
# AI News – February 2026 Roundup

## Grok-4 Released by xAI

On February 10, 2026, xAI announced Grok-4 with:
- 480B active parameters (MoE architecture)
- Native tool use & long context (256k tokens)
- Improved reasoning on math & code benchmarks
- First model to break 92% on GPQA Diamond

## OpenAI o4-mini becomes default for ChatGPT free tier

Faster, cheaper successor to o1-mini — now used for all non-Plus users.

## Anthropic Claude 4 family delayed to Q2 2026

Rumors point to internal safety testing bottlenecks.

## European AI Act enforcement begins March 2026

High-risk systems must comply with transparency & human oversight rules.

(continued ~500 words...)
""".strip(),
    "company_policy.md": """
# Refund & Support Policy – ExampleCorp 2026

1. **Refund Window**
   - Digital products: 14 days from purchase
   - Hardware: 30 days with original packaging
   - No refund after activation for software licenses

2. **Support Channels**
   - Email: support@examplecorp.com (response < 24h)
   - Live chat: Mon–Fri 9–18 UTC
   - Phone (premium only): +1-800-555-0123

3. **Warranty**
   - 1 year standard
   - Extended warranty available at checkout

Ticket priority levels:
- P0 – service down completely
- P1 – critical feature broken
- P2 – affects many users
- P3 – cosmetic / minor

""".strip(),
    "sample_code.py": """
# utils/math.py – ExampleCorp internal math utilities

def calculate_risk_score(prob: float, impact: int) -> float:
    \"\"\"Risk = probability × impact × 1.2 (2026 formula)\"\"\"
    if not 0 <= prob <= 1:
        raise ValueError("Probability must be [0,1]")
    return round(prob * impact * 1.2, 2)


class ExponentialBackoff:
    def __init__(self, base: float = 0.5, factor: float = 2.0, max_delay: float = 60.0):
        self.base = base
        self.factor = factor
        self.max_delay = max_delay
        self.attempt = 0

    def next_delay(self) -> float:
        delay = min(self.base * (self.factor ** self.attempt), self.max_delay)
        self.attempt += 1
        return delay

    def reset(self):
        self.attempt = 0


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return default

# ... more utility functions ...
""".strip(),
}
