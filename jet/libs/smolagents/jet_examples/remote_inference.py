# remote_inference.py

from smolagents import InferenceClientModel, CodeAgent, DuckDuckGoSearchTool
import os

model = InferenceClientModel(
    model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",   # correct large variant (~400B total)
    # or "meta-llama/Llama-4-Scout-17B-16E-Instruct" for efficiency + 10M context
    provider="fireworks-ai",                                    # ← reliable for Llama 4 Maverick
    # Alternatives: "novita", "sambanova", "cerebras", "groq" (if available)
    token=os.getenv("HF_TOKEN"),                                # use env var (safer)
    temperature=0.3,
    max_tokens=2048,
)

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=8,
)

result = agent.run(
    "Compare inference cost & performance of DeepSeek-V3 vs Llama-4 on SWE-bench Verified in January 2026"
)
print(result)