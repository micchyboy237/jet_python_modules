# run_visit_webpage_tool.py
import shutil
from pathlib import Path

from jet.libs.smolagents.tools.visit_webpage_tool import (
    VisitWebpageTool,  # ← adjust import path if needed
)
from smolagents import LogLevel, OpenAIModel, ToolCallingAgent

OUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = OpenAIModel(
    model_id="local-model",
    api_base="http://shawn-pc.local:8080/v1",  # ← change to your local LLM endpoint
    api_key="not-needed",
    temperature=0.6,
    max_tokens=4096,
)

# Create tool instance with logging enabled
visit_page = VisitWebpageTool(
    max_output_length=28000,
    default_k_final=8,
    verbose=True,
    logs_dir=OUT_DIR / "visit_webpage_logs",
)

agent = ToolCallingAgent(
    tools=[visit_page],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,  # or LogLevel.INFO
)

if __name__ == "__main__":
    # Try different URLs and see both full_raw=True and default (smart excerpts) behavior
    question = """
    Visit https://en.wikipedia.org/wiki/Quezon_City and tell me:
    1. When was it founded?
    2. What is its current population (latest estimate)?
    3. What are the most important landmarks or districts?
    """

    # Alternative test questions:
    # question = "Go to https://ph.investing.com/indices/pse and tell me the current value of PSEi and the change today."
    # question = "Visit https://www.gmanetwork.com/news/ with full_raw=true and summarize the top 5 headlines right now."

    print(f"\nRunning query:\n{question}\n")
    result = agent.run(question)
    print("\n" + "=" * 80)
    print("Final agent answer:")
    print(result)
    print("=" * 80 + "\n")

    print(f"Logs saved in: {OUT_DIR / 'visit_webpage_logs'}")
    print("Look for folders like call_0001/ containing:")
    print("  • request.json")
    print("  • response.json")
    print(
        "  • full_results.md  ← full content (excerpts or raw markdown) returned to the agent"
    )
