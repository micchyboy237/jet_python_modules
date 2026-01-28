# multi_agent_web_browser_local_llamacpp.py

import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException

# ─── 1. Install dependencies ──────────────────────────────────────
# Run once:
# pip install 'smolagents[toolkit,litellm]' --upgrade -q
# pip install markdownify requests rich tqdm   # (rich & tqdm usually come via smolagents[toolkit])

# ─── 2. Import smolagents components ───────────────────────────────
from smolagents import (
    CodeAgent,
    OpenAIModel,
    ToolCallingAgent,
    LiteLLMModel,  # ← key change
    WebSearchTool,
    tool,
    FinalAnswerTool,
)

from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool


# ─── 3. Define custom VisitWebpageTool (same as original) ──────────
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        md = markdownify(response.text).strip()
        md = re.sub(r"\n{3,}", "\n\n", md)  # clean up excessive newlines
        return md[:12000]  # safety truncate (avoid token blowup)

    except RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ─── 4. Create local llama.cpp model client via OpenAIModel ────────────
# Assuming llama.cpp server is running on http://shawn-pc.local:8080/v1

model = OpenAIModel(
    model_id="qwen3-4b-instruct-2507",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="local-model",
    temperature=0.65,
    max_tokens=4096,
    top_p=0.95,
    # These are critical for reliable tool calling with local models
    extra_body={
        # "stop": None,
        # "frequency_penalty": 0.0,
        # "presence_penalty": 0.0,
    },
)

# Optional: quick smoke test
if False:  # change to True to test connectivity
    print(model.generate([{"role": "user", "content": "Say llama"}]))


# ─── 5. Create the web-search sub-agent ────────────────────────────
web_agent = ToolCallingAgent(
    tools=[SearXNGSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Sub-agent for running web searches and reading specific web pages.",
    add_base_tools=True,  # ← gives final_answer etc. — usually good
)

# ─── 6. Create the manager (planning) agent ────────────────────────
manager_agent = CodeAgent(
    tools=[FinalAnswerTool()],  # minimal completion tool
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],  # keep if needed
    add_base_tools=True,  # ← very helpful for manager to finish properly
    max_steps=15,  # can tune as needed
)

# ─── 7. Run the multi-agent system ─────────────────────────────────
if __name__ == "__main__":
    question = (
        "If LLM training continues to scale up at the current rhythm until 2030, "
        "what would be the electric power in GW required to power the biggest "
        "training runs by 2030? What would that correspond to, compared to some "
        "countries? Please provide a source for any numbers used."
    )

    print("\n" + "=" * 70)
    print("Running multi-agent system with local llama.cpp …")
    print("=" * 70 + "\n")

    try:
        answer = manager_agent.run(question)
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70 + "\n")
        print(answer)
    except Exception as e:
        print(f"\nError during agent execution:\n{str(e)}")
