# jet_python_modules/jet/libs/smolagents/jet_examples/search_tool_searxng.py
import shutil
from pathlib import Path

from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from smolagents import LogLevel, ToolCallingAgent

OUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = OpenAIModel(
    temperature=0.7,
    max_tokens=2048,
)


agent = ToolCallingAgent(
    tools=[SearXNGSearchTool(max_results=10)],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,  # ← Add this line (or LogLevel.INFO)
)

if __name__ == "__main__":
    result = agent.run("What's the weather like in Las Piñas City right now?")
    print(result)
