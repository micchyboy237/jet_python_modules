from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.utils.model_utils import create_local_model
from smolagents import CodeAgent

model = create_local_model()

agent = CodeAgent(
    tools=[SearXNGSearchTool()],
    model=model,
    planning_interval=4,  # ← planning every 4 steps
    max_steps=20,
    verbosity_level=1,
)

result = agent.run(
    "Find the population trend of Tokyo from 1990 to 2025 and forecast 2030"
)
