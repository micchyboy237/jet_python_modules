import os
import shutil
from swarms import Agent
from jet.adapters.swarms.ollama_function_caller2 import OllamaFunctionCaller
from jet.file.utils import save_file
from jet.logger import logger
from jet.search.tools.searxng_tools import search_web
from jet.search.tools.web_tools import scrape_urls


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Initialize the agent
agent = Agent(
    agent_name="Anime-Search-Agent",
    agent_description="A specialized agent designed to search and retrieve information about anime, providing accurate and up-to-date answers using web search tools.",
    dynamic_temperature_enabled=True,
    max_loops=1,
    tools=[search_web, scrape_urls],
    dynamic_context_window=True,
    streaming_on=False,
    model_name="ollama/llama3.2",
)

out = agent.run(
    task="What are the top 10 isekai anime today?"
)

logger.success(out)
save_file(out, f"{OUTPUT_DIR}/agent_output.md")