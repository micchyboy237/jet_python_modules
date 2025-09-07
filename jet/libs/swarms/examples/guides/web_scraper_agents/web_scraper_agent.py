from swarms import Agent

from jet.adapters.swarms.tools.search.web_scraper import scrape_and_format_sync
from jet.adapters.swarms.mlx_function_caller import MLXFunctionCaller
from jet.logger import CustomLogger

import os
import shutil
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
log_dir = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
filename_no_ext = os.path.splitext(os.path.basename(__file__))[0]
log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

llm = MLXFunctionCaller(
    max_tokens=4000,
    temperature=0.7,
)

# Update the Agent initialization
agent = Agent(
    agent_name="Web Scraper Agent",
    model_name="ollama/llama3.2",
    # llm=llm,
    tools=[scrape_and_format_sync],
    streaming_on=True,
    log_directory=log_dir,
    workspace_dir=f"{OUTPUT_DIR}/agent_workspace",
    dynamic_context_window=True,
    dynamic_temperature_enabled=True,
    max_loops=1,
    system_prompt="You are a web scraper agent. You are given a URL and you need to scrape the website and return the data in a structured format. The format type should be one of: detailed, summary, minimal, or markdown. Use 'detailed' by default.",
)

out = agent.run(
    # "Scrape https://ollama.com/search and provide a full report of the latest models. The format type should be full."
    "Scrape https://myanimelist.net website, navigate through relevant pages and provide a full report of the latest and upcoming isekai and romantic comedy anime. The format type should be full."
)
print(out)
