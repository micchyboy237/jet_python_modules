import os
import shutil
from swarms import Agent
from swarms_tools import exa_search

from jet.adapters.swarms.ollama_model import OllamaModel


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

model = OllamaModel(
    model_name="llama3.2",
    temperature=0.1,
    agent_name="Quantitative-Trading-Agent",
)

agent = Agent(
    name="Exa Search Agent",
    llm=model,
    tools=[exa_search],
    tool_call_summary=False,
)

agent.run("What are the latest experimental treatments for diabetes?")
