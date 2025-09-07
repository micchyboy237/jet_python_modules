import asyncio

from browser_use import Agent
from dotenv import load_dotenv
from jet.llm.mlx.adapters.mlx_langchain_llm_adapter import ChatMLX

from swarms import ConcurrentWorkflow

load_dotenv()


class BrowserAgent:
    def __init__(self, agent_name: str = "BrowserAgent"):
        self.agent_name = agent_name

    async def browser_agent_test(self, task: str):
        agent = Agent(
            task=task,
            llm=ChatMLX(model="llama-3.2-3b-instruct-4bit"),
        )
        result = await agent.run()
        return result

    def run(self, task: str):
        return asyncio.run(self.browser_agent_test(task))


swarm = ConcurrentWorkflow(
    agents=[BrowserAgent() for _ in range(10)],
)

swarm.run(
    """
    Go to coinpost.jp and find the latest news about the crypto market.
    """
)
