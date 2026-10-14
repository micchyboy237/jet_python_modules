from dotenv import load_dotenv
from jet.adapters.browser_use.factory import create_agent, get_browser_use_llm

load_dotenv()

llm = get_browser_use_llm()
agent = create_agent(
    task="Find the number of stars of the following repos: browser-use, playwright, stagehand, react, nextjs",
    llm=llm,
)
agent.run_sync()
