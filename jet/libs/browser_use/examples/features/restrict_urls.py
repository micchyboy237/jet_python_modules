from browser_use.browser import BrowserProfile, BrowserSession
from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


llm = ChatOllama(model='llama3.2')
task = (
    "go to google.com and search for openai.com and click on the first link then extract content and scroll down - what's there?"
)

allowed_domains = ['google.com']

browser_session = BrowserSession(
    browser_profile=BrowserProfile(
        executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        allowed_domains=allowed_domains,
        user_data_dir='~/.config/browseruse/profiles/default',
    ),
)

agent = Agent(
    task=task,
    llm=llm,
    browser_session=browser_session,
)


async def main():
    await agent.run(max_steps=25)

    input('Press Enter to close the browser...')
    await browser_session.kill()


asyncio.run(main())
