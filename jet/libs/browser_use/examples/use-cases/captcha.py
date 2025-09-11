"""
Goal: Automates CAPTCHA solving on a demo website.


Simple try of the agent.
@dev You need to add OPENAI_API_KEY to your environment variables.
NOTE: captchas are hard. For this example it works. But e.g. for iframes it does not.
for this example it helps to zoom in.
"""

from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


async def main():
    llm = ChatOllama(model='llama3.2')
    agent = Agent(
        task='go to https://captcha.com/demos/features/captcha-demo.aspx and solve the captcha',
        llm=llm,
    )
    await agent.run()
    input('Press Enter to exit')


if __name__ == '__main__':
    asyncio.run(main())
