"""
Getting Started Example 2: Form Filling

This example demonstrates how to:
- Navigate to a website with forms
- Fill out input fields
- Submit forms
- Handle basic form interactions

This builds on the basic search example by showing more complex interactions.
"""

from browser_use import Agent, BrowserProfile
from dotenv import load_dotenv
import asyncio
import os
import sys

from jet.adapters.browser_use.ollama.chat import ChatOllama

# Add the parent directory to the path so we can import browser_use
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


async def main():
    # Initialize the model
    llm = ChatOllama(model='llama3.2')

    # Define a form filling task
    task = """
    Go to https://httpbin.org/forms/post and fill out the contact form with:
    - Customer name: John Doe
    - Telephone: 555-123-4567
    - Email: john.doe@example.com
    - Size: Medium
    - Topping: cheese
    - Delivery time: now
    - Comments: This is a test form submission
    
    Then submit the form and tell me what response you get.
    """

    browser_profile = BrowserProfile(
        headless=True,
        # Set browser window size to 1440x900 pixels
        window_size={"width": 1440, "height": 900}
    )

    # Create and run the agent
    agent = Agent(
        task=task,
        llm=llm,
        llm_timeout=300.0,
        step_timeout=300.0,
        browser_profile=browser_profile,
    )
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
