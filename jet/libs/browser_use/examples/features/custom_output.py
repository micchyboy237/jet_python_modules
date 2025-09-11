"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()


class Post(BaseModel):
    post_title: str
    post_url: str
    num_comments: int
    hours_since_post: int


class Posts(BaseModel):
    posts: list[Post]


async def main():
    task = 'Go to hackernews show hn and give me the first  5 posts'
    model = ChatOllama(model='llama3.2')
    agent = Agent(task=task, llm=model, output_model_schema=Posts)

    history = await agent.run()

    result = history.final_result()
    if result:
        parsed: Posts = Posts.model_validate_json(result)

        for post in parsed.posts:
            print('\n--------------------------------')
            print(f'Title:            {post.post_title}')
            print(f'URL:              {post.post_url}')
            print(f'Comments:         {post.num_comments}')
            print(f'Hours since post: {post.hours_since_post}')
    else:
        print('No result')


if __name__ == '__main__':
    asyncio.run(main())
