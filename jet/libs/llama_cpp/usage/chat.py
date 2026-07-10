import os

from jet.logger import logger
from openai import OpenAI


def main():
    messages = [
        {
            "role": "user",
            "content": "Write a 2 sentence short story",
        },
    ]
    client = OpenAI(
        base_url=os.getenv("LLAMA_CPP_LLM_URL"), api_key="sk-1234"
    )  # Dummy API key
    response = client.chat.completions.create(
        model=os.getenv("LLAMA_CPP_LLM_MODEL"),
        messages=messages,
    )
    logger.teal(response.choices[0].message.content)


if __name__ == "__main__":
    main()
