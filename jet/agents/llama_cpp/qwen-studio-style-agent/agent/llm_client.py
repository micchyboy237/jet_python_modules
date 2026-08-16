from openai import OpenAI

from agent.config import Config


class LLMClient:
    def __init__(self):
        self.client = OpenAI(base_url=Config.LLAMA_BASE_URL, api_key="no-key-needed")
        self.model = Config.LLAMA_MODEL

    def chat(self, messages: list, tools: list | None = None) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message
