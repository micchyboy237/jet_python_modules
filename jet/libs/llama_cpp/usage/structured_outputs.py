from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="sk-1234",
)


class FriendInfo(BaseModel):
    name: str
    age: int
    is_available: bool


class FriendList(BaseModel):
    friends: list[FriendInfo]


stream = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": "I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format",
        }
    ],
    response_format={"type": "json_object", "schema": FriendList.model_json_schema()},
    max_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    presence_penalty=1.5,
    stream_options={"include_usage": True},
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    },
    stream=True,
)

response = ""
for part in stream:
    if part.choices and part.choices[0].delta:
        delta = part.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            print(delta.reasoning_content, end="", flush=True)
        elif hasattr(delta, "content") and delta.content:
            response += delta.content
            print(delta.content, end="", flush=True)

    usage = getattr(part, "usage", None)
    if usage is not None:
        print(
            f"\n\nUsage Info:"
            f"\nPrompt tokens: {usage.prompt_tokens}"
            f"\nCompletion tokens: {usage.completion_tokens}"
            f"\nTotal tokens: {usage.total_tokens}"
        )

friends_response = FriendList.model_validate_json(response)
print(friends_response)
