from pydantic import BaseModel
from openai import OpenAI

class FriendInfo(BaseModel):
    name: str
    age: int
    is_available: bool

class FriendList(BaseModel):
    friends: list[FriendInfo]

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

response = client.chat.completions.create(
    model="qwen3-instruct-2507:4b",
    messages=[
        {
            'role': 'user',
            'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'
        }
    ],
    response_format={'type': 'json_object', 'schema': FriendList.model_json_schema()},
    temperature=0,
)

friends_response = FriendList.model_validate_json(response.choices[0].message.content)
print(friends_response)