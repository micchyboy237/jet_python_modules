from pydantic import BaseModel

from ollama import Client

from jet.logger import logger
from jet.transformers.formatters import format_json

client = Client(host='http://localhost:11435')

# Define the schema for the response
class FriendInfo(BaseModel):
    name: str
    age: int
    is_available: bool


class FriendList(BaseModel):
    friends: list[FriendInfo]


# schema = {'type': 'object', 'properties': {'friends': {'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}, 'is_available': {'type': 'boolean'}}, 'required': ['name', 'age', 'is_available']}}}, 'required': ['friends']}
response_stream = client.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'}],
    # Use Pydantic to generate the schema or format=schema
    format=FriendList.model_json_schema(),
    options={'temperature': 0},  # Make responses more deterministic
    stream=True
)

response_text = ""
for chunk in response_stream:
    response_text += chunk.message.content
    logger.teal(chunk.message.content, flush=True)

# Use Pydantic to validate the response
friends_response = FriendList.model_validate_json(response_text)
logger.gray("\n\nStructured Response:")
logger.success(format_json(friends_response))
