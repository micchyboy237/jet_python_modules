from pydantic import BaseModel
from jet.adapters.swarms.ollama_function_caller import OllamaFunctionCaller


class TestModel(BaseModel):
    name: str
    age: int


model = OllamaFunctionCaller(
    system_prompt="You are a helpful assistant that returns structured data about people.",
    base_model=TestModel,
    temperature=0.7,
    max_tokens=1000,
)

response = model.run("Tell me about a person named Alice who is 30 years old")
print(response)
