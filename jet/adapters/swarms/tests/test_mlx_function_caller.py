# from mlx_function_caller import MLXFunctionCaller
from pydantic import BaseModel

from jet.adapters.swarms.mlx_function_caller import MLXFunctionCaller


class TestModel(BaseModel):
    name: str
    age: int


model = MLXFunctionCaller(
    system_prompt="You are a helpful assistant that returns structured data about people.",
    base_model=TestModel,
    temperature=0.7,
    max_tokens=500,
)

response = model.run("Tell me about a person named Bob who is 42 years old")
print(response)
