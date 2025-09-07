
from jet.adapters.swarms.mlx_model import MLXModel, Message


mlx = MLXModel("mlx-community/Llama-3.2-3B-Instruct-4bit",
               temperature=0.7, max_tokens=200)

messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Explain the benefits of exercise."),
]

print("=== Chat ===")
print(mlx.chat(messages))

print("\n=== Generate ===")
print(mlx.generate("Write a short poem about the ocean."))
