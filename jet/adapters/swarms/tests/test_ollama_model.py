from jet.adapters.swarms.ollama_model import OllamaModel, Message


ollama = OllamaModel("llama3.1")

messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Explain the benefits of exercise."),
]

print("=== Chat ===")
chat_response = ollama.chat(messages)
print(chat_response)

print("\n=== Generate ===")
gen_response = ollama.generate("Write a short poem about the ocean.")
print(gen_response)
