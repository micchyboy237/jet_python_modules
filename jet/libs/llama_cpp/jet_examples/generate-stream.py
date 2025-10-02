from openai import OpenAI

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
stream = client.completions.create(
    model="ggml-org/gemma-3-4b-it-GGUF",
    prompt="Why is the sky blue?",
    stream=True,
)
for part in stream:
    print(part.choices[0].text or "", end="", flush=True)