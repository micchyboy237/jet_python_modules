from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(
    model_name="ollama/llama3.2",
    temperature=0.5,
    max_tokens=1000,
    stream=True,
)

out = llm.run("What is the capital of France?")

print(out)
for chunk in out:
    out = chunk["choices"][0]["delta"]
    print(type(out))
    print(out)
