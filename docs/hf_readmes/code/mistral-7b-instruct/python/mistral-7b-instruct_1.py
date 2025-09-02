from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)