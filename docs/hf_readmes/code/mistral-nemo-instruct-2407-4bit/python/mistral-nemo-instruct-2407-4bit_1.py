from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-Nemo-Instruct-2407-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)