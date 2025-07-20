from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-14B-Instruct-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)