---
license: apache-2.0
tags:
- mlx
---

# mlx-community/Mistral-7B-Instruct-v0.3-4bit

The Model [mlx-community/Mistral-7B-Instruct-v0.3-4bit](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit) was converted to MLX format from [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) using mlx-lm version **0.14.3**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```
