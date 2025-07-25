---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-0.6B
tags:
- transformers
- sentence-transformers
- sentence-similarity
- feature-extraction
- mlx
library_name: mlx
pipeline_tag: text-generation
---

# mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ

This model [mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) was
converted to MLX format from [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
using mlx-lm version **0.24.1**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
