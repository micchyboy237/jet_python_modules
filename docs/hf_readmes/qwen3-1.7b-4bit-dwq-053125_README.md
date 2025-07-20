---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE
pipeline_tag: text-generation
base_model: Qwen/Qwen3-1.7B
tags:
- mlx
---

# mlx-community/Qwen3-1.7B-4bit-DWQ-053125

This model [mlx-community/Qwen3-1.7B-4bit-DWQ-053125](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit-DWQ-053125) was
converted to MLX format from [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
using mlx-lm version **0.24.1**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
