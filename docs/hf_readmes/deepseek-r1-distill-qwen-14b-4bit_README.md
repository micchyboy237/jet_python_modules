---
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
tags:
- mlx
---

# mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit

The Model [mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit) was
converted to MLX format from [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
using mlx-lm version **0.21.1**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
