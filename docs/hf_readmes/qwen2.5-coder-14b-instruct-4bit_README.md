---
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
language:
- en
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- code
- codeqwen
- chat
- qwen
- qwen-coder
- mlx
---

# mlx-community/Qwen2.5-Coder-14B-Instruct-4bit

The Model [mlx-community/Qwen2.5-Coder-14B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-14B-Instruct-4bit) was converted to MLX format from [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) using mlx-lm version **0.19.3**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
