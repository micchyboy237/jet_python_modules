---
license: llama3.1
datasets:
- OpenCoder-LLM/opc-sft-stage1
- OpenCoder-LLM/opc-sft-stage2
- microsoft/orca-agentinstruct-1M-v1
- microsoft/orca-math-word-problems-200k
- NousResearch/hermes-function-calling-v1
- AI-MO/NuminaMath-CoT
- AI-MO/NuminaMath-TIR
- allenai/tulu-3-sft-mixture
- cognitivecomputations/dolphin-coder
- HuggingFaceTB/smoltalk
- cognitivecomputations/samantha-data
- m-a-p/CodeFeedback-Filtered-Instruction
- m-a-p/Code-Feedback
language:
- en
base_model: cognitivecomputations/Dolphin3.0-Llama3.1-8B
tags:
- mlx
---

# mlx-community/Dolphin3.0-Llama3.1-8B-4bit

The Model [mlx-community/Dolphin3.0-Llama3.1-8B-4bit](https://huggingface.co/mlx-community/Dolphin3.0-Llama3.1-8B-4bit) was
converted to MLX format from [cognitivecomputations/Dolphin3.0-Llama3.1-8B](https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B)
using mlx-lm version **0.20.5**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Dolphin3.0-Llama3.1-8B-4bit")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
