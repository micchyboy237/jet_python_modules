---
license: other
license_name: qwen
license_link: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
- OpenGVLab/InternVL3-1B-Instruct
base_model_relation: finetune
datasets:
- OpenGVLab/MMPR-v1.2
language:
- multilingual
tags:
- internvl
- custom_code
- mlx
---

# mlx-community/gemma-3-12b-it-qat-4bit
This model was converted to MLX format from [`google/gemma-3-12b-it-qat-q4_0-unquantized`]() using mlx-vlm version **0.1.25**.
Refer to the [original model card](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-3-12b-it-qat-4bit --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
