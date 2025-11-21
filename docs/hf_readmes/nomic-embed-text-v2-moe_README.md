---
base_model:
- nomic-ai/nomic-embed-text-v2-moe-unsupervised
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
license: apache-2.0
language:
- en
- es
- fr
- de
- it
- pt
- pl
- nl
- tr
- ja
- vi
- ru
- id
- ar
- cs
- ro
- sv
- el
- uk
- zh
- hu
- da
- 'no'
- hi
- fi
- bg
- ko
- sk
- th
- he
- ca
- lt
- fa
- ms
- sl
- lv
- mr
- bn
- sq
- cy
- be
- ml
- kn
- mk
- ur
- fy
- te
- eu
- sw
- so
- sd
- uz
- co
- hr
- gu
- ce
- eo
- jv
- la
- zu
- mn
- si
- ga
- ky
- tg
- my
- km
- mg
- pa
- sn
- ha
- ht
- su
- gd
- ny
- ps
- ku
- am
- ig
- lo
- mi
- nn
- sm
- yi
- st
- tl
- xh
- yo
- af
- ta
- tn
- ug
- az
- ba
- bs
- dv
- et
- gl
- gn
- gv
- hy
---

# nomic-embed-text-v2-moe: Multilingual Mixture of Experts Text Embeddings

[Blog](https://www.nomic.ai/blog/posts/nomic-embed-text-v2) | [Technical Report](https://huggingface.co/papers/2502.07972) | [AWS SageMaker](https://aws.amazon.com/marketplace/seller-profile?id=seller-tpqidcj54zawi) | [Atlas Embedding and Unstructured Data Analytics Platform](https://atlas.nomic.ai)

This model was presented in the paper [Training Sparse Mixture Of Experts Text Embedding Models](https://huggingface.co/papers/2502.07972).

## Model Overview
`nomic-embed-text-v2-moe` is a SoTA multilingual MoE text embedding model that excels at multilingual retrieval:

- **High Performance**: SoTA Multilingual performance compared to ~300M parameter models, competitive with models 2x in size
- **Multilinguality**: Supports ~100 languages and trained on over 1.6B pairs
- **Flexible Embedding Dimension**: Trained with [Matryoshka Embeddings](https://arxiv.org/abs/2205.13147) with 3x reductions in storage cost with minimal performance degradations
- **Fully Open-Source**: Model weights, [code](https://github.com/nomic-ai/contrastors), and training data (see code repo) released


| Model | Params (M) | Emb Dim | BEIR | MIRACL | Pretrain Data | Finetune Data | Code |
|-------|------------|----------|------|---------|---------------|---------------|------|
| **Nomic Embed v2** | 305 | 768 | 52.86 | **65.80** | ✅ | ✅ | ✅ |
| mE5 Base | 278 | 768 | 48.88 | 62.30 | ❌   | ❌   | ❌   |
| mGTE Base | 305 | 768 | 51.10 | 63.40 | ❌ | ❌ | ❌ |
| Arctic Embed v2 Base | 305 | 768 | **55.40** | 59.90 | ❌ | ❌ | ❌ |
|   |
| BGE M3 | 568 | 1024 | 48.80 | **69.20** | ❌ | ✅ | ❌ |
| Arctic Embed v2 Large | 568 | 1024 | **55.65** | 66.00 | ❌ | ❌ | ❌ |
| mE5 Large | 560 | 1024 | 51.40 | 66.50 | ❌ | ❌ | ❌ |

## Model Architecture
- **Total Parameters**: 475M
- **Active Parameters During Inference**: 305M
- **Architecture Type**: Mixture of Experts (MoE)
- **MoE Configuration**: 8 experts with top-2 routing
- **Embedding Dimensions**: Supports flexible dimension from 768 to 256 through Matryoshka representation learning
- **Maximum Sequence Length**: 512 tokens
- **Languages**: Supports dozens of languages (see Performance section)

## Paper Abstract

Transformer-based text embedding models have improved their performance on benchmarks like MIRACL and BEIR by increasing their parameter counts. However, this scaling approach introduces significant deployment challenges, including increased inference latency and memory usage. These challenges are particularly severe in retrieval-augmented generation (RAG) applications, where large models' increased memory requirements constrain dataset ingestion capacity, and their higher latency directly impacts query-time performance. While causal language models have addressed similar efficiency challenges using Mixture of Experts (MoE) architectures, this approach hasn't been successfully adapted to the general text embedding setting. In this paper, we introduce Nomic Embed v2, the first general purpose MoE text embedding model. Our model outperforms models in the same parameter class on both monolingual and multilingual benchmarks while also maintaining competitive performance with models twice its size. We open-source all code, models, and evaluation data to ensure full reproducibility of our training pipeline at https://github.com/nomic-ai/contrastors.



## Usage Guide

### Installation

The model can be used through SentenceTransformers and Transformers.

For best performance on GPU, please install

```bash
pip install torch transformers einops git+https://github.com/nomic-ai/megablocks.git
```

> [!IMPORTANT]
> **Important!**
> The text prompt *must* include a *task instruction prefix*, instructing the model which task is being performed. 

Please use `search_query: ` before your queries/questions, and `search_document: ` before your documents.

### Transformers

If using Transformers, **make sure to prepend the task instruction prefix**.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

sentences = ['search_document: Hello!', 'search_document: ¡Hola!']

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
model.eval()
with torch.no_grad():
    model_output = model(**encoded_input)
embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings.shape)
# torch.Size([2, 768])

similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
print(similarity)
# tensor(0.9118)
```

For truncation, you can trucate before applying normalization
```diff
+ embeddings = embeddings[:, :matryoshka_dim]
embeddings = F.normalize(embeddings, p=2, dim=1)
```

### SentenceTransformers

With SentenceTransformers, you can specify the `prompt_name` as either `"query"` or `"passage"`, and the task instruction will be included automatically.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
sentences = ["Hello!", "¡Hola!"]
embeddings = model.encode(sentences, prompt_name="passage")
print(embeddings.shape)
# (2, 768)

similarity = model.similarity(embeddings[0], embeddings[1])
print(similarity)
# tensor([[0.9118]])
```

For truncation/Matryoshka embeddings, you can specify `truncate_dim` and use the model similarly

```python
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, truncate_dim=256)
...
```

## Performance

nomic-embed-text-v2-moe performance on BEIR and MIRACL compared to other open-weights embedding models:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/607997c83a565c15675055b3/xadjrezEIM0Q1jbgmjqO7.png)

nomic-embed-text-v2-moe performance on BEIR at 768 dimension and truncated to 256 dimensions:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/607997c83a565c15675055b3/8hmhWQ_TTmlrviZFIBSxo.png)

## Best Practices
- Add appropriate prefixes to your text:
  - For queries: "search_query: "
  - For documents: "search_document: "
- Maximum input length is 512 tokens
- For optimal efficiency, consider using the 256-dimension embeddings if storage/compute is a concern

## Limitations
- Performance may vary across different languages
- Resource requirements may be higher than traditional dense models due to MoE architecture
- Must use `trust_remote_code=True` when loading the model to use our custom architecture implementation

## Training Details

![image/png](https://cdn-uploads.huggingface.co/production/uploads/607997c83a565c15675055b3/F0lyAtV8wXMBmxSbtIgL4.png)

- Trained on 1.6 billion high-quality pairs across multiple languages
- Uses consistency filtering to ensure high-quality training data
- Incorporates Matryoshka representation learning for dimension flexibility
- Training includes both weakly-supervised contrastive pretraining and supervised finetuning

For more details, please check out the [blog post](https://www.nomic.ai/blog/posts/nomic-embed-text-v2) and [technical report](https://www.arxiv.org/abs/2502.07972).



## Join the Nomic Community

- Nomic: [https://nomic.ai](https://nomic.ai)
- Discord: [https://discord.gg/myY5YDR8z8](https://discord.gg/myY5YDR8z8)
- Twitter: [https://twitter.com/nomic_ai](https://twitter.com/nomic_ai)

# Citation

If you find the model, dataset, or training code useful, please cite our work

```bibtex
@misc{nussbaum2025trainingsparsemixtureexperts,
      title={Training Sparse Mixture Of Experts Text Embedding Models}, 
      author={Zach Nussbaum and Brandon Duderstadt},
      year={2025},
      eprint={2502.07972},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07972}, 
}
```
