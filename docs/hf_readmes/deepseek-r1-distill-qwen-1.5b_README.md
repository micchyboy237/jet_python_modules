---
license: apache-2.0
datasets:
- sentence-transformers/stsb
language:
- en
base_model:
- FacebookAI/roberta-base
pipeline_tag: text-ranking
library_name: sentence-transformers
tags:
- transformers
---
# Cross-Encoder for Semantic Textual Similarity
This model was trained using [SentenceTransformers](https://sbert.net) [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) class.

## Training Data
This model was trained on the [STS benchmark dataset](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark). The model will predict a score between 0 and 1 how for the semantic similarity of two sentences. 


## Usage and Performance

Pre-trained models can be used like this:
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/stsb-roberta-base')
scores = model.predict([('Sentence 1', 'Sentence 2'), ('Sentence 3', 'Sentence 4')])
```

The model will predict scores for the pairs `('Sentence 1', 'Sentence 2')` and `('Sentence 3', 'Sentence 4')`.

You can use this model also without sentence_transformers and by just using Transformers ``AutoModel`` class