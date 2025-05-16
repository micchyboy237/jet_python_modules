from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore

import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5', revision="refs/pr/13")
model_ort = ORTModelForFeatureExtraction.from_pretrained('BAAI/bge-large-en-v1.5', revision="refs/pr/13",file_name="onnx/model.onnx")

# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

model_output_ort = model_ort(**encoded_input)
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# model_output and model_output_ort are identical