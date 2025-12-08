import torch
from transformers import pipeline
from datasets import load_dataset

# config
model_id = "kotoba-tech/kotoba-whisper-v2.0"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "ja", "task": "transcribe"}

# load model
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs
)

# load sample audio
dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
sample = dataset[0]["audio"]

# run inference
result = pipe(sample, generate_kwargs=generate_kwargs)
print(result["text"])