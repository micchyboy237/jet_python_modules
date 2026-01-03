import torch
from transformers import pipeline

generate_kwargs = {
    "language": "Japanese",
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
}
pipe = pipeline(
    "automatic-speech-recognition",
    model="litagin/anime-whisper",
    device="cuda",
    torch_dtype=torch.float16,
    chunk_length_s=30.0,
    batch_size=64,
)

audio_path = "test.wav"
result = pipe(audio_path, generate_kwargs=generate_kwargs)
print(result["text"])