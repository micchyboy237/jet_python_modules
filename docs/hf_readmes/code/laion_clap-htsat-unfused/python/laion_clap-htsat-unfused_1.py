from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("ashraq/esc50")
audio = dataset["train"]["audio"][-1]["array"]

audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
output = audio_classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
print(output)
>>> [{"score": 0.999, "label": "Sound of a dog"}, {"score": 0.001, "label": "Sound of vaccum cleaner"}]