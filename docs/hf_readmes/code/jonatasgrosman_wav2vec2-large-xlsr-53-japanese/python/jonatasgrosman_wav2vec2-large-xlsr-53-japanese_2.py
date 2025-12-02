import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "ja"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
SAMPLES = 10

# === REPLACE Common Voice with ReazonSpeech (high-quality, clean Japanese ASR dataset) ===
# Used in recent papers, much better transcription quality than Common Voice
test_dataset = load_dataset("reazon-research/reazonspeech", "all", split=f"test[:{SAMPLES}]")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")

def speech_file_to_array_fn(batch):
    # ReazonSpeech already provides audio arrays at 16kHz
    batch["speech"] = batch["audio"]["array"]
    batch["sentence"] = batch["transcription"].upper()  # or .strip().upper() if needed
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn, remove_columns=["audio"])

# Process all samples
inputs = processor(
    test_dataset["speech"],
    sampling_rate=16_000,
    return_tensors="pt",
    padding=True
).to(model.device)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

print(f"{'='*20} Evaluating on ReazonSpeech (high-quality Japanese ASR test set) {'='*20}\n")
for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference: ", test_dataset[i]["sentence"])
    print("Prediction:", predicted_sentence.strip())
    print()