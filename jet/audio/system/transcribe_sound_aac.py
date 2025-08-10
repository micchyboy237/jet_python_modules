from faster_whisper import WhisperModel

# Path to your audio file
AUDIO_FILE = "last5min.aac"

# Load model (you can use "small", "medium", "large-v2" depending on speed vs accuracy)
model_size = "small"
# use "metal" if GPU Metal backend is set up
model = WhisperModel(model_size, device="cpu")

# Transcribe
segments, info = model.transcribe(AUDIO_FILE)

print(
    f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
print("\n--- Transcript ---")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
