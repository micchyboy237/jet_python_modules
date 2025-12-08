from faster_whisper import WhisperModel

model = WhisperModel("kotoba-tech/kotoba-whisper-bilingual-v1.0-faster")

# Example usage
AUDIO_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav"

# Japanese ASR
segments, info = model.transcribe(AUDIO_PATH, language="ja", task="transcribe", condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# Japanese (speech) to English (text) Translation
segments, info = model.transcribe(AUDIO_PATH, language="en", task="translate", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
