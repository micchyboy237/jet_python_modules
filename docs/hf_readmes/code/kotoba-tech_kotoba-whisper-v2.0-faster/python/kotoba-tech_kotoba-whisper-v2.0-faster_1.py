from faster_whisper import WhisperModel

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster")

segments, info = model.transcribe("sample_ja_speech.wav", language="ja", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))