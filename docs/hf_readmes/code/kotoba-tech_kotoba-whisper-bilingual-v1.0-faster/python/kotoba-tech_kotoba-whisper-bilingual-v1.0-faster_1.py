from faster_whisper import WhisperModel

model = WhisperModel("kotoba-tech/kotoba-whisper-bilingual-v1.0-faster")

# Japanese ASR
segments, info = model.transcribe("sample_ja.flac", language="ja", task="transcribe", condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# English ASR
segments, info = model.transcribe("sample_en.wav", language="en", task="transcribe", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# Japanese (speech) to English (text) Translation
segments, info = model.transcribe("sample_ja.flac", language="en", task="translate", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# English (speech) to Japanese (text) Translation
segments, info = model.transcribe("sample_en.wav", language="ja", task="translate", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))