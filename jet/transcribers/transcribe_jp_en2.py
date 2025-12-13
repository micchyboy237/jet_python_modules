from faster_whisper import WhisperModel
from jet.translators.translate_jp_en_ct2 import translate_ja_to_en

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster")

# Example usage
AUDIO_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav"

# Japanese ASR
segments, info = model.transcribe(AUDIO_PATH, language="ja", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]")
    print(f"JP: {segment.text}")
    text_original = segment.text
    text_en = translate_ja_to_en(text_original)
    print(f"EN: {text_en}")
