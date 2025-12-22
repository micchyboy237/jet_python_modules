from faster_whisper import WhisperModel
from jet.translators.translate_jp_en_ct2 import translate_ja_to_en

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster", device="cpu", compute_type="int8")

AUDIO_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251222_125319.wav"

segments, info = model.transcribe(AUDIO_PATH, language="ja", chunk_length=30, condition_on_previous_text=False)

all_jp = []
all_en = []

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]")
    jp_text = segment.text
    print(f"JP: {jp_text}")
    en_text = translate_ja_to_en(jp_text)
    print(f"EN: {en_text}")
    all_jp.append(jp_text)
    all_en.append(en_text)

print("\n===== FINAL TRANSCRIPT (JP) =====")
print("".join(all_jp).strip())

print("\n===== FINAL TRANSLATION (EN) =====")
print("".join(all_en).strip())
