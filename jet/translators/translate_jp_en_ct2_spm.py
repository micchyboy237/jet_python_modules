from pathlib import Path
import ctranslate2
import sentencepiece as spm

QUANTIZED_MODEL_PATH = str(Path("~/.cache/hf_ctranslate2_models/opus-ja-en-ct2").expanduser().resolve())
SOURCE_SPM_PATH = str(Path("~/.cache/hf_ctranslate2_models/opus-ja-en-ct2/source.spm").expanduser().resolve())
ja_single = "おい、そんな一気に冷たいものを食べると腹を壊すぞ！"

sp = spm.SentencePieceProcessor()
sp.load(SOURCE_SPM_PATH)

source = sp.encode(ja_single, out_type=str)

translator = ctranslate2.Translator(QUANTIZED_MODEL_PATH)
results = translator.translate_batch([source])

output = sp.decode(results[0].hypotheses[0])
print(output)