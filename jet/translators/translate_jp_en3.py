# Synchronous (scripts, legacy code, Jupyter, etc.)
from jet.translators.google_translator import translate_text

ja_texts = [
    "そんな一気に冷たいものを食べると腹を壊すぞ",
    "世界各国が水面架で知列な情報戦を繰り広げる時代に、にらみ合う2つの国、東のオスタニア、西のウェスタリス、戦",
    "争を加わだてるオスタニア政府要順の動向をさせ、",
]

ja = ja_texts[0]
en = translate_text(ja)

print(f"\nJA → {ja}")
print(f"EN → {en}\n")
