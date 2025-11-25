# example_3_m2m100.py
# pip install transformers sentencepiece torch

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import Optional
import torch

from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class M2MTranslator:
    def __init__(self, model_name: str = "facebook/m2m100_418M", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name, use_safetensors=False)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name, use_safetensors=False).to(self.device)

    def translate(self, text: str, src_lang: str = "ja", tgt_lang: str = "en") -> str:
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos_token_id = self.tokenizer.get_lang_id(tgt_lang)
        generated = self.model.generate(**encoded, forced_bos_token_id=forced_bos_token_id, max_new_tokens=200)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

if __name__ == "__main__":
    tr = M2MTranslator()
    japanese_text = "世界各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏百の顔を使い分ける彼の任務は家族を作ること父ロイドフォージャー精神"

    english_text = tr.translate("こんにちは、元気ですか？", src_lang="ja", tgt_lang="en")
    
    print(f"JA Transcript: {japanese_text[:100]}...")  # Preview (optional)
    print(f"EN Translation: {english_text[:100]}...")  # Preview (optional)

    # Save both
    save_file(japanese_text, f"{OUTPUT_DIR}/transcript_ja.txt")
    save_file(english_text, f"{OUTPUT_DIR}/transcript_en.txt")
