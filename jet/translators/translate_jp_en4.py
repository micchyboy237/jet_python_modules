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
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "cpu"  # Force CPU on Apple Silicon (avoids known MPS bugs)
            else:
                self.device = "cpu"

        # === PREVENT AUTO DOWNLOAD OF SAFETENSORS + ANY MISSING FILES ===
        preload_kwargs = {
            "local_files_only": True,       # Fail if files not already cached
            "use_safetensors": False,       # Explicitly disable safetensors
            "torch_dtype": torch.float32,   # Optional: ensure consistency
        }

        try:
            self.tokenizer = M2M100Tokenizer.from_pretrained(
                model_name, **preload_kwargs
            )
            self.model = M2M100ForConditionalGeneration.from_pretrained(
                model_name, **preload_kwargs
            ).to(self.device)
        except OSError as e:
            if "local_files_only" in str(e):
                raise RuntimeError(
                    f"Model '{model_name}' not found in cache and downloading is disabled. "
                    "Run once without 'local_files_only=True' to download, or place files manually."
                ) from e
            raise

    def translate(self, text: str, src_lang: str = "ja", tgt_lang: str = "en") -> str:
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos_token_id = self.tokenizer.get_lang_id(tgt_lang)

        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=200,
            # Retain these for better quality/reduced repetition (works fine on CPU)
            no_repeat_ngram_size=3,
            num_beams=5,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

if __name__ == "__main__":
    tr = M2MTranslator()
    japanese_text = "世界各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏百の顔を使い分ける彼の任務は家族を作ること父ロイドフォージャー精神"

    english_text = tr.translate(japanese_text, src_lang="ja", tgt_lang="en")
    
    print(f"JA Transcript: {japanese_text[:100]}...")  # Preview (optional)
    print(f"EN Translation: {english_text[:100]}...")  # Preview (optional)

    # Save both
    save_file(japanese_text, f"{OUTPUT_DIR}/transcript_ja.txt")
    save_file(english_text, f"{OUTPUT_DIR}/transcript_en.txt")
