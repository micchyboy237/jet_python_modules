import os
import shutil
import argostranslate.package
import argostranslate.translate
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def setup_argos_model(from_lang: str = "ja", to_lang: str = "en") -> None:
    """
    Modular setup: Updates package index and installs JA-EN model if missing.
    Idempotent—one-time download (~200-300MB).
    """
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_lang and x.to_code == to_lang,
            available_packages
        ),
        None
    )
    if package_to_install:
        argostranslate.package.install_from_path(package_to_install.download())

def translate_text(text: str, from_lang: str = "ja", to_lang: str = "en") -> str:
    """
    Reusable translation function: Offline JA→EN using Argos Translate.
    Returns natural English; handles empty input gracefully.
    """
    if not text.strip():
        return ""
    return argostranslate.translate.translate(text, from_lang, to_lang)

if __name__ == "__main__":
    japanese_text = "世界各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏百の顔を使い分ける彼の任務は家族を作ること父ロイドフォージャー精神"

    # One-time model setup (call once; safe to rerun)
    setup_argos_model()

    # Translate to English (offline with Argos Translate)
    english_text = translate_text(japanese_text)

    print(f"JA Transcript: {japanese_text[:100]}...")  # Preview (optional)
    print(f"EN Translation: {english_text[:100]}...")  # Preview (optional)

    # Save both
    save_file(japanese_text, f"{OUTPUT_DIR}/transcript_ja.txt")
    save_file(english_text, f"{OUTPUT_DIR}/transcript_en.txt")
