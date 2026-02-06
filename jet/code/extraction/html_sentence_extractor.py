import re
import unicodedata

import pysbd
import trafilatura
from bs4 import BeautifulSoup

_SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=True)


def _normalize_text(text: str) -> str:
    """
    Unicode + whitespace normalization.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fallback_bs4_text(html: str) -> str:
    """
    Fallback extraction if trafilatura fails.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator=" ")


def extract_main_text(html: str) -> str:
    """
    Extracts main readable text from HTML.
    Uses trafilatura with a BeautifulSoup fallback.
    """
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=True,
    )

    if not extracted:
        extracted = _fallback_bs4_text(html)

    return _normalize_text(extracted)


def split_into_sentences(text: str) -> list[str]:
    """
    Splits normalized text into clean sentences.
    """
    sentences = _SENTENCE_SEGMENTER.segment(text)

    cleaned: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue
        if not re.search(r"[a-zA-Z]", s):
            continue
        cleaned.append(s)

    return cleaned


def html_to_sentences(html: str) -> list[str]:
    """
    End-to-end pipeline:
    HTML -> main text -> normalized -> sentences
    """
    text = extract_main_text(html)
    return split_into_sentences(text)


if __name__ == "__main__":
    from pathlib import Path

    from jet.file.utils import save_file

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    # Load HTML content from the specified file
    html_file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors/semantic_search/generated/run_web_search/top_isekai_anime_2026/pages/gamerant_com_new_isekai_anime_2026/page.html"
    with open(html_file_path, encoding="utf-8") as file:
        html = file.read()

    sentences = html_to_sentences(html)

    for i, sentence in enumerate(sentences[:10], start=1):
        print(f"{i}. {sentence}")

    save_file(sentences, OUTPUT_DIR / "sentences.json")
