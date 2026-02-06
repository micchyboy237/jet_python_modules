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
    example_html = """
    <html>
        <head>
            <title>Sample Page</title>
            <style>.hidden{display:none}</style>
            <script>console.log("noise")</script>
        </head>
        <body>
            <nav>Home | About | Contact</nav>
            <article>
                <h1>Extracting Text from HTML</h1>
                <p>This is the first sentence.</p>
                <p>This pipeline removes boilerplate and splits sentences correctly.</p>
                <p>It works well for scraped content!</p>
            </article>
            <footer>© 2026 Example Corp</footer>
        </body>
    </html>
    """

    sentences = html_to_sentences(example_html)

    for i, sentence in enumerate(sentences, start=1):
        print(f"{i}. {sentence}")
