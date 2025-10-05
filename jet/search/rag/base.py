import re
from tqdm import tqdm

from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
from jet.search.formatters import clean_string

def preprocess_texts(texts: str | list[str]) -> list[str]:
    def replace_underscores_dashes(text: str) -> str:
        # e.g. "AI-powered" -> "AI powered", "max_depth" -> "max depth", but "AI - powered" stays unchanged
        return re.sub(r'(?<!\s)[_-](?!\s)', ' ', text)

    if isinstance(texts, str):
        texts = [texts]

    # Lowercase
    preprocessed_texts = [text.lower() for text in texts]
    for idx, text in enumerate(tqdm(preprocessed_texts, desc="Preprocessing texts")):
        # e.g. "AI-powered" -> "AI powered", "max_depth" -> "max depth"
        text = replace_underscores_dashes(text)
        
        text = clean_newlines(text, max_newlines=1)
        text = clean_punctuations(text)
        text = clean_spaces(text)
        text = clean_string(text)
        
        preprocessed_texts[idx] = text

    return preprocessed_texts