import shutil
from pathlib import Path

from jet.wordnet.histogram import TextAnalysis
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DATA = []

TOP_N = 200


if __name__ == "__main__":
    ta = TextAnalysis(SAMPLE_DATA)

    most_start_results = ta.generate_histogram(
        is_top=True,
        from_start=True,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 1), (2, 3)],
        top_n=TOP_N,
    )
    least_start_results = ta.generate_histogram(
        is_top=False,
        from_start=True,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 1), (2, 3)],
        top_n=TOP_N,
    )
    most_any_results = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 3), (4, 6)],
        top_n=TOP_N,
    )
    least_any_results = ta.generate_histogram(
        is_top=False,
        from_start=False,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 3), (4, 6)],
        top_n=TOP_N,
    )
