import shutil
from jet.file.utils import save_file
from jet.wordnet.analyzers.analyze_ngrams import analyze_ngrams
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    # Example dataset
    texts = [
        "The battery life of this phone is excellent and lasts all day.",
        "Camera performance is stunning even in low light conditions.",
        "Gaming performance is fast but drains the battery quickly.",
        "Display quality is vibrant and sharp.",
        "The phone overheats during heavy use but cools down quickly."
    ]

    # Example 1 — default 2–3 n-grams, TF-IDF threshold 0.05
    print("\n=== Example 1: Default Parameters ===")
    result = analyze_ngrams(texts)
    for r in result:
        print(f"- {r['text']} | max_tfidf={r['max_tfidf']:.4f}")
    save_file(result, f"{OUTPUT_DIR}/example1_ngrams.json")

    # Example 2 — 1–2 n-grams with stricter filtering
    print("\n=== Example 2: Custom n-gram range (1–2) & min_tfidf=0.1 ===")
    result = analyze_ngrams(
        texts,
        min_tfidf=0.1,
        ngram_ranges=[(1, 2)],
        top_k_texts=3
    )
    for r in result:
        print(f"- {r['text']} | matched_ngrams={len(r['matched_ngrams'])}")
    save_file(result, f"{OUTPUT_DIR}/example2_ngrams.json")

    # Example 3 — Excluding specific phrases
    print("\n=== Example 3: Excluding stop n-grams ===")
    stop_ngrams = {"battery life", "camera performance"}
    result = analyze_ngrams(
        texts,
        stop_ngrams=stop_ngrams,
        ngram_ranges=[(2, 3)],
    )
    for r in result:
        print(f"- {r['text']} | filtered n-grams")
    save_file(result, f"{OUTPUT_DIR}/example3_ngrams.json")
