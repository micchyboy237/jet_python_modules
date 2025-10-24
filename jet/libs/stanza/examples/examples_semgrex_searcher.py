# jet/examples/semgrex_examples.py
"""
Real-world Semgrex usage examples as reusable functions + a main block.
All heavy lifting is done by `SemgrexSearcher` imported from the library.
"""

from typing import List, Dict
import sys
import pathlib

# ----------------------------------------------------------------------
# Adjust import path so the script works when executed from any cwd
# ----------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]   # one level up from examples/
sys.path.insert(0, str(PROJECT_ROOT))

# Import the reusable searcher
from jet.libs.stanza.semgrex_searcher import SemgrexSearcher


# ----------------------------------------------------------------------
# Helper: flatten matched nodes to just the token text
# ----------------------------------------------------------------------
def _node_text(nodes: Dict[str, Dict], name: str) -> str:
    """Return the ``text`` field of a named node or an empty string."""
    return nodes.get(name, {}).get("text", "")


# ----------------------------------------------------------------------
# Example functions (pure – they only transform the raw matches)
# ----------------------------------------------------------------------
def find_acquisition_events(searcher: SemgrexSearcher, text: str) -> List[Dict[str, str]]:
    """
    Extract company-acquisition events:
        PROPN subject → acquire (any form) → PROPN object
    """
    pattern = (
        '{pos:/NNP|PROPN/}=subject >nsubj '
        '{lemma:acquire}=verb >dobj '
        '{pos:/NNP|PROPN/}=object'
    )
    raw = searcher.search_in_doc(text, pattern)
    return [
        {
            "subject": _node_text(m["matched_nodes"], "subject"),
            "verb": _node_text(m["matched_nodes"], "verb"),
            "object": _node_text(m["matched_nodes"], "object"),
            "sentence": m["sentence_text"],
        }
        for m in raw
    ]


def find_passive_constructions(searcher: SemgrexSearcher, text: str) -> List[Dict[str, str]]:
    """
    Detect passive voice: verb with a passive ``be`` auxiliary.
    """
    pattern = '{}=verb >aux:pass {lemma:be}=aux'
    raw = searcher.search_in_doc(text, pattern)
    return [
        {
            "aux": _node_text(m["matched_nodes"], "aux"),
            "verb": _node_text(m["matched_nodes"], "verb"),
            "sentence": m["sentence_text"],
        }
        for m in raw
    ]


def find_wh_subject_questions(searcher: SemgrexSearcher, text: str) -> List[Dict[str, str]]:
    """
    Find WH-questions where the WH-word is the subject of the verb.
    """
    pattern = '{pos:/WP|WDT/}=wh >nsubj {}=verb'
    raw = searcher.search_in_doc(text, pattern)
    return [
        {
            "wh_word": _node_text(m["matched_nodes"], "wh"),
            "verb": _node_text(m["matched_nodes"], "verb"),
            "sentence": m["sentence_text"],
        }
        for m in raw
    ]


# ----------------------------------------------------------------------
# Main block – demo of the three use-cases
# ----------------------------------------------------------------------
if __name__ == "__main__":
    searcher = SemgrexSearcher()
    try:
        # ------------------------------------------------------------------
        # Sample corpora
        # ------------------------------------------------------------------
        news_text = (
            "Apple acquires Beats for $3 billion in 2014. "
            "Google buys DeepMind. Microsoft partners with OpenAI."
        )
        review_text = (
            "The movie was praised by critics. "
            "Users loved the ending. It was hated by some."
        )
        qa_text = (
            "Who invented the telephone? "
            "What is artificial intelligence? Bell did it."
        )

        # ------------------------------------------------------------------
        # 1. Acquisition events
        # ------------------------------------------------------------------
        print("\n=== Acquisition Events ===")
        for ev in find_acquisition_events(searcher, news_text):
            print(f"{ev['subject']} → {ev['verb']} → {ev['object']} | \"{ev['sentence']}\"")

        # ------------------------------------------------------------------
        # 2. Passive voice
        # ------------------------------------------------------------------
        print("\n=== Passive Constructions ===")
        for p in find_passive_constructions(searcher, review_text):
            print(f"{p['aux']} {p['verb']} | \"{p['sentence']}\"")

        # ------------------------------------------------------------------
        # 3. WH-subject questions
        # ------------------------------------------------------------------
        print("\n=== WH-Subject Questions ===")
        for q in find_wh_subject_questions(searcher, qa_text):
            print(f"{q['wh_word']} {q['verb']} | \"{q['sentence']}\"")

    finally:
        searcher.close()