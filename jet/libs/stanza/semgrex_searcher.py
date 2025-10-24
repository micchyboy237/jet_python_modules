from typing import List, Dict, Any
import stanza
from stanza.server.semgrex import Semgrex


class SemgrexSearcher:
    """
    A reusable class for searching dependency patterns in Stanza-parsed documents using Semgrex.
    Initializes a Stanza pipeline once for efficiency; supports batch processing of sentences.
    """
    def __init__(self, lang: str = "en", processors: str = "tokenize,pos,lemma,depparse", use_java: bool = True):
        self.nlp = stanza.Pipeline(lang=lang, processors=processors, use_gpu=True)  # GPU for M1/Windows perf
        self.semgrex = Semgrex() if use_java else None  # Java server required for Semgrex

    def search_in_doc(self, text: str, pattern: str) -> List[Dict[str, Any]]:
        """
        Parse text into a Document, query Semgrex pattern, and return matches.
        Each match is a dict with named nodes (if used in pattern) and sentence text.
        """
        if not self.semgrex:
            raise RuntimeError("Semgrex requires Java server; ensure CoreNLP is accessible.")

        doc: stanza.Document = self.nlp(text)
        matches: List[Dict[str, Any]] = []

        with self.semgrex as sem:
            for sentence in doc.sentences:
                result = sem.process(sentence, pattern)
                if result and result.get("sentences"):
                    for sent_match in result["sentences"]:
                        match_info = {
                            "sentence_text": sentence.text,
                            "matched_nodes": sent_match.get("nodes", {}),  # Named captures, e.g., {}=subject
                            "length": sent_match.get("length", 0)
                        }
                        matches.append(match_info)

        return matches

    def close(self) -> None:
        """Cleanup: Close Java server if open."""
        if self.semgrex:
            self.semgrex.close()