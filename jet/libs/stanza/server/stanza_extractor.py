# jet_python_modules/jet/libs/stanza/server/stanza_extractor.py
import os
from typing import Any, Dict, List, Optional, Union, Literal
from stanza.server import CoreNLPClient, StartServer, to_text
from stanza.models.constituency import tree_reader

class StanzaExtractor:
    """
    A reusable wrapper around Stanford CoreNLP via Stanza's CoreNLPClient.

    Supports annotation, TokensRegex, Semgrex, and Tregex queries.
    """

    def __init__(
        self,
        *,
        annotators: str = "tokenize,ssplit,pos,lemma,ner,depparse",
        endpoint: str = "http://localhost:9000",
        start_server: Union[bool, StartServer] = StartServer.TRY_START,
        timeout: int = 60000,
        memory: str = "5G",
        corenlp_home: Optional[str] = None,
        quiet: bool = True,
    ) -> None:
        self.client: CoreNLPClient = CoreNLPClient(
            annotators=annotators,
            endpoint=endpoint,
            start_server=start_server,
            timeout=timeout,
            memory=memory,
            be_quiet=quiet,
            corenlp_home=corenlp_home or os.getenv("CORENLP_HOME"),
        )

    def start(self) -> None:
        """Starts the CoreNLP client connection."""
        self.client.start()

    def stop(self) -> None:
        """Stops the CoreNLP client and releases resources."""
        self.client.stop()

    def annotate_text(
        self, text: str, *, output_format: Literal["json", "text", "xml"] = "json"
    ) -> Any:
        """Annotates text using the configured pipeline."""
        return self.client.annotate(text, output_format=output_format)

    def get_plain_text(self, annotation: Any) -> str:
        """
        Extracts plain text from an annotation.

        Works for both protobuf and JSON annotations.
        """
        # Case 1: protobuf annotation (has .sentence)
        if hasattr(annotation, "sentence"):
            return to_text(annotation)

        # Case 2: JSON dict (contains 'sentences')
        if isinstance(annotation, dict) and "sentences" in annotation:
            sentences = [
                " ".join(token["word"] for token in sent["tokens"])
                for sent in annotation["sentences"]
            ]
            return " ".join(sentences)

        raise TypeError("Unsupported annotation type for get_plain_text()")

    def tokensregex(
        self, text: str, pattern: str
    ) -> Dict[str, Any]:
        """Applies a TokensRegex pattern."""
        return self.client.tokensregex(text, pattern)

    def semgrex(
        self, text: str, pattern: str, *, to_words: bool = True
    ) -> List[Dict[str, Any]]:
        """Applies a Semgrex pattern."""
        return self.client.semgrex(text, pattern, to_words=to_words)

    def tregex(
        self,
        text_or_pattern: str,
        pattern: Optional[str] = None,
        *,
        trees: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Applies a Tregex pattern to text or parsed trees.
        If `trees` is provided, it runs directly on trees instead of text.
        """
        if trees:
            return self.client.tregex(pattern=text_or_pattern, trees=trees)
        return self.client.tregex(text_or_pattern, pattern)

    def parse_trees(self, trees_str: str) -> Any:
        """Parses a string containing multiple tree structures."""
        return tree_reader.read_trees(trees_str)
