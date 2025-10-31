from __future__ import annotations
from typing import Any, List, Optional, TypedDict, Dict, overload, Literal

class SemgrexNode(TypedDict):
    name: str
    text: str
    attributes: dict

class SemgrexMatch(TypedDict):
    doc_index: int
    sentence_index: int
    match_index: int
    nodes: List[SemgrexNode]

class SemgrexSearcher:
    def __init__(self, client: Optional[Any] = None) -> None:
        self.client: Optional[Any] = client

    def set_client(self, client: Any) -> None:
        """Set or update the CoreNLP client after initialization."""
        self.client = client

    @overload
    def search(self, text: str, pattern: str) -> List[SemgrexMatch]: ...
    @overload
    def search(self, raw_matches: Any, mode: Literal["raw"]) -> List[SemgrexMatch]: ...

    def search(self, arg1: Any, arg2: Optional[str] = None, *, mode: Optional[Literal["raw"]] = None) -> List[SemgrexMatch]:
        """
        Unified search interface.

        - With client: `search(text, pattern)`
        - Without client: `search(raw_matches, mode="raw")`
        """
        if mode == "raw":
            if arg2 is not None:
                raise ValueError("When mode='raw', only one argument (raw_matches) is expected.")
            return self._normalize(arg1)
        else:
            if not isinstance(arg1, str) or not isinstance(arg2, str):
                raise ValueError("When mode is not 'raw', provide (text: str, pattern: str).")
            return self.search_documents([arg1], arg2)

    @overload
    def search_documents(self, docs: List[str], pattern: str) -> List[SemgrexMatch]: ...
    @overload
    def search_documents(self, raw_matches_list: List[Any], mode: Literal["raw"]) -> List[SemgrexMatch]: ...

    def search_documents(
        self,
        arg1: Any,
        arg2: Optional[str] = None,
        *,
        mode: Optional[Literal["raw"]] = None
    ) -> List[SemgrexMatch]:
        """
        Batch version of search.

        - With client: `search_documents(docs_list, pattern)`
        - Without client: `search_documents(raw_list, mode="raw")`
        """
        if mode == "raw":
            if arg2 is not None:
                raise ValueError("When mode='raw', only one argument (raw_matches_list) is expected.")
            all_matches: List[SemgrexMatch] = []
            normalized_list = [self._normalize(raw) for raw in arg1]
            for doc_idx, normalized in enumerate(normalized_list):
                for m in normalized:
                    m["doc_index"] = doc_idx
                all_matches.extend(normalized)
            return all_matches
        else:
            if not isinstance(arg1, list) or not all(isinstance(d, str) for d in arg1) or not isinstance(arg2, str):
                raise ValueError("When mode is not 'raw', provide (docs: List[str], pattern: str).")
            if self.client is None:
                raise RuntimeError("No client configured. Use set_client() or pass raw data with mode='raw'.")
            if not hasattr(self.client, "semgrex"):
                raise RuntimeError("Client does not expose `semgrex` method.")

            all_matches: List[SemgrexMatch] = []
            for doc_idx, doc in enumerate(arg1):
                raw = self.client.semgrex(text=doc, pattern=arg2)
                normalized = self._normalize(raw)
                for m in normalized:
                    m["doc_index"] = doc_idx
                all_matches.extend(normalized)
            return all_matches

    def _normalize(self, raw_matches: Any) -> List[SemgrexMatch]:
        """
        Normalize any supported semgrex return shape into List[SemgrexMatch].
        Supported inputs:
          - dict with "matches" key → list
          - list of match dicts
          - single match dict
        """
        normalized: List[SemgrexMatch] = []

        # 1. Extract iterable of raw match dicts
        if isinstance(raw_matches, dict):
            if "matches" in raw_matches:
                raw_iter = raw_matches["matches"]
            else:
                # single match dict (no wrapper)
                raw_iter = [raw_matches]
        elif isinstance(raw_matches, (list, tuple)):
            raw_iter = raw_matches
        else:
            # fallback: treat as single malformed match
            raw_iter = [raw_matches]

        # 2. Normalize each match
        for raw in raw_iter:
            # Skip non-dict entries
            if not isinstance(raw, dict):
                continue

            sentence_index = int(raw.get("sentenceIndex", raw.get("sentence_index", -1)))
            match_index = int(raw.get("matchNumber", raw.get("match_index", raw.get("matchIndex", -1))))
            nodes_raw = raw.get("nodes", {})

            nodes_list: List[SemgrexNode] = []
            if isinstance(nodes_raw, dict):
                for name, node in nodes_raw.items():
                    text_val: Optional[str] = None
                    attributes: dict = {}
                    if isinstance(node, dict):
                        text_val = node.get("text") or node.get("word") or node.get("origText")
                        attributes = {
                            k: v for k, v in node.items()
                            if k not in ("text", "word", "origText")
                        }
                    else:
                        text_val = str(node)
                    nodes_list.append(SemgrexNode(
                        name=name,
                        text=text_val or "",
                        attributes=attributes
                    ))
            else:
                nodes_list.append(SemgrexNode(name="node", text=str(nodes_raw), attributes={}))

            normalized.append(SemgrexMatch(
                doc_index=-1,  # placeholder
                sentence_index=sentence_index,
                match_index=match_index,
                nodes=nodes_list
            ))

        return normalized

    def filter_matches(
        self,
        matches: List[SemgrexMatch],
        *,
        doc_index: Optional[int] = None,
        sentence_index: Optional[int] = None,
        node_name: Optional[str] = None,
        node_text_contains: Optional[str] = None,
        node_attr: Optional[Dict[str, Any]] = None,
    ) -> List[SemgrexMatch]:
        filtered: List[SemgrexMatch] = []
        for match in matches:
            if doc_index is not None and match["doc_index"] != doc_index:
                continue
            if sentence_index is not None and match["sentence_index"] != sentence_index:
                continue
            nodes = match["nodes"]
            if node_name is not None and not any(n["name"] == node_name for n in nodes):
                continue
            if node_text_contains is not None and not any(
                node_text_contains.lower() in n["text"].lower() for n in nodes
            ):
                continue
            if node_attr is not None:
                if not any(
                    all(k in n["attributes"] and n["attributes"][k] == v for k, v in node_attr.items())
                    for n in nodes
                ):
                    continue
            filtered.append(match)
        return filtered