import re
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from smolagents import Tool

# If available, could import from jet.models.utils, else fallback (shown here)
# from jet.models.utils import cosine_similarity

STOP_WORDS = {"click", "here", "link", "more", "read", "view", "go", "this", "page"}


def clean_text_for_embedding(text: str) -> str:
    text = re.sub(r"\W+", " ", text.lower()).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)


def lexical_path_bonus(url: str, goal_terms: list[str]) -> float:
    """Small bonus (0–0.15) if URL path contains goal keywords"""
    if not goal_terms:
        return 0.0
    path = urlparse(url).path.lower()
    score = sum(1 for term in goal_terms if term in path) / len(goal_terms)
    return min(score * 0.15, 0.15)


def extract_link_context(page_text: str, link_text: str, link_url: str) -> str:
    """Very simple context extraction — text ~50 chars before & after markdown link"""
    pattern = rf"\[({re.escape(link_text)})\]\({re.escape(link_url)}\)"
    match = re.search(pattern, page_text)
    if not match:
        return ""
    start = max(0, match.start() - 60)
    end = min(len(page_text), match.end() + 60)
    context = page_text[start:end].strip()
    # remove the link itself to avoid duplication
    context = re.sub(pattern, link_text, context)
    return context


class MarkdownSectionExtractor:
    """
    Extract markdown sections for ANY header level (# .. ######).
    """

    HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    @classmethod
    def extract_sections(cls, text: str) -> list[dict[str, str]]:
        matches = list(cls.HEADER_RE.finditer(text))
        if not matches:
            return []

        sections: list[dict[str, str]] = []

        for idx, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            # level is int — could be fixed with TypedDict. Keep as-is for now.
            sections.append(
                {
                    "level": level,
                    "title": title,
                    "content": content,
                }
            )

        return sections


class LinkExtractor:
    """
    Minimal markdown link extractor.
    Keeps logic isolated and replaceable.
    """

    MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    @classmethod
    def extract_markdown_links(cls, text: str) -> list[dict[str, str]]:
        return [
            {
                "text": label.strip(),
                "url": href.strip(),
            }
            for label, href in cls.MARKDOWN_LINK_RE.findall(text)
        ]


def normalize_text(text: str) -> str:
    return re.sub(r"\W+", " ", text).lower().strip()


class NavigationalSearchTool(Tool):
    """
    Suggest likely next navigation links from the current page.
    """

    name = "navigational_search"
    description = (
        "Analyze the current page and suggest relevant next navigation links "
        "based on a user goal or intent."
    )

    inputs = {
        "goal": {
            "type": "string",
            "description": "Optional navigation goal or intent used to rank links.",
            "nullable": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of navigation suggestions to return.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        browser,
        embedding_model: str = "nomic-embed-text",
    ):
        super().__init__()
        self.browser = browser
        self.embedder = LlamacppEmbedding(
            model=embedding_model,
            use_cache=True,
            verbose=False,
            use_dynamic_batch_sizing=True,
        )
        self.embed_func = self.embedder.get_embedding_function(
            return_format="numpy",
            batch_size=16,
            show_progress=False,
        )
        self.MIN_SIMILARITY_THRESHOLD = 0.40  # tune this after testing

    def _get_link_representation(self, link: dict[str, str], page_text: str) -> str:
        """Improved: title + surrounding context + cleaned"""
        text = (link.get("text") or "").strip()
        if not text:
            text = link.get("url", "")
        context = extract_link_context(
            page_text, link.get("text", ""), link.get("url", "")
        )
        parts = [text]
        if context:
            parts.append(context)
        raw = " | ".join(parts)
        cleaned = clean_text_for_embedding(raw)
        if not cleaned:
            cleaned = clean_text_for_embedding(text)  # fallback
        return cleaned

    def _rank_links(
        self,
        links: list[dict[str, str]],
        goal: str | None,
    ) -> list[dict[str, Any]]:
        if not goal or not goal.strip():
            return [
                {
                    **link,
                    "score": 0.0,
                    "reason": "No navigation goal provided — showing all links.",
                }
                for link in links
            ]

        goal_clean = clean_text_for_embedding(goal)
        goal_emb = self.embed_func(goal_clean)
        if getattr(goal_emb, "ndim", None) == 1:
            goal_emb = goal_emb[None, :]

        # page_text is needed for context → pass it
        link_reprs = [
            self._get_link_representation(link, self.browser.page_content)
            for link in links
        ]

        if not any(link_reprs):  # all empty after cleaning
            link_reprs = [
                clean_text_for_embedding(link.get("text") or link.get("url", ""))
                for link in links
            ]

        link_embs = self.embed_func(link_reprs)  # shape (N, d)

        similarities = np.dot(link_embs, goal_emb.T).flatten()

        goal_terms = goal_clean.split()

        scored = []
        for link, sim in zip(links, similarities):
            bonus = lexical_path_bonus(link["url"], goal_terms)
            final_score = float(sim) + bonus

            reason_parts = [f"Embedding similarity = {sim:.3f}"]
            if bonus > 0:
                reason_parts.append(f"+ path bonus {bonus:.3f}")

            scored.append(
                {
                    **link,
                    "score": final_score,
                    "reason": "; ".join(reason_parts),
                }
            )

        # Filter + sort
        if self.MIN_SIMILARITY_THRESHOLD > 0:
            scored = [
                s for s in scored if s["score"] >= self.MIN_SIMILARITY_THRESHOLD - 0.05
            ]  # soft filter
        scored.sort(key=lambda x: x["score"], reverse=True)

        # If nothing passed filter → return top 2 anyway (avoid empty result)
        if not scored and len(links) > 0:
            scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:2]

        return scored

    def forward(self, goal: str | None = None, max_results: int = 5) -> str:
        page_text = self.browser.page_content
        base_url = self.browser.address

        links = LinkExtractor.extract_markdown_links(page_text)
        if not links:
            return "No navigational links found on the current page."

        # Resolve relative URLs
        for link in links:
            link["url"] = urljoin(base_url, link["url"])

        ranked = self._rank_links(links, goal)

        output_lines: list[str] = []
        for idx, item in enumerate(ranked[:max_results], start=1):
            output_lines.append(
                f"{idx}. {item['text'] or '(no text)'}"
                f"\n   → URL: {item['url']}"
                f"\n   → Score: {item['score']:.3f}"
                f"\n   → Reason: {item['reason']}"
            )

        if not output_lines:
            return "No relevant navigation suggestions found."

        return "\n\n".join(output_lines)
