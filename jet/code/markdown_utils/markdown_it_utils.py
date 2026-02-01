# markdown_it_utils.py

from __future__ import annotations

import re
from typing import NotRequired, TypedDict

from markdown_it import MarkdownIt
from markdown_it.token import Token
from markdown_it.utils import OptionsDict
from markdownify import markdownify


class ParseResult(TypedDict):
    """Structured result of markdown-it-py parsing and rendering."""

    markdown_input: str
    """Original markdown text that was passed in (possibly normalized)."""

    tokens: list[Token]
    """Flat list of parsing tokens produced by md.parse()."""

    html: str
    """Final rendered HTML string from md.render()."""

    env: NotRequired[dict]
    """Environment dictionary that may contain plugin-specific data
    (wordcount, front_matter parsed data, etc.)."""

    options: NotRequired[OptionsDict]
    """Copy of the options used during rendering."""


def parse_and_render_markdown(
    md: MarkdownIt,
    text: str,
    /,
    *,
    normalize_whitespace: bool = True,
) -> ParseResult:
    """Parse markdown text and render it to HTML using the provided MarkdownIt instance.

    This is a thin, reusable wrapper that:

    - normalizes input whitespace (optional)
    - captures tokens, html output, env and options
    - keeps strong typing and clear contract

    Args:
        md: Configured MarkdownIt instance (with plugins already loaded)
        text: Markdown source string
        normalize_whitespace: Whether to strip trailing/leading whitespace and
                              normalize line endings (recommended for testing)

    Returns:
        Structured parse & render result
    """
    # Optional normalization — helps make tests stable
    if normalize_whitespace:
        text = text.strip()
        text = "\n".join(line.rstrip() for line in text.splitlines())

    tokens = md.parse(text)
    html = md.render(text)

    result: ParseResult = {
        "markdown_input": text,
        "tokens": tokens,
        "html": html,
    }

    # Capture interesting side-effects if present
    if hasattr(tokens, "env") and tokens.env is not None:
        result["env"] = dict(tokens.env)  # shallow copy

    result["options"] = dict(md.options)  # shallow copy

    return result


# HTML → Markdown conversion utilities.


from typing import Literal


def html_to_clean_markdown(
    html: str,
    *,
    # ──────────────────────────────────────────────
    heading_style: str = "ATX_CLOSED",
    bullets: str = "-*+",
    strip: list[str] | None = None,
    autolink: bool = True,
    keep_inline_images_in: list[str] | None = None,
    strong_em_symbol: Literal["*", "_"] = "*",
    table_infer_header: bool = False,
    strip_document: Literal["lstrip", "rstrip", "strip", None] = "strip",
    strip_pre: Literal["lstrip", "rstrip", "strip", "strip_one", None] = "strip",
    escape_misc: bool = True,
    escape_misc_chars: str = r"""$%^&`~\|{}[]""",  # added control
    escape_asterisks: bool = True,
    escape_underscores: bool = True,
    newline_style: str = "SPACES",
    wrap: bool = False,
    wrap_width: int = 80,
) -> str:
    """Convert HTML string to clean, readable Markdown using markdownify.

    Optimized defaults for LLM/RAG pipelines:
    • Cleaner output with fewer artifacts
    • More predictable structure for chunking & embedding
    • Aggressive removal of noisy tags
    • Better heading & table handling
    """
    clean_html = html.strip()
    if keep_inline_images_in is None:
        keep_inline_images_in = []

    # Stronger default stripping for RAG — removes most tracking/noise
    result = markdownify(
        clean_html,
        heading_style=heading_style,
        bullets=bullets,
        strip=strip
        or [
            "script",
            "style",
            "noscript",
            "iframe",
            "form",
            "svg",
            "path",
            "symbol",
            "defs",  # vector graphics junk
            "nav",
            "footer",
            "header",  # often repetitive
            "aside",
            "figure",
            "figcaption",
        ],
        autolinks=autolink,
        keep_inline_images_in=keep_inline_images_in or [],
        # Better table readability for RAG
        table_infer_header=table_infer_header,
        strong_em_symbol=strong_em_symbol,
        strip_document=strip_document,
        strip_pre=strip_pre,
        escape_misc=escape_misc,
        escape_asterisks=escape_asterisks,
        escape_underscores=escape_underscores,
        # You can pass custom chars if needed
        # escape_misc_chars=escape_misc_chars,
        newline_style="SPACES",  # most stable for chunking
        # Usually better to disable wrapping for RAG — preserves original intent
        # and avoids cutting tokens awkwardly
        wrap=False,
        # wrap_width=80,
    )
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
