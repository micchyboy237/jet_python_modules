from typing import List, Literal, TypedDict


# Define supported markdown extensions as Literal for type safety
MarkdownExtension = Literal[
    "extra",
    "abbr",
    "attr_list",
    "def_list",
    "fenced_code",
    "footnotes",
    "md_in_html",
    "tables",
    "admonition",
    "codehilite",
    "legacy_attrs",
    "legacy_em",
    "meta",
    "nl2br",
    "sane_lists",
    "smarty",
    "toc",
    "wikilinks",
]


class MarkdownExtensions(TypedDict):
    extensions: List[MarkdownExtension]
