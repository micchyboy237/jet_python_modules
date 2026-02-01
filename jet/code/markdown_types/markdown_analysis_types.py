from typing import Literal, TypedDict


class HeaderCountsDict(TypedDict):
    h1: int
    h2: int
    h3: int
    h4: int
    h5: int
    h6: int


class SummaryDict(TypedDict):
    headers: int
    header_counts: HeaderCountsDict
    paragraphs: int
    blockquotes: int
    code_blocks: int
    ordered_lists: int
    unordered_lists: int
    tables: int
    html_blocks: int
    html_inline_count: int
    words: int
    characters: int
    text_links: int
    image_links: int


class HeaderItemDict(TypedDict):
    line: int
    level: Literal[1, 2, 3, 4, 5, 6]
    text: str


class CodeBlockItemDict(TypedDict):
    start_line: int
    content: str
    language: str


class ListItemDict(TypedDict):
    text: str
    task_item: bool
    checked: bool | None


class TableItemDict(TypedDict):
    header: list[str]
    rows: list[list[str]]


class LinkItemDict(TypedDict):
    line: int
    text: str | None
    url: str
    alt_text: str | None


class FootnoteItemDict(TypedDict):
    line: int
    id: str
    content: str


class InlineCodeItemDict(TypedDict):
    line: int
    code: str


class EmphasisItemDict(TypedDict):
    line: int
    text: str


class TaskItemDict(TypedDict):
    line: int
    text: str
    checked: bool


class HtmlBlockItemDict(TypedDict):
    line: int
    content: str


class HtmlInlineItemDict(TypedDict):
    line: int
    html: str


class TokenSequentialItemDict(TypedDict):
    id: int
    type: Literal[
        "title",
        "header1",
        "header2",
        "header3",
        "header4",
        "header5",
        "header6",
        "paragraph",
        "inline_code",
        "link",
        "blockquote",
        "unordered_list",
        "task_item",
        "list_item",
        "code",
        "table",
        "italic",
        "html_block",
        "ordered_list",
    ]
    content: str
    url: str | None
    checked: bool | None


class MarkdownAnalysis(TypedDict):
    summary: SummaryDict
    headers: list[HeaderItemDict]
    paragraphs: list[str]
    blockquotes: list[str]
    code_blocks: list[CodeBlockItemDict]
    unordered_lists: list[list[ListItemDict]]
    ordered_lists: list[list[ListItemDict]]
    tables: list[TableItemDict]
    text_links: list[LinkItemDict]
    image_links: list[LinkItemDict]
    footnotes: list[FootnoteItemDict]
    inline_code: list[InlineCodeItemDict]
    emphasis: list[EmphasisItemDict]
    task_items: list[TaskItemDict]
    html_blocks: list[HtmlBlockItemDict]
    html_inline: list[HtmlInlineItemDict]
    tokens_sequential: list[TokenSequentialItemDict]
    word_count: int
    char_count: int


__all__ = [
    "HeaderCountsDict",
    "SummaryDict",
    "HeaderItemDict",
    "CodeBlockItemDict",
    "ListItemDict",
    "TableItemDict",
    "LinkItemDict",
    "FootnoteItemDict",
    "InlineCodeItemDict",
    "EmphasisItemDict",
    "TaskItemDict",
    "HtmlBlockItemDict",
    "HtmlInlineItemDict",
    "TokenSequentialItemDict",
    "MarkdownAnalysis",
]
