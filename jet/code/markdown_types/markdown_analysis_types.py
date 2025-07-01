from typing import TypedDict, List, Optional, Literal


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
    checked: Optional[bool]


class TableItemDict(TypedDict):
    header: List[str]
    rows: List[List[str]]


class LinkItemDict(TypedDict):
    line: int
    text: Optional[str]
    url: str
    alt_text: Optional[str]


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
        "header1", "header2", "header3", "header4", "header5", "header6",
        "paragraph", "inline_code", "link", "blockquote", "unordered_list",
        "task_item", "list_item", "code", "table", "italic", "html_block",
        "ordered_list"
    ]
    content: str
    url: Optional[str]
    checked: Optional[bool]


class MarkdownAnalysis(TypedDict):
    summary: SummaryDict
    headers: List[HeaderItemDict]
    paragraphs: List[str]
    blockquotes: List[str]
    code_blocks: List[CodeBlockItemDict]
    unordered_lists: List[List[ListItemDict]]
    ordered_lists: List[List[ListItemDict]]
    tables: List[TableItemDict]
    text_links: List[LinkItemDict]
    image_links: List[LinkItemDict]
    footnotes: List[FootnoteItemDict]
    inline_code: List[InlineCodeItemDict]
    emphasis: List[EmphasisItemDict]
    task_items: List[TaskItemDict]
    html_blocks: List[HtmlBlockItemDict]
    html_inline: List[HtmlInlineItemDict]
    tokens_sequential: List[TokenSequentialItemDict]
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
