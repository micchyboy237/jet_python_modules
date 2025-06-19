from typing import TypedDict, List, Optional, Literal, Union
from pathlib import Path

# Reuse MarkdownToken from previous response


class ListItem(TypedDict, total=False):
    text: str
    task_item: bool
    checked: bool


class CodeMeta(TypedDict, total=False):
    language: Optional[str]
    code_type: Optional[Literal["indented"]]


class TableMeta(TypedDict):
    header: List[str]
    rows: List[List[str]]


MetaType = Union[ListItem, CodeMeta, TableMeta, dict]


class MarkdownToken(TypedDict):
    type: Literal[
        "header",
        "hr",
        "code",
        "html_block",
        "table",
        "frontmatter",
        "blockquote",
        "ordered_list",
        "unordered_list",
        "paragraph"
    ]
    content: Optional[str]
    level: Optional[int]
    meta: Optional[MetaType]
    line: int

# Typed dicts for analyze_markdown components


class HeaderItem(TypedDict):
    line: int
    level: int
    text: str


class CodeBlockItem(TypedDict):
    start_line: int
    content: str
    language: Optional[str]


class TableItem(TypedDict):
    header: List[str]
    rows: List[List[str]]


class TextLinkItem(TypedDict):
    line: int
    text: str
    url: str


class ImageLinkItem(TypedDict):
    line: int
    alt_text: str
    url: str


class FootnoteItem(TypedDict):
    line: int
    id: str
    content: str


class InlineCodeItem(TypedDict):
    line: int
    code: str


class EmphasisItem(TypedDict):
    line: int
    text: str


class TaskItem(TypedDict):
    line: int
    text: str
    checked: bool


class HtmlBlockItem(TypedDict):
    line: int
    content: str


class HtmlInlineItem(TypedDict):
    line: int
    html: str


class Summary(TypedDict):
    headers: int
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


class MarkdownAnalysis(TypedDict):
    headers: dict[Literal["Header"], List[HeaderItem]]
    paragraphs: dict[Literal["Paragraph"], List[str]]
    blockquotes: dict[Literal["Blockquote"], List[str]]
    code_blocks: dict[Literal["Code block"], List[CodeBlockItem]]
    lists: dict[Literal["Ordered list",
                        "Unordered list"], List[List[ListItem]]]
    tables: dict[Literal["Table"], List[TableItem]]
    links: dict[Literal["Text link", "Image link"],
                List[Union[TextLinkItem, ImageLinkItem]]]
    footnotes: List[FootnoteItem]
    inline_code: List[InlineCodeItem]
    emphasis: List[EmphasisItem]
    task_items: List[TaskItem]
    html_blocks: List[HtmlBlockItem]
    html_inline: List[HtmlInlineItem]
    tokens_sequential: List[MarkdownToken]
    word_count: dict[Literal["word_count"], int]
    char_count: List[int]
    summary: Summary
