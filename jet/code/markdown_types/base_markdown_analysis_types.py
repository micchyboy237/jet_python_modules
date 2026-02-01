from typing import Literal, TypedDict


class Header(TypedDict):
    level: Literal[1, 2, 3, 4, 5, 6]
    line: int
    text: str


class CodeBlock(TypedDict):
    content: str
    language: str
    start_line: int


class TableRow(TypedDict):
    cells: list[str]


class Table(TypedDict):
    header: list[str]
    rows: list[TableRow]


class ListItem(TypedDict):
    checked: bool
    task_item: bool
    text: str


class TextLink(TypedDict):
    line: int
    text: str
    url: str


class ImageLink(TypedDict):
    alt_text: str
    line: int
    url: str


class Footnote(TypedDict):
    content: str
    id: str
    line: int


class InlineCode(TypedDict):
    code: str
    line: int


class Emphasis(TypedDict):
    line: int
    text: str


class TaskItem(TypedDict):
    checked: bool
    line: int
    text: str


class HtmlBlock(TypedDict):
    content: str
    line: int


class TokenSequential(TypedDict):
    checked: bool | None
    content: str
    id: int
    type: Literal[
        "head",
        "header",
        "paragraph",
        "blockquote",
        "code_block",
        "table",
        "unordered_list",
        "ordered_list",
        "text_link",
        "image_link",
        "footnote",
        "inline_code",
        "emphasis",
        "task_item",
        "html_inline",
        "html_block",
    ]
    url: str | None
    meta: dict | None


class HeaderCounts(TypedDict):
    h1: int
    h2: int
    h3: int
    h4: int
    h5: int
    h6: int


class Analysis(TypedDict):
    headers: int
    header_counts: HeaderCounts
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


class BaseMarkdownAnalysis(TypedDict):
    analysis: Analysis
    header: list[Header]
    paragraph: list[str]
    blockquote: list[str]
    code_block: list[CodeBlock]
    table: list[Table]
    unordered_list: list[ListItem]
    ordered_list: list[ListItem]
    text_link: list[TextLink]
    image_link: list[ImageLink]
    footnotes: list[Footnote]
    inline_code: list[InlineCode]
    emphasis: list[Emphasis]
    task_items: list[TaskItem]
    html_inline: list[str]
    html_blocks: list[HtmlBlock]
    tokens_sequential: list[TokenSequential]
