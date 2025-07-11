from typing import List, Literal, TypedDict, Optional


class Header(TypedDict):
    level: Literal[1, 2, 3, 4, 5, 6]
    line: int
    text: str


class CodeBlock(TypedDict):
    content: str
    language: str
    start_line: int


class TableRow(TypedDict):
    cells: List[str]


class Table(TypedDict):
    header: List[str]
    rows: List[TableRow]


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
    checked: Optional[bool]
    content: str
    id: int
    type: Literal[
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
        "html_block"
    ]
    url: Optional[str]


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
    header: List[Header]
    paragraph: List[str]
    blockquote: List[str]
    code_block: List[CodeBlock]
    table: List[Table]
    unordered_list: List[List[ListItem]]
    ordered_list: List[List[ListItem]]
    text_link: List[TextLink]
    image_link: List[ImageLink]
    footnotes: List[Footnote]
    inline_code: List[InlineCode]
    emphasis: List[Emphasis]
    task_items: List[TaskItem]
    html_inline: List[str]
    html_blocks: List[HtmlBlock]
    tokens_sequential: List[TokenSequential]
