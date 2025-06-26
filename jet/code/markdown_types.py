from typing import List, Optional, TypedDict, Literal
from typing import TypedDict, List, Optional, Literal, Union
from pathlib import Path

# Reuse MarkdownToken from previous response


class ListItem(TypedDict, total=False):
    text: str
    task_item: bool
    # checked is optional as not all list items have it
    checked: Optional[bool]


class CodeMeta(TypedDict, total=False):
    language: Optional[str]
    code_type: Optional[Literal["indented"]]


class TableMeta(TypedDict):
    header: List[str]
    rows: List[List[str]]


class ListMeta(TypedDict):
    items: List[ListItem]


MetaType = Union[ListMeta, CodeMeta, TableMeta, dict]

ContentType = Literal[
    "header",
    "paragraph",
    "blockquote",
    "code",
    "table",
    "unordered_list",
    "ordered_list",
    "html_block"
]


class MarkdownToken(TypedDict):
    type: ContentType
    content: str  # content is always present in the JSON, can be empty string
    level: Optional[int]
    meta: MetaType
    line: int

# Typed dicts for analyze_markdown components


class SummaryDict(TypedDict):
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


class HeaderItemDict(TypedDict):
    line: int
    level: Literal[1, 2, 3, 4, 5, 6]
    text: str


class HeadersDict(TypedDict):
    header: List[HeaderItemDict]


class ParagraphsDict(TypedDict):
    paragraph: List[str]


class BlockquotesDict(TypedDict):
    blockquote: List[str]


class CodeBlockItemDict(TypedDict):
    start_line: int
    content: str
    language: str


class CodeBlocksDict(TypedDict):
    code_block: List[CodeBlockItemDict]


class ListItemDict(TypedDict):
    text: str
    task_item: bool
    checked: Optional[bool]


class ListsDict(TypedDict):
    unordered_list: List[List[ListItemDict]]
    ordered_list: List[List[ListItemDict]]


class TableItemDict(TypedDict):
    header: List[str]
    rows: List[List[str]]


class TablesDict(TypedDict):
    table: List[TableItemDict]


class LinkItemDict(TypedDict):
    line: int
    text: Optional[str]
    url: str
    alt_text: Optional[str]


class LinksDict(TypedDict):
    text_link: List[LinkItemDict]
    image_link: List[LinkItemDict]


class FootnoteItemDict(TypedDict):
    line: int
    id: str
    content: str


class InlineCodeItemDict(TypedDict):
    line: int
    code: str


class EmphasisInnerDict(TypedDict):
    line: int
    text: str


class EmphasisItemDict(TypedDict):
    emphasis: EmphasisInnerDict
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


class WordCountDict(TypedDict):
    word_count: int


class CharCountDict(TypedDict):
    char: int


class MarkdownAnalysis(TypedDict):
    summary: SummaryDict
    headers: HeadersDict
    paragraphs: ParagraphsDict
    blockquotes: BlockquotesDict
    code_blocks: CodeBlocksDict
    lists: ListsDict
    tables: TablesDict
    links: LinksDict
    footnotes: List[FootnoteItemDict]
    inline_code: List[InlineCodeItemDict]
    emphasis: List[EmphasisItemDict]
    task_items: List[TaskItemDict]
    html_blocks: List[HtmlBlockItemDict]
    html_inline: List[HtmlInlineItemDict]
    tokens_sequential: List[TokenSequentialItemDict]
    word_count: WordCountDict
    char_count: CharCountDict
