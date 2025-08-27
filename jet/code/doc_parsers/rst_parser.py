# rst_parser.py
from typing import Any, Dict, List, Optional
from docutils import nodes
from docutils.core import publish_doctree


def node_tag(node: nodes.Node) -> str:
    """Return a stable tag name for a node."""
    return getattr(node, "tagname", None) or node.__class__.__name__


def node_to_dict(node: nodes.Node) -> Dict[str, Any]:
    """
    Convert a docutils node into a serializable dict that captures:
      - type (node.tagname or node.__class__.__name__)
      - text (short textual capture for textual nodes)
      - attrs (useful attributes)
      - children (list of child node dicts)
    """
    tag = node_tag(node)
    out: Dict[str, Any] = {"type": tag,
                           "text": None, "children": [], "attrs": {}}

    # capture short text for nodes that commonly carry text content
    try:
        if tag in {
            "paragraph",
            "literal_block",
            "title",
            "rubric",
            "line",
            "line_block",
            "block_quote",
            "reference",
            "field",
            "field_name",
            "field_body",
            "system_message",
            "footnote",
            "citation",
            "definition",
            "term",
        }:
            # Use astext() to get aggregate text for this node
            out["text"] = node.astext()
        elif tag == "table":
            out["text"] = ""
    except Exception:
        out["text"] = None

    # capture some attributes that are commonly useful
    attrs: Dict[str, Any] = {}
    if hasattr(node, "names") and node.names:
        attrs["names"] = list(node.names)
    if getattr(node, "rawsource", None):
        attrs["rawsource"] = node.rawsource
    if getattr(node, "classes", None):
        attrs["classes"] = list(node.classes)
    if getattr(node, "ids", None):
        attrs["ids"] = list(node.ids)
    for key in ("name", "refuri", "refid", "label", "language"):
        try:
            val = node.get(key)
            if val:
                attrs[key] = val
        except Exception:
            # some node classes don't implement get()
            pass

    out["attrs"] = attrs

    # Recursively process children, but collapse Text nodes into parent text when appropriate
    for child in node.children:
        # skip plain Text nodes (their text is included in astext())
        if isinstance(child, nodes.Text):
            continue
        # skip comment nodes entirely (tests expect comments to be ignored)
        if isinstance(child, nodes.comment):
            continue
        # Special handling for tables
        if isinstance(child, nodes.table):
            out["children"].append(_table_to_dict(child))
            continue
        # recurse
        out_child = node_to_dict(child)
        out["children"].append(out_child)

    return out


def _table_to_dict(table_node: nodes.table) -> Dict[str, Any]:
    """
    Convert a docutils table node into a dict with rows->cells (text).
    Supports basic grid/simple tables as produced by docutils.
    """
    table = {"type": "table", "attrs": {}, "children": []}
    # collect rows: rows appear as nodes.row under tbody/tgroup
    rows: List[List[str]] = []
    for row in table_node.traverse(nodes.row):
        cells: List[str] = []
        # each entry in a row is a nodes.entry
        for entry in row.traverse(nodes.entry, include_self=False):
            cells.append(entry.astext())
        if cells:
            rows.append(cells)
    table["children"] = rows
    return table


def _is_paragraph_blockquote_like(node: nodes.Node) -> bool:
    """
    Heuristic: detect paragraphs that use '>' as blockquote marker (non-standard reST).
    This is included to satisfy tests that expect block_quote for '>'-prefixed paragraphs.
    """
    if isinstance(node, nodes.paragraph):
        text = node.astext().lstrip()
        return text.startswith("> ")
    return False


def _make_block_quote_from_para(node: nodes.Node) -> Dict[str, Any]:
    """
    Convert a paragraph node whose text starts with '> ' into a block_quote dict node.
    Strips the leading '> ' from each line.
    """
    raw = node.astext()
    # strip a single leading '>' plus optional space per line
    lines = raw.splitlines()
    stripped_lines = []
    for ln in lines:
        l = ln.lstrip()
        if l.startswith("> "):
            stripped_lines.append(l[2:])
        elif l.startswith(">"):
            stripped_lines.append(l[1:])
        else:
            stripped_lines.append(ln)
    inner_para_text = "\n".join(stripped_lines).strip()
    return {"type": "block_quote", "text": inner_para_text, "children": [{"type": "paragraph", "text": inner_para_text, "children": [], "attrs": {}}], "attrs": {}}


def parse_rst_to_blocks(rst_text: str) -> Dict[str, Any]:
    """
    Parse an RST string with docutils and return a serializable dict representing
    the document tree with block nodes.

    Additional behaviors:
    - Skip comment nodes entirely.
    - If docutils does not emit a 'section' wrapper but there is a top-level title
      followed by other nodes, group them into a synthetic 'section' node (title + its following content).
    - Convert '>'-prefixed paragraphs into block_quote nodes via a heuristic (to match tests that use '>' style).
    """
    doctree = publish_doctree(rst_text)
    result = {"type": "document", "children": []}

    children = [
        c for c in doctree.children if not isinstance(c, nodes.comment)]
    i = 0
    n = len(children)
    while i < n:
        child = children[i]
        tag = node_tag(child)

        # If docutils didn't create a section node but there's a top-level title followed by content,
        # create a synthetic section node grouping the title and following nodes until the next title.
        if tag == "title":
            # collect title + following nodes until next title (or end)
            j = i + 1
            group_nodes = [child]
            while j < n and node_tag(children[j]) != "title":
                group_nodes.append(children[j])
                j += 1
            # build a section dict
            section = {"type": "section", "text": None,
                       "attrs": {}, "children": []}
            # add title dict first
            section["children"].append(node_to_dict(child))
            # add remaining children in the group (converted)
            for g in group_nodes[1:]:
                # special-case paragraph starting with '>' -> block_quote
                if _is_paragraph_blockquote_like(g):
                    section["children"].append(_make_block_quote_from_para(g))
                else:
                    section["children"].append(node_to_dict(g))
            result["children"].append(section)
            i = j
            continue

        # If paragraph uses '>' as blockquote marker, convert to block_quote at top-level
        if _is_paragraph_blockquote_like(child):
            result["children"].append(_make_block_quote_from_para(child))
            i += 1
            continue

        # Normal case: append converted child
        result["children"].append(node_to_dict(child))
        i += 1

    return result
