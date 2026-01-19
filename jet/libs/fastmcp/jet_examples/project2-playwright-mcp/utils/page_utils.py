import re
from typing import Any, List, Dict, Literal, cast, TypedDict

# Possible tag kinds - you can extend this Literal as needed
TagKind = Literal[
    "generic", "div", "main", "navigation", "link", "img", "search",
    "textbox", "button", "combobox", "option", "paragraph", "heading",
    "article", "group", "complementary", "contentinfo"
]

class ExtractedElement(TypedDict, total=False):
    ref: str
    kind: TagKind
    original_tag: str                     # full tag before normalization
    text: str | None                      # main text content if present
    attributes: Dict[str, str]            # e.g. {"cursor": "pointer", "level": "3"}
    url: str | None                       # convenience field for /url
    children_count: int                   # how many direct children it had
    is_leaf: bool

def _parse_tag_key(key: str) -> tuple[str, str | None, str | None, Dict[str, str]]:
    """
    Robust parser for tag keys with flexible format:
        generic [ref=e1]
        link "About Us" [ref=e4] [cursor=pointer]
        button [active] [ref=btn1] [type=submit]
        combobox "Search language" [ref=e76]
    """
    print(f"DEBUG _parse_tag_key input: {key!r}")

    # 1. Find all bracketed attributes
    attributes: Dict[str, str] = {}
    ref_id: str | None = None

    bracket_matches = list(re.finditer(r'\[([^][]+)\]', key))
    print(f"DEBUG   found {len(bracket_matches)} bracket parts")

    for match in bracket_matches:
        part = match.group(1).strip()
        if '=' in part:
            k, v = [x.strip() for x in part.split('=', 1)]
            attributes[k] = v
        else:
            attributes[part] = "true"

        print(f"DEBUG     attr: {part}")

    if 'ref' in attributes:
        ref_id = attributes.pop('ref')

    # 2. Try to extract quoted text
    # We remove all [ ] parts to help find quoted string more reliably
    cleaned = re.sub(r'\[.*?\]', '', key).strip()
    quoted_match = re.search(r'"([^"]*)"', cleaned)

    text_content = quoted_match.group(1).strip() if quoted_match else None
    print(f"DEBUG   extracted text: {text_content!r}")

    # 3. Tag name = first word before any quote or bracket
    tag_match = re.match(r'^\s*(\w+)(?:\s+|$)', key)
    tag_part = tag_match.group(1) if tag_match else "unknown"

    print(f"DEBUG   tag_part: {tag_part!r} | ref_id: {ref_id}")
    print("---")

    return tag_part, text_content, ref_id, attributes

def extract_referenced_elements(
    data: Any,
    result: List[ExtractedElement] | None = None
) -> List[ExtractedElement]:
    """
    Recursively extracts all elements that contain [ref=xxx] into a flat list.
    Normalizes 'generic' → 'div'
    """
    if result is None:
        result = []

    if isinstance(data, dict):
        for key, value in data.items():
            tag_part, text_quote, ref_id, attrs = _parse_tag_key(key)

            if ref_id:
                element: ExtractedElement = {
                    "ref": ref_id,
                    "kind": "div" if tag_part.lower() == "generic" else cast(TagKind, tag_part),
                    "original_tag": key,
                    "attributes": attrs,
                    "text": text_quote,
                    "children_count": 0,
                    "is_leaf": False,
                }

                # Find /url
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "/url" in item:
                            element["url"] = item["/url"]
                            break

                # ── IMPROVED children count ──────────────────────────────
                # Count only dict children (most meaningful for this domain)
                if isinstance(value, list):
                    element["children_count"] = sum(1 for v in value if isinstance(v, dict))
                    element["is_leaf"] = element["children_count"] == 0

                # Fallback text when value is simple string and no quoted text
                if element.get("text") is None:
                    if isinstance(value, str):
                        element["text"] = value.strip()
                    elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                        element["text"] = value[0].strip()

                result.append(element)

            # Always recurse into dicts/lists
            if isinstance(value, (dict, list)):
                extract_referenced_elements(value, result)

            # ── NEW: Handle strings inside list that look like tagged elements ──
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and "[ref=" in item:
                        tag_part2, text_quote2, ref_id2, attrs2 = _parse_tag_key(item)
                        if ref_id2:
                            child_element: ExtractedElement = {
                                "ref": ref_id2,
                                "kind": "div" if tag_part2.lower() == "generic" else cast(TagKind, tag_part2),
                                "original_tag": item,
                                "attributes": attrs2,
                                "text": text_quote2,
                                "children_count": 0,
                                "is_leaf": True,  # most inline string refs are leaves
                            }
                            result.append(child_element)

    elif isinstance(data, list):
        for item in data:
            extract_referenced_elements(item, result)

    return result

# ── Convenience wrapper with sorting by ref id (e1, e2, ...) ────────────────
def extract_all_references_ordered(json_data: Any) -> List[ExtractedElement]:
    items = extract_referenced_elements(json_data)

    # Sort by ref id - assumes format e<number>
    def ref_key(item: ExtractedElement) -> int:
        try:
            return int(item["ref"][1:]) if "ref" in item and item["ref"][1:].isdigit() else 999_999
        except (ValueError, IndexError, KeyError):
            return 999_999

    items.sort(key=ref_key)
    return items