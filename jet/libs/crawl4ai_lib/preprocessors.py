import re

from bs4 import BeautifulSoup


def preprocess_for_schema_generation(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")  # or "html.parser"

    # 1. Remove worst offenders
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "iframe",
            "form",
            "input",
            "button",
            "svg",
            "path",
            "symbol",
            "meta",
            "link",
            "head",
        ]
    ):
        tag.decompose()

    # 2. Remove common noise classes/ids (customize per site!)
    noise_classes = [
        "advert",
        "ad-",
        "banner",
        "cookie",
        "popup",
        "modal",
        "sidebar",
        "footer",
        "nav",
        "menu",
        "header",
        "related",
        "recommended",
        "comments",
    ]
    for elem in soup.find_all(class_=re.compile("|".join(noise_classes), re.I)):
        elem.decompose()

    for elem in soup.find_all(id=re.compile("|".join(noise_classes), re.I)):
        elem.decompose()

    # 3. Optional: keep only main content areas (very effective)
    # Try common article containers first
    # main_candidates = (
    #     soup.find("main")
    #     or soup.find("article")
    #     or soup.find(attrs={"role": "main"})
    #     or soup.find(
    #         "div", class_=re.compile(r"(content|post|article|entry|body)", re.I)
    #     )
    #     or soup.body
    # )

    # if main_candidates:
    #     soup = BeautifulSoup(str(main_candidates), "lxml")

    # 4. Clean up excessive whitespace & empty tags
    cleaned = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    # But keep structure → better to return str(soup) instead of plain text
    return cleaned
