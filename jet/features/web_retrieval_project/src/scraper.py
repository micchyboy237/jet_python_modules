from typing import Optional, Sequence
from bs4 import BeautifulSoup
import re
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document

def scrape_recursive_url(
    root_url: str,
    max_depth: Optional[int] = 2,
    exclude_dirs: Optional[Sequence[str]] = None,
    timeout: Optional[int] = 10
) -> list[Document]:
    """Recursively scrape a website starting from root_url up to max_depth."""
    def bs4_extractor(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        text = re.sub(r"\n\n+", "\n\n", soup.get_text()).strip()
        return text

    loader = RecursiveUrlLoader(
        url=root_url,
        max_depth=max_depth,
        extractor=bs4_extractor,
        exclude_dirs=exclude_dirs or (),
        timeout=timeout,
        prevent_outside=True
    )
    return loader.load()
