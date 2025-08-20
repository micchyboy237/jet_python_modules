from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


class FileInput(BaseModel):
    file_path: str = Field(...,
                           description="Path to the file (e.g., 'example.txt')")
    encoding: Literal["utf-8",
                      "ascii"] = Field("utf-8", description="File encoding")


class FileOutput(BaseModel):
    content: str = Field(..., description="File contents or error message")


class UrlInput(BaseModel):
    url: str = Field(..., description="URL to navigate to (e.g., 'https://example.com')",
                     pattern=r"^https?://")


class UrlOutput(BaseModel):
    title: str = Field(..., description="Page title or error message")
    nav_links: Optional[List[str]] = Field(
        None, description="List of links from the same server")
    text_content: Optional[str] = Field(
        None, description="All visible text content on the page")


class SummarizeTextInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_words: int = Field(
        100, description="Maximum number of words for the summary", ge=10, le=500)


class SummarizeTextOutput(BaseModel):
    summary: str = Field(..., description="Summarized text")
    word_count: int = Field(..., description="Number of words in the summary")
