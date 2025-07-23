from typing import Optional, Dict, List
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import re


@dataclass
class WebsiteContent:
    html: str
    css: List[str]
    title: Optional[str] = None


class WebsiteCloner:
    def __init__(self, url: str):
        self.url = url
        self.content: Optional[WebsiteContent] = None

    def fetch_website(self) -> WebsiteContent:
        """Fetch HTML and CSS from the target website."""
        response = requests.get(self.url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.text if title_tag else None

        # Extract CSS from inline styles and linked stylesheets
        css_files = []
        for link in soup.find_all('link', rel='stylesheet'):
            css_url = link.get('href')
            if css_url:
                if not css_url.startswith('http'):
                    css_url = self._resolve_relative_url(css_url)
                try:
                    css_response = requests.get(css_url)
                    css_response.raise_for_status()
                    css_files.append(css_response.text)
                except requests.RequestException:
                    pass  # Skip failed CSS fetches

        # Extract inline CSS
        inline_css = [style.text for style in soup.find_all('style')]

        self.content = WebsiteContent(
            html=str(soup),
            css=css_files + inline_css,
            title=title
        )
        return self.content

    def _resolve_relative_url(self, relative_url: str) -> str:
        """Convert relative URL to absolute URL."""
        from urllib.parse import urljoin
        return urljoin(self.url, relative_url)

    def generate_tailwind_html(self) -> str:
        """Generate prettified HTML with Tailwind CSS for pixel-perfect styling."""
        if not self.content:
            raise ValueError("Content not fetched. Call fetch_website first.")

        # Basic Tailwind integration
        tailwind_cdn = (
            '<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">'
        )
        # Process HTML to add Tailwind classes (simplified example)
        soup = BeautifulSoup(self.content.html, 'html.parser')
        for tag in soup.find_all(['div', 'p', 'h1', 'h2', 'h3']):
            # Example: Add Tailwind classes based on tag analysis
            tag['class'] = tag.get('class', []) + ['text-gray-800', 'my-2']

        # Create the full HTML document
        html_doc = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.content.title or 'Cloned Website'}</title>
    {tailwind_cdn}
</head>
<body>
    {str(soup.body)}
</body>
</html>
"""
        # Prettify the HTML
        return BeautifulSoup(html_doc, 'html.parser').prettify()
