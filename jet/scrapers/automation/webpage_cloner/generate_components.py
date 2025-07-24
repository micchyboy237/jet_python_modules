from typing import Dict, List, TypedDict, Optional
import re
import os
import shutil
import tempfile
import subprocess
import cssutils
import urllib.request
import logging
from bs4 import BeautifulSoup, Comment, Tag
from pathlib import Path
from urllib.parse import urljoin, urlparse
from jet.logger import logger
from jet.transformers.formatters import format_html


class Component(TypedDict):
    name: str
    html: str
    styles: str


PRETTIER_CONFIG = """\
{
  "singleQuote": true,
  "trailingComma": "es5",
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true
}
"""

COMPONENT_CODE_TEMPLATE = """\
const {component_name} = () => {{
  return (
    {component_html}
  );
}};
window.{component_name} = {component_name};
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Components Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.development.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.development.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.25.6/babel.min.js"></script>
    {css_links}
    {font_links}
    {js_links}
</head>
<body>
    <div id="root"></div>
    {component_scripts}
    <script type="text/babel">
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<window.App />);
    </script>
</body>
</html>
"""


def parse_style_to_object(style: str) -> str:
    """Convert a CSS style string to a JavaScript object string for React using cssutils."""
    if not style:
        return "{}"
    cssutils.ser.prefs.useMinified()
    cssutils.log.setLevel(logging.ERROR)
    sheet = cssutils.parseString(f"dummy {{ {style} }}", validate=False)
    style_dict: Dict[str, str] = {}
    try:
        for rule in sheet:
            if rule.type == rule.STYLE_RULE:
                for property in rule.style:
                    prop = property.name
                    camel_prop = "".join(
                        word.capitalize() if i > 0 else word
                        for i, word in enumerate(prop.split("-"))
                    )
                    value = property.value
                    if not value:
                        logger.debug(
                            f"Skipping empty value for property: {prop}")
                        continue
                    if prop in {"flex", "flex-direction", "flex-grow", "flex-basis", "grid", "grid-gap",
                                "grid-template-columns", "grid-auto-flow", "justify-content", "align-items",
                                "flex-wrap", "gap", "transform", "transition", "transition-delay", "align-self",
                                "inline-size", "block-size", "animation"}:
                        style_dict[camel_prop] = f"'{value}'"
                    elif prop == "font-color":
                        style_dict["fontColor"] = f"'{value}'"
                    elif prop == "x-border-bottom":
                        style_dict["borderBottom"] = f"'{value}'"
                    elif value.startswith("linear-gradient"):
                        style_dict[camel_prop] = f"'{value}'"
                    elif value.startswith("calc") or "rem" in value or value in {"max-content", "unset", "inline-flex"}:
                        style_dict[camel_prop] = f"'{value}'"
                    elif prop in {"border-radius", "border", "border-bottom"} and value == "inherit":
                        style_dict[camel_prop] = "'inherit'"
                    elif prop == "vertical-align" and value in {"base", "center"}:
                        style_dict[camel_prop] = f"'{value}'"
                    elif prop == "unicode-bidi" and value == "isolate":
                        style_dict[camel_prop] = "'isolate'"
                    elif prop == "text-decoration" and value.startswith("underline"):
                        style_dict[camel_prop] = f"'{value}'"
                    elif "rgb(" in value and not re.match(r'rgb\(\d+,\s*\d+,\s*\d+\)', value):
                        style_dict[camel_prop] = f"'{value}'"
                    elif prop == "box-shadow" and value.startswith("rgb"):
                        style_dict[camel_prop] = f"'{value}'"
                    else:
                        no_quote_values = {"flex", "block",
                                           "inline", "none", "inherit", "initial"}
                        if value in no_quote_values:
                            style_dict[camel_prop] = value
                        else:
                            try:
                                float(value)
                                style_dict[camel_prop] = value
                            except ValueError:
                                style_dict[camel_prop] = f"'{value}'"
    except Exception as e:
        logger.error(f"Error parsing CSS: {e}")
        return "{}"
    style_items = [f"{key}: {value}" for key, value in style_dict.items()]
    logger.debug(f"Parsed style object: {{ {', '.join(style_items)} }}")
    return "{" + ", ".join(style_items) + "}"


def download_font(url: str, dest_path: Path, base_url: str = "") -> bool:
    """Download a font file from a URL to the destination path."""
    if not base_url:
        logger.warning(
            f"Base URL not provided; cannot resolve font URL: {url}")
        return False
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme and url.startswith("/"):
            parsed_base = urlparse(base_url)
            font_url = f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
        elif not parsed_url.scheme:
            font_url = urljoin(base_url.rstrip("/"), url.lstrip("/"))
        else:
            font_url = url
        parsed_url = urlparse(font_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.warning(f"Invalid font URL after resolving: {font_url}")
            return False
        font_path = dest_path / Path(parsed_url.path).name
        font_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(font_url) as response:
            if response.status != 200:
                logger.warning(
                    f"Failed to download font from {font_url}: HTTP {response.status}")
                return False
            content = response.read()
            if len(content) == 0:
                logger.warning(f"Downloaded font from {font_url} is empty")
                return False
            with open(font_path, "wb") as f:
                f.write(content)
            logger.success(f"Downloaded font to {font_path} from {font_url}")
            return True
    except Exception as e:
        logger.error(f"Error downloading font from {font_url}: {e}")
        return False


def extract_fonts_from_css(css_content: str, assets_dir: Path, base_url: str = "") -> List[str]:
    """Extract and download font files from CSS @font-face rules."""
    font_urls = []
    try:
        cssutils.log.setLevel(logging.ERROR)
        sheet = cssutils.parseString(css_content, validate=False)
        for rule in sheet:
            if rule.type == rule.FONT_FACE_RULE:
                for property in rule.style:
                    if property.name == "src":
                        urls = re.findall(r'url\((.*?)\)', property.value)
                        for url in urls:
                            url = url.strip().strip("'\"")
                            if url and url.endswith((".woff", ".woff2", ".ttf", ".otf")):
                                if download_font(url, assets_dir, base_url):
                                    font_urls.append(
                                        f"./assets/{Path(urlparse(url).path).name}")
                                else:
                                    font_urls.append(url)
    except Exception as e:
        logger.error(f"Error parsing CSS for fonts: {e}")
    return font_urls


def format_with_prettier(content: str, parser: str, config_path: str, file_suffix: str) -> str:
    """Format content using Prettier CLI with fallback to original content."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=file_suffix, delete=False, encoding="utf-8") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        result = subprocess.run(
            ["prettier", "--config",
                str(config_path), "--parser", parser, temp_file_path],
            capture_output=True, text=True, check=True
        )
        formatted_content = result.stdout
        return formatted_content
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Failed to format with Prettier (parser: {parser}): {e}, stderr: {e.stderr}")
        return content
    except FileNotFoundError:
        logger.warning(f"Prettier CLI not found for {parser} formatting")
        return content
    finally:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(
                    f"Failed to delete temporary file {temp_file_path}: {e}")


DEFAULT_STYLES: Dict[str, str] = {
    "a": "color: blue; text-decoration: underline;"
}


def generate_component_name(tag: Tag, idx: int, seen_identifiers: Dict[str, int]) -> tuple[str, str]:
    """Generate a unique component name based on ID or tag name."""
    if tag.get("id"):
        base_name = tag["id"]
        component_name = ''.join(word.capitalize()
                                 for word in base_name.split('-'))
        identifier = base_name
        logger.debug(
            f"Using ID '{base_name}' for component name: {component_name}")
    else:
        base_name = tag.name
        component_name = base_name.capitalize()
        identifier = base_name
        logger.debug(
            f"Using tag name '{base_name}' for component name: {component_name}")
    if base_name in seen_identifiers:
        seen_identifiers[base_name] += 1
        return f"{component_name}{seen_identifiers[base_name]}", identifier
    else:
        seen_identifiers[base_name] = 0
        return component_name, identifier


def is_nested_tag(tag: Tag, processed_tags: List[Tag], target_tags: List[str]) -> bool:
    """Check if a tag is a child of any processed tag of a target type, allowing top-level target tags."""
    if not isinstance(tag, Tag):
        logger.debug(f"Skipping non-Tag object: {type(tag)}")
        return True
    if tag.name in target_tags and tag.get("id"):
        logger.debug(f"Allowing tag with ID: {tag.get('id')} ({tag.name})")
        return False
    for processed_tag in processed_tags:
        if processed_tag.name in target_tags and tag in processed_tag.find_all(recursive=True):
            logger.debug(
                f"Skipping nested tag: {tag.name} with class {tag.get('class', [])} inside {processed_tag.name}")
            return True
    return False


def generate_react_components(html: str, output_dir: str, component_code_template: str = COMPONENT_CODE_TEMPLATE, base_url: str = "") -> tuple[List[Component], List[str]]:
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    assets_dir = output_dir_path / "assets"
    components_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    original_html_path = output_dir_path / "original.html"
    with open(original_html_path, "w", encoding="utf-8") as f:
        f.write(format_html(html))
    logger.success(f"Saved original HTML at {original_html_path}")
    if not html.strip().endswith("</html>"):
        logger.warning(
            "Original HTML is incomplete; may result in missing components. Check HTML capture process in run_grok_webpage_cloner.py.")
    prettier_config = PRETTIER_CONFIG
    prettier_config_path = components_dir / ".prettierrc"
    with open(prettier_config_path, "w", encoding="utf-8") as f:
        f.write(prettier_config.rstrip())
    logger.success(f"Generated Prettier config file at {prettier_config_path}")
    font_urls = []
    for file in assets_dir.glob("*"):
        if file.suffix in [".css", ".js", ".png", ".jpeg", ".svg", ".webp"]:
            if file.stat().st_size == 0:
                logger.warning(
                    f"Skipping zero-byte file: {file} (possible download failure)")
                continue
            logger.debug(f"Asset {file} remains in assets directory")
            if file.suffix == ".css":
                with open(file, "r", encoding="utf-8") as f:
                    css_content = f.read()
                    font_urls.extend(extract_fonts_from_css(
                        css_content, assets_dir, base_url))
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        logger.error(f"Failed to parse original HTML: {e}")
        soup = BeautifulSoup("", "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    root_div = soup.find("div", id="root")
    noscript = soup.find("noscript")
    noscript_content = noscript.get_text(strip=True) if noscript else ""
    if root_div and not root_div.get_text(strip=True) and soup.find("script", src=True):
        logger.warning(
            "Detected single-page application (SPA) with empty <div id='root'>. "
            "Content may be dynamically loaded by JavaScript. To capture rendered content, "
            "modify run_grok_webpage_cloner.py to use a headless browser with page.content() "
            "after waiting for JavaScript execution (e.g., waitUntil: 'networkidle2')."
        )
    components: List[Component] = []
    target_tags = ["header", "footer", "section", "div",  # "a",
                   "article", "aside", "main", "nav", "button"]
    tags = [tag for tag in soup.find_all(target_tags) if isinstance(tag, Tag)]
    seen_identifiers: Dict[str, int] = {}
    processed_tags: List[Tag] = []
    logger.debug("Initialized processed_tags for tracking processed elements")

    def replace_style(match):
        style_str = match.group(1)
        style_object = parse_style_to_object(style_str)
        logger.debug(f"Converting style: {style_str} to {{ {style_object} }}")
        return f'style={{ {style_object} }}'

    def update_asset_references(html_content: str) -> str:
        """Update asset references in HTML to point to the assets directory."""
        for ext in [".png", ".jpeg", ".svg", ".webp"]:
            html_content = re.sub(
                rf'src="components/([^"]*{ext})"', rf'src="assets/\1"', html_content)
        return html_content

    def remove_nested_target_tags(tag: Tag, target_tags: List[str]) -> str:
        """Remove nested target tags from the tag's HTML to avoid duplication."""
        tag_copy = tag.__copy__()
        for nested_tag in tag_copy.find_all(target_tags, recursive=True):
            if nested_tag in processed_tags or is_nested_tag(nested_tag, processed_tags, target_tags):
                nested_tag.decompose()
        return str(tag_copy)

    for idx, tag in enumerate(tags):
        if is_nested_tag(tag, processed_tags, target_tags):
            continue
        if not tag.get_text(strip=True) and not any(child for child in tag.children if child.name):
            logger.debug(
                f"Skipping empty tag: {tag.name} with class {tag.get('class', [])}")
            continue
        component_name, identifier = generate_component_name(
            tag, idx, seen_identifiers)
        styles = tag.get("style", "").strip()
        if not styles:
            css_files = [assets_dir / f for f in os.listdir(
                assets_dir) if f.endswith(".css") and os.path.getsize(assets_dir / f) > 0]
            class_styles = []
            if tag.get("id"):
                for css_file in css_files:
                    with open(css_file, "r", encoding="utf-8") as f:
                        css_content = f.read()
                        pattern = rf"#{re.escape(tag['id'])}\s*{{([^}}]*)}}"
                        matches = re.findall(pattern, css_content, re.DOTALL)
                        for match in matches:
                            cleaned_style = match.strip()
                            if cleaned_style and cleaned_style not in class_styles:
                                class_styles.append(cleaned_style)
                                logger.debug(
                                    f"Extracted ID-based style for #{tag['id']}: {cleaned_style}")
            styles = ";".join(class_styles).strip()
        if not styles and tag.name in DEFAULT_STYLES:
            styles = DEFAULT_STYLES[tag.name]
            logger.debug(f"Applied default styles for {tag.name}: {styles}")
        component_html = remove_nested_target_tags(tag, target_tags)
        component_html = re.sub(r'\bclass=', 'className=', component_html)
        component_html = re.sub(r'\bclassName="[^"]*"\s*className="[^"]*"',
                                f'className="{identifier}"', component_html, flags=re.IGNORECASE)
        component_html = component_html.replace(
            'autocomplete=', 'autoComplete=')
        component_html = component_html.replace('for=', 'htmlFor=')
        component_html = component_html.replace('onclick=', 'onClick=')
        component_html = update_asset_references(component_html)
        if styles:
            component_html = re.sub(
                r'style="([^"]*)"', replace_style, component_html)
        elif 'style="' in component_html:
            component_html = re.sub(
                r'style="([^"]*)"', replace_style, component_html)
        if not tag.get("class") and not tag.get("className") and not tag.get("id"):
            tag_name = tag.name
            component_html = re.sub(
                rf'<{tag_name}\b', f'<{tag_name} className="{identifier}"', component_html, 1)
        components.append({
            "name": component_name,
            "html": component_html,
            "styles": styles
        })
        processed_tags.append(tag)
        component_code = component_code_template.format(
            component_name=component_name,
            css_import="",
            component_html=component_html
        )
        formatted_component_code = format_with_prettier(
            component_code, "babel", str(prettier_config_path), ".jsx"
        )
        component_path = components_dir / f"{component_name}.jsx"
        with open(component_path, "w", encoding="utf-8") as f:
            f.write(formatted_component_code.rstrip())
        logger.success(f"Generated React component file at {component_path}")
        if styles:
            css_class_name = identifier
            css_content = f".{css_class_name} {{{styles}}}" if not tag.get(
                "id") else f"#{css_class_name} {{{styles}}}"
            logger.debug(
                f"Generating CSS file for {component_name} with content: {css_content}")
            formatted_styles = format_with_prettier(
                css_content, "css", str(prettier_config_path), ".css"
            )
            css_path = components_dir / f"{component_name}.css"
            with open(css_path, "w", encoding="utf-8") as f:
                f.write(formatted_styles.rstrip())
            logger.success(f"Generated CSS file at {css_path}")
        else:
            logger.debug(
                f"No specific styles generated for {component_name}; relying on copied CSS files")

    js_files = soup.find_all("script", src=True)
    js_includes = []
    for js in js_files:
        src = js.get("src")
        if src:
            if src.startswith("http"):
                js_includes.append(f'<script src="{src}"></script>')
            elif (assets_dir / Path(src).name).exists():
                js_includes.append(
                    f'<script src="./assets/{Path(src).name}"></script>')
            else:
                logger.warning(
                    f"JavaScript dependency {src} not found in assets directory; may affect dynamic content rendering")
    inline_scripts = [script.get_text() for script in soup.find_all(
        "script") if not script.get("src")]
    js_includes.extend(
        [f"<script>{script}</script>" for script in inline_scripts if script.strip()])
    app_jsx_content = """\
const App = () => {{
  return (
    <div>
      {component_renders}
      {noscript_content}
    </div>
  );
}};
window.App = App;
"""
    component_renders = "\n      ".join(
        f'<window.{component["name"]} />'
        for component in components
    ) or "<div>Generated content is empty; original page may rely on JavaScript rendering. To see the full content, ensure run_grok_webpage_cloner.py captures the rendered DOM after JavaScript execution (e.g., use Puppeteer with page.content() and waitUntil: 'networkidle2').</div>"
    noscript_content = f"<div>{noscript_content}</div>" if noscript_content else ""
    try:
        formatted_app_jsx = format_with_prettier(
            app_jsx_content.format(
                component_renders=component_renders, noscript_content=noscript_content),
            "babel",
            str(prettier_config_path),
            ".jsx"
        )
    except Exception as e:
        logger.error(f"Error formatting App.jsx: {e}")
        logger.debug(f"Problematic App.jsx content: {app_jsx_content}")
        formatted_app_jsx = app_jsx_content.format(
            component_renders=component_renders, noscript_content=noscript_content)
    app_path = components_dir / "App.jsx"
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(formatted_app_jsx.rstrip())
    logger.success(f"Generated App.jsx file at {app_path}")
    return components, font_urls


def generate_entry_point(components: List[Component], output_dir: str, font_urls: List[str], html_template: str = HTML_TEMPLATE, base_url: str = "") -> None:
    """Generate an index.html file to preview all React components using App.jsx."""
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    assets_dir = output_dir_path / "assets"
    components_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    prettier_config = PRETTIER_CONFIG
    prettier_config_path = components_dir / ".prettierrc"
    with open(prettier_config_path, "w", encoding="utf-8") as f:
        f.write(prettier_config.rstrip())
    logger.success(f"Generated Prettier config file at {prettier_config_path}")
    css_links = "\n".join(
        f'<link rel="stylesheet" href="./components/{component["name"]}.css">'
        for component in components
        if component["styles"] and (components_dir / f"{component["name"]}.css").exists()
    )
    css_files = [f for f in assets_dir.glob(
        "*.css") if f.stat().st_size > 0]
    css_links += "\n".join(
        f'<link rel="stylesheet" href="./assets/{f.name}">'
        for f in css_files
        if f.name not in [f"{component['name']}.css" for component in components]
    )
    font_links = "\n".join(
        f'<link rel="stylesheet" href="{url}">' if url.startswith(
            "http") else f'<link rel="stylesheet" href="./assets/{Path(urlparse(url).path).name}">'
        for url in set(font_urls)
    )
    component_scripts = "\n".join(
        f'<script type="text/babel" src="./components/{component["name"]}.jsx"></script>'
        for component in components
    )
    component_scripts += '\n<script type="text/babel" src="./components/App.jsx"></script>'
    soup = BeautifulSoup(
        open(output_dir_path / "original.html"), "html.parser")
    js_links = "\n".join(
        f'<script src="{js.get("src")}"></script>'
        for js in soup.find_all("script", src=True)
        if js.get("src").startswith("http") or (assets_dir / Path(js.get("src")).name).exists()
    )
    inline_scripts = [script.get_text() for script in soup.find_all(
        "script") if not script.get("src")]
    js_links += "\n".join(
        [f"<script>{script}</script>" for script in inline_scripts if script.strip()])
    html_content = html_template.format(
        css_links=css_links,
        font_links=font_links,
        js_links=js_links,
        component_scripts=component_scripts
    )
    formatted_html = format_with_prettier(
        html_content, "html", str(prettier_config_path), ".html"
    )
    index_path = output_dir_path / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(formatted_html.rstrip())
    logger.success(f"Generated entry point file at {index_path}")
    logger.warning(
        "To view index.html, serve it via a local web server (e.g., 'python -m http.server 8000') "
        "instead of opening directly with file:// to avoid module resolution errors."
    )
