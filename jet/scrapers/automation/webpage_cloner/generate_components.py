from typing import Dict, List, TypedDict, Optional
import re
import os
import shutil
import tempfile
import subprocess
import cssutils
from bs4 import BeautifulSoup, Comment
from pathlib import Path
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
    sheet = cssutils.parseString(f"dummy {{ {style} }}")
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
        logger.error(f"Error parsing CSS with cssutils: {e}")
        return "{}"
    style_items = [f"{key}: {value}" for key, value in style_dict.items()]
    logger.debug(f"Parsed style object: {{ {', '.join(style_items)} }}")
    return "{" + ", ".join(style_items) + "}"


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


def generate_react_components(html: str, output_dir: str, component_code_template: str = COMPONENT_CODE_TEMPLATE) -> List[Component]:
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    # Save original HTML as original.html
    original_html_path = output_dir_path / "original.html"
    with open(original_html_path, "w", encoding="utf-8") as f:
        f.write(format_html(html))
    logger.success(f"Saved original HTML at {original_html_path}")

    prettier_config = PRETTIER_CONFIG
    prettier_config_path = components_dir / ".prettierrc"
    with open(prettier_config_path, "w", encoding="utf-8") as f:
        f.write(prettier_config.rstrip())
    logger.success(f"Generated Prettier config file at {prettier_config_path}")

    # Copy assets/iana_website.css to components directory
    assets_dir = output_dir_path / "assets"
    if assets_dir.exists():
        css_files = [f for f in assets_dir.glob("*.css")]
        for css_file in css_files:
            dest_path = components_dir / css_file.name
            shutil.copy(css_file, dest_path)
            logger.success(f"Copied {css_file} to {dest_path}")

    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    components: List[Component] = []
    seen_html_content = set()
    tags = soup.find_all(["header", "footer", "section", "div", "a"])
    seen_class_names = {}

    def is_nested_tag(tag, processed_tags):
        """Check if a tag is a child of any processed tag."""
        for processed_tag in processed_tags:
            if tag in processed_tag.find_all(recursive=True):
                return True
        return False

    processed_tags = []

    def replace_style(match):
        style_str = match.group(1)
        style_object = parse_style_to_object(style_str)
        logger.debug(f"Converting style: {style_str} to {{ {style_object} }}")
        return f'style={{ {style_object} }}'

    for idx, tag in enumerate(tags):
        # Skip nested tags to avoid duplicates
        if is_nested_tag(tag, processed_tags):
            logger.debug(
                f"Skipping nested tag: {tag.name} with class {tag.get('class', [])}")
            continue

        class_names = tag.get("class", [f"component{idx}"])
        base_class_name = class_names[0]
        base_component_name = ''.join(word.capitalize()
                                      for word in base_class_name.split('-'))

        if base_class_name in seen_class_names:
            seen_class_names[base_class_name] += 1
            component_name = f"{base_component_name}{seen_class_names[base_class_name]}"
        else:
            seen_class_names[base_class_name] = 0
            component_name = base_component_name

        styles = tag.get("style", "").strip()
        if not styles and tag.get("class"):
            css_files = [
                components_dir / f for f in os.listdir(components_dir) if f.endswith(".css")]
            class_styles = []
            primary_class = base_class_name
            for css_file in css_files:
                with open(css_file, "r", encoding="utf-8") as f:
                    css_content = f.read()
                    pattern = rf"\.{re.escape(primary_class)}\s*{{([^}}]*)}}"
                    matches = re.findall(pattern, css_content, re.DOTALL)
                    for match in matches:
                        cleaned_style = match.strip()
                        if cleaned_style and cleaned_style not in class_styles:
                            class_styles.append(cleaned_style)
            styles = ";".join(class_styles).strip()

        if not styles and tag.name in DEFAULT_STYLES:
            styles = DEFAULT_STYLES[tag.name]
            logger.debug(f"Applied default styles for {tag.name}: {styles}")

        component_html = str(tag)
        if not component_html.strip():
            logger.warning(
                f"Skipping empty tag: {tag.name} with class {base_class_name}")
            continue
        if component_html in seen_html_content:
            logger.debug(
                f"Skipping duplicate HTML content for tag: {tag.name} with class {base_class_name}")
            continue
        seen_html_content.add(component_html)

        if tag.name == "nav" and not tag.get_text(strip=True):
            logger.warning(
                f"Empty nav tag detected: {tag.get('id', 'no-id')} may require dynamic content")

        component_html = re.sub(r'\bclass=', 'className=', component_html)
        component_html = re.sub(r'\bclassName="[^"]*"\s*className="[^"]*"',
                                f'className="{base_class_name}"', component_html, flags=re.IGNORECASE)
        component_html = component_html.replace(
            'autocomplete=', 'autoComplete=')
        component_html = component_html.replace('for=', 'htmlFor=')
        component_html = component_html.replace('onclick=', 'onClick=')

        if styles:
            component_html = re.sub(
                r'style="([^"]*)"', replace_style, component_html)
        elif 'style="' in component_html:
            component_html = re.sub(
                r'style="([^"]*)"', replace_style, component_html)

        if not tag.get("class") and not tag.get("className"):
            tag_name = tag.name
            component_html = re.sub(
                rf'<{tag_name}\b', f'<{tag_name} className="{base_class_name}"', component_html, 1)

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
            css_class_name = base_class_name
            css_content = f".{css_class_name} {{{styles}}}"
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
                f"No specific styles generated for {component_name}; relying on iana_website.css")

    # Log missing JavaScript dependencies
    js_files = soup.find_all("script", src=True)
    for js in js_files:
        src = js.get("src")
        if src and not (components_dir / src).exists():
            logger.warning(
                f"JavaScript dependency {src} not processed; may affect dynamic functionality")

    # Generate App.jsx
    app_jsx_content = """\
const App = () => {{
  return (
    <div>
      {component_renders}
    </div>
  );
}};
window.App = App;
"""
    component_renders = "\n      ".join(
        f'<window.{component["name"]} />'
        for component in components
    )
    try:
        formatted_app_jsx = format_with_prettier(
            app_jsx_content.format(component_renders=component_renders),
            "babel",
            str(prettier_config_path),
            ".jsx"
        )
    except Exception as e:
        logger.error(f"Error formatting App.jsx: {e}")
        logger.debug(f"Problematic App.jsx content: {app_jsx_content}")
        formatted_app_jsx = app_jsx_content.format(
            component_renders=component_renders)

    app_path = components_dir / "App.jsx"
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(formatted_app_jsx.rstrip())
    logger.success(f"Generated App.jsx file at {app_path}")

    return components


def generate_entry_point(components: List[Component], output_dir: str, html_template: str = HTML_TEMPLATE) -> None:
    """Generate an index.html file to preview all React components using App.jsx."""
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

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
    if (components_dir / "iana_website.css").exists():
        css_links += '\n<link rel="stylesheet" href="./components/iana_website.css">'

    component_scripts = "\n".join(
        f'<script type="text/babel" src="./components/{component["name"]}.jsx"></script>'
        for component in components
    )
    component_scripts += '\n<script type="text/babel" src="./components/App.jsx"></script>'

    html_content = html_template.format(
        css_links=css_links,
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
