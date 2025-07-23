import re
import os
import tempfile
import subprocess
import cssutils

from typing import Dict, List, TypedDict
from bs4 import BeautifulSoup, Comment
from pathlib import Path

from jet.logger import logger


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
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.25.6/babel.min.js"></script>
    {css_links}
</head>
<body>
    <div id="root" class="p-4"></div>
    {component_scripts}
    <script type="text/babel">
        function App() {{
            return (
                <div className="space-y-4">
                    {component_renders}
                </div>
            );
        }}
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>
"""


def parse_style_to_object(style: str) -> str:
    """Convert a CSS style string to a JavaScript object string for React using cssutils."""
    if not style:
        return "{}"

    # Initialize cssutils stylesheet
    cssutils.ser.prefs.useMinified()  # Minimize output for cleaner parsing
    # Wrap in dummy selector for parsing
    sheet = cssutils.parseString(f"dummy {{ {style} }}")

    style_dict: Dict[str, str] = {}
    try:
        for rule in sheet:
            if rule.type == rule.STYLE_RULE:
                for property in rule.style:
                    # Convert CSS property to camelCase (e.g., background-color -> backgroundColor)
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
                    # Quote non-numeric values
                    try:
                        float(value)  # Check if value is numeric
                        style_dict[camel_prop] = value  # Keep numbers unquoted
                    except ValueError:
                        style_dict[camel_prop] = f"'{value}'"  # Quote strings
    except Exception as e:
        logger.error(f"Error parsing CSS with cssutils: {e}")
        return "{}"

    # Format as JavaScript object
    style_items = [f"{key}: {value}" for key, value in style_dict.items()]
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


def generate_react_components(html: str, output_dir: str, component_code_template: str = COMPONENT_CODE_TEMPLATE) -> List[Component]:
    """Generate React components from HTML content with unique component names and valid JSX styles."""
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    prettier_config = PRETTIER_CONFIG
    prettier_config_path = components_dir / ".prettierrc"
    with open(prettier_config_path, "w", encoding="utf-8") as f:
        f.write(prettier_config.rstrip())
    logger.success(f"Generated Prettier config file at {prettier_config_path}")

    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    components: List[Component] = []
    tags = soup.find_all(["header", "footer", "section", "div", "a"])
    seen_class_names = {}  # Track class names to ensure uniqueness

    def replace_style(match):
        style_str = match.group(1)
        style_object = parse_style_to_object(style_str)
        logger.debug(f"Converting style: {style_str} to {{ {style_object} }}")
        # Ensure double curly braces for JSX
        return f'style={{ {style_object} }}'

    for idx, tag in enumerate(tags):
        class_name = tag.get("class", [f"component{idx}"])[0]
        base_component_name = ''.join(word.capitalize()
                                      for word in class_name.split('-'))
        if class_name in seen_class_names:
            seen_class_names[class_name] += 1
            component_name = f"{base_component_name}{seen_class_names[class_name]}"
        else:
            seen_class_names[class_name] = 0
            component_name = base_component_name

        styles = ""
        if tag.get("style"):
            styles = tag["style"].strip()
        elif tag.get("class"):
            assets_dir = output_dir_path / "assets"
            css_files = []
            if assets_dir.exists():
                css_files = [os.path.join(assets_dir, f) for f in os.listdir(
                    assets_dir) if f.endswith(".css")]
            class_styles = []
            primary_class = class_name
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

        component_html = str(tag)
        # Convert HTML attributes to JSX
        component_html = component_html.replace('class=', 'className=')
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
                rf'<{tag_name}\b', f'<{tag_name} className="{class_name}"', component_html, 1)

        components.append({
            "name": component_name,
            "html": component_html,
            "styles": styles
        })

        component_code = component_code_template.format(
            component_name=component_name,
            css_import="",  # No CSS imports
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
            css_class_name = class_name
            css_content = f".{css_class_name} {{{styles}}}"
            formatted_styles = format_with_prettier(
                css_content, "css", str(prettier_config_path), ".css"
            )
            css_path = components_dir / f"{component_name}.css"
            with open(css_path, "w", encoding="utf-8") as f:
                f.write(formatted_styles.rstrip())
            logger.success(f"Generated CSS file at {css_path}")

    return components


def generate_entry_point(components: List[Component], output_dir: str, html_template: str = HTML_TEMPLATE) -> None:
    """Generate an index.html file to preview all React components."""
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    prettier_config = PRETTIER_CONFIG
    prettier_config_path = components_dir / ".prettierrc"
    with open(prettier_config_path, "w", encoding="utf-8") as f:
        f.write(prettier_config.rstrip())
    logger.success(f"Generated Prettier config file at {prettier_config_path}")

    # Include CSS files as <link> tags only if they exist
    css_links = "\n".join(
        f'<link rel="stylesheet" href="./components/{component["name"]}.css">'
        for component in components
        if component["styles"] and (components_dir / f"{component['name']}.css").exists()
    )
    component_scripts = "\n".join(
        f'<script type="text/babel" src="./components/{component["name"]}.jsx"></script>'
        for component in components
    )
    component_renders = "\n".join(
        f"<{component['name']} />"
        for component in components
    )

    html_content = html_template.format(
        css_links=css_links,
        component_scripts=component_scripts,
        component_renders=component_renders
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
