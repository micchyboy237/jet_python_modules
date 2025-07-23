from typing import List, TypedDict
from bs4 import BeautifulSoup, Comment
import re
import os
from pathlib import Path
import subprocess
import tempfile

from jet.logger import logger


class Component(TypedDict):
    name: str
    html: str
    styles: str


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


def generate_react_components(html: str, output_dir: str) -> List[Component]:
    """Generate React components from HTML content."""
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    # Write .prettierrc to components directory
    prettier_config = """\
{
  "singleQuote": true,
  "trailingComma": "es5",
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true
}
"""
    prettier_config_path = components_dir / ".prettierrc"
    with open(prettier_config_path, "w", encoding="utf-8") as f:
        f.write(prettier_config.rstrip())

    soup = BeautifulSoup(html, "html.parser")
    # Remove all HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    components: List[Component] = []
    tags = soup.find_all(["header", "footer", "section", "div"])

    for idx, tag in enumerate(tags):
        # Normalize class name to PascalCase for component name
        class_name = tag.get("class", [f"component{idx}"])[0]
        component_name = ''.join(word.capitalize()
                                 for word in class_name.split('-'))
        styles = ""
        if tag.get("style"):
            styles = tag["style"].strip()
        elif tag.get("class"):
            css_files = [
                os.path.join(output_dir_path, "assets", f)
                for f in os.listdir(os.path.join(output_dir_path, "assets"))
                if f.endswith(".css")
            ]
            for css_file in css_files:
                with open(css_file, "r", encoding="utf-8") as f:
                    css_content = f.read()
                    for cls in tag["class"]:
                        pattern = rf"\.{cls}\s*{{([^}}]*)}}"
                        matches = re.findall(pattern, css_content, re.DOTALL)
                        styles += "\n".join(match.strip() for match in matches)
        # Convert class to className for React
        component_html = str(tag).replace('class=', 'className=')
        components.append({
            "name": component_name,
            "html": component_html,
            "styles": styles.strip()
        })
        css_import = f"import './{component_name}.css';" if styles else ""
        component_code = f"""\
import React from 'react';
{css_import}

const {component_name} = () => {{
  return (
    {component_html}
  );
}};

export default {component_name};
"""
        # Format JSX using Prettier
        formatted_component_code = format_with_prettier(
            component_code, "babel", str(prettier_config_path), ".jsx"
        )

        component_path = components_dir / f"{component_name}.jsx"
        with open(component_path, "w", encoding="utf-8") as f:
            f.write(formatted_component_code.rstrip())

        if styles:
            # Use original class name for CSS selector
            css_class_name = class_name
            css_content = f".{css_class_name} {{{styles}}}"
            # Format CSS using Prettier
            formatted_styles = format_with_prettier(
                css_content, "css", str(prettier_config_path), ".css"
            )

            css_path = components_dir / f"{component_name}.css"
            with open(css_path, "w", encoding="utf-8") as f:
                f.write(formatted_styles.rstrip())

    return components
