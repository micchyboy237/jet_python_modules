from typing import List, TypedDict
from bs4 import BeautifulSoup
import re
import os
from pathlib import Path


class Component(TypedDict):
    name: str
    html: str
    styles: str


def generate_react_components(html: str, output_dir: str) -> List[Component]:
    """Generate React components from HTML content."""
    output_dir_path = Path(output_dir).resolve()
    components_dir = output_dir_path / "components"
    components_dir.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(html, "html.parser")
    components: List[Component] = []

    # Identify potential components (e.g., header, footer, sections)
    tags = soup.find_all(["header", "footer", "section", "div"])
    for idx, tag in enumerate(tags):
        class_name = tag.get("class", [f"Component{idx}"])[0].capitalize()
        component_name = re.sub(r"[^a-zA-Z0-9]", "", class_name)

        # Extract inline styles or linked CSS rules
        styles = ""
        if tag.get("style"):
            styles = tag["style"]
        elif tag.get("class"):
            # Extract CSS rules for the class
            css_files = [os.path.join(output_dir_path, "assets", f) for f in os.listdir(
                os.path.join(output_dir_path, "assets")) if f.endswith(".css")]
            for css_file in css_files:
                with open(css_file, "r", encoding="utf-8") as f:
                    css_content = f.read()
                    for cls in tag["class"]:
                        pattern = rf"\.{cls}\s*{{([^}}]*)}}"
                        matches = re.findall(pattern, css_content, re.DOTALL)
                        styles += "\n".join(matches)

        # Generate React component code
        component_html = str(tag)
        components.append({
            "name": component_name,
            "html": component_html,
            "styles": styles
        })

        # Include CSS import only if styles exist
        css_import = f"import './{component_name}.css';" if styles else ""
        component_code = f"""import React from 'react';
{css_import}

const {component_name} = () => {{
  return (
    {component_html}
  );
}};

export default {component_name};
"""
        component_path = components_dir / f"{component_name}.jsx"
        with open(component_path, "w", encoding="utf-8") as f:
            f.write(component_code)

        # Write styles to CSS file only if styles exist
        if styles:
            css_path = components_dir / f"{component_name}.css"
            with open(css_path, "w", encoding="utf-8") as f:
                f.write(f".{component_name.lower()} {{{styles}}}")

    return components
