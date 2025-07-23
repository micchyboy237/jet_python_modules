import shutil
from typing import List
from pathlib import Path
import pytest
from jet.scrapers.automation.webpage_cloner.generate_components import generate_react_components, Component


@pytest.mark.asyncio
async def test_generate_components_creates_jsx_without_styles():
    """
    Test that generate_react_components creates a JSX component without styles.
    Given an HTML string and output directory
    When generate_react_components is called
    Then a JSX component file is created with the expected content and no CSS file
    """
    # Given
    html = "<div>Hello, World!</div>"
    output_dir = "test_output"
    expected_component_name = "Component0"
    expected_html = "<div>Hello, World!</div>"
    expected_jsx_content = f"""import React from 'react';

const {expected_component_name} = () => {{
  return (
    {expected_html}
  );
}};

export default {expected_component_name};
"""

    # When
    components: List[Component] = generate_react_components(html, output_dir)

    # Then
    result_component = components[0]
    component_path = Path(output_dir) / "components" / \
        f"{expected_component_name}.jsx"
    css_path = Path(output_dir) / "components" / \
        f"{expected_component_name}.css"

    assert result_component["name"] == expected_component_name, "Component name does not match"
    assert result_component["html"] == expected_html, "Component HTML content does not match"
    assert result_component["styles"] == "", "Styles should be empty"
    assert component_path.exists(), "JSX component file was not created"
    assert not css_path.exists(), "CSS file should not be created"

    result_jsx_content = component_path.read_text(encoding="utf-8")
    assert result_jsx_content == expected_jsx_content, "JSX file content does not match expected"

    # Cleanup
    shutil.rmtree(output_dir)
