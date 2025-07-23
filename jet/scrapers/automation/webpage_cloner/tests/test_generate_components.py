# jet_python_modules/jet/scrapers/automation/webpage_cloner/test_generate_components.py
import os
import shutil
from pathlib import Path
import pytest
from jet.scrapers.automation.webpage_cloner.generate_components import generate_react_components, Component


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> str:
    """Create a temporary output directory with assets."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    with open(assets_dir / "styles.css", "w", encoding="utf-8") as f:
        f.write(".test-class { color: blue; font-size: 16px; }")
    return str(tmp_path)


class TestGenerateReactComponents:
    def test_generates_prettified_jsx_and_css(self, temp_output_dir: str) -> None:
        # Given: HTML with a styled div
        html = '<div class="test-class">Test Content</div>'
        expected_component: Component = {
            "name": "TestClass",
            "html": '<div className="test-class">Test Content</div>',
            "styles": "color: blue; font-size: 16px;"
        }
        expected_jsx = """\
import React from 'react';
import './TestClass.css';

const TestClass = () => {
  return <div className="test-class">Test Content</div>;
};

export default TestClass;
"""
        expected_css = """\
.test-class {
  color: blue;
  font-size: 16px;
}
"""

        # When: Generating components
        result = generate_react_components(html, temp_output_dir)

        # Then: Component is generated correctly
        assert len(result) == 1
        assert result[0] == expected_component

        # Then: JSX file is prettified
        component_path = Path(temp_output_dir) / "components" / "TestClass.jsx"
        assert component_path.exists()
        with open(component_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert content == expected_jsx.rstrip()

        # Then: CSS file is prettified
        css_path = Path(temp_output_dir) / "components" / "TestClass.css"
        assert css_path.exists()
        with open(css_path, "r", encoding="utf-8") as f:
            assert f.read() == expected_css.rstrip()

    def test_generates_component_without_styles(self, temp_output_dir: str) -> None:
        # Given: HTML without styles
        html = '<div>Test Content</div>'
        expected_component: Component = {
            "name": "Component0",
            "html": '<div>Test Content</div>',
            "styles": ""
        }
        expected_jsx = """\
import React from 'react';

const Component0 = () => {
  return <div>Test Content</div>;
};

export default Component0;
"""

        # When: Generating components
        result = generate_react_components(html, temp_output_dir)

        # Then: Component is generated correctly
        assert len(result) == 1
        assert result[0] == expected_component

        # Then: JSX file is prettified and no CSS file is created
        component_path = Path(temp_output_dir) / \
            "components" / "Component0.jsx"
        assert component_path.exists()
        with open(component_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert content == expected_jsx.rstrip()
        css_path = Path(temp_output_dir) / "components" / "Component0.css"
        assert not css_path.exists()
