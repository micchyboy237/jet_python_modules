import os
import pytest
import shutil
from pathlib import Path
from typing import List
from jet.scrapers.automation.webpage_cloner.generate_components import generate_react_components, generate_entry_point, Component


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory and clean up after tests."""
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


class TestGenerateReactComponents:
    def test_generates_scoped_css_module(self, temp_output_dir: Path) -> None:
        """Test that a component is generated with a scoped CSS Module."""
        # Given: A simple HTML with a styled div
        html = '<div class="header" style="color: blue; font-size: 16px;">Hello</div>'
        expected_component_name = "Header"
        expected_html = '<div className={styles.header}>Hello</div>'
        expected_styles = "color: blue;font-size: 16px"

        # When: Generating React components
        components = generate_react_components(html, str(temp_output_dir))

        # Then: Verify component and CSS Module
        assert len(components) == 1
        component = components[0]
        assert component["name"] == expected_component_name
        assert expected_html in component["html"]
        assert component["styles"] == expected_styles

        component_path = temp_output_dir / "components" / \
            f"{expected_component_name}.jsx"
        css_path = temp_output_dir / "components" / \
            f"{expected_component_name}.module.css"
        assert component_path.exists()
        assert css_path.exists()

        with open(component_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert f"import styles from './{expected_component_name}.module.css'" in content
            assert f"className={{styles.header}}" in content
        with open(css_path, "r", encoding="utf-8") as f:
            assert ".header {color: blue;font-size: 16px}" in f.read()

    def test_deduplicates_styles(self, temp_output_dir: Path) -> None:
        """Test that duplicate styles are removed."""
        # Given: HTML with redundant styles
        html = '<div class="content" style="color: red; color: blue;">Text</div>'
        expected_styles = "color: blue"

        # When: Generating React components
        components = generate_react_components(html, str(temp_output_dir))

        # Then: Verify deduplicated styles
        assert len(components) == 1
        assert components[0]["styles"] == expected_styles
        css_path = temp_output_dir / "components" / "Content.module.css"
        assert css_path.exists()
        with open(css_path, "r", encoding="utf-8") as f:
            assert ".content {color: blue}" in f.read()


class TestGenerateEntryPoint:
    def test_generates_index_with_sorted_css_links(self, temp_output_dir: Path) -> None:
        """Test that index.html includes sorted CSS Module links."""
        # Given: Components with CSS files
        components: List[Component] = [
            {"name": "ZComponent", "html": "<div>Test</div>", "styles": "color: red"},
            {"name": "AComponent", "html": "<div>Test</div>", "styles": "color: blue"}
        ]
        (temp_output_dir / "components").mkdir()
        for component in components:
            with open(temp_output_dir / "components" / f"{component['name']}.module.css", "w") as f:
                f.write(
                    f".{component['name'].lower()} {{{component['styles']}}}")

        # When: Generating entry point
        generate_entry_point(components, str(temp_output_dir))

        # Then: Verify index.html has sorted CSS links
        index_path = temp_output_dir / "index.html"
        assert index_path.exists()
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert 'href="./components/AComponent.module.css"' in content
            assert 'href="./components/ZComponent.module.css"' in content
            # Verify order: AComponent comes before ZComponent
            assert content.index("AComponent.module.css") < content.index(
                "ZComponent.module.css")
