import pytest
from pathlib import Path
from jet.scrapers.automation.webpage_cloner.generate_components import generate_react_components, parse_style_to_object
import re


class TestGenerateReactComponents:
    """Tests for generate_react_components function, focusing on JSX style attribute handling."""

    def test_style_attribute_jsx_syntax(self):
        """Test that style attributes are converted to valid JSX syntax with double curly braces."""
        # Given an HTML string with a style attribute
        html_input = '<div class="test" style="background-color: red; font-size: 16px;"></div>'
        output_dir = str(Path(__file__).parent / "temp_output")
        expected_style = "{{backgroundColor: 'red', fontSize: '16px'}}"
        expected_pattern = re.compile(
            r'<div className="test" style=' + re.escape(expected_style) + r'>')

        # When generating React components
        components = generate_react_components(html_input, output_dir)

        # Then the generated component HTML should have a style prop with double curly braces
        assert len(
            components) == 1, f"Expected 1 component, but got {len(components)}"
        result_html = components[0]["html"]
        assert expected_pattern.search(result_html), (
            f"Expected style prop with {expected_style} in {result_html}"
        )

    def test_rgba_style_jsx_syntax(self):
        """Test that RGBA style values are converted to valid JSX syntax."""
        # Given an HTML string with an RGBA style
        html_input = '<a class="btn" style="background-color: rgba(255, 255, 255, .5);"></a>'
        output_dir = str(Path(__file__).parent / "temp_output")
        expected_style = "{{backgroundColor: 'rgba(255, 255, 255, 0.5)'}}"
        expected_pattern = re.compile(
            r'<a className="btn" style=' + re.escape(expected_style) + r'>')

        # When generating React components
        components = generate_react_components(html_input, output_dir)

        # Then the generated component HTML should have a style prop with double curly braces
        assert len(
            components) == 1, f"Expected 1 component, but got {len(components)}"
        result_html = components[0]["html"]
        assert expected_pattern.search(result_html), (
            f"Expected style prop with {expected_style} in {result_html}"
        )

    def test_numeric_style_jsx_syntax(self):
        """Test that numeric style values are converted to valid JSX syntax without quotes."""
        # Given an HTML string with a numeric style property
        html_input = '<div class="overlay" style="z-index: 2147483647; position: fixed;"></div>'
        output_dir = str(Path(__file__).parent / "temp_output")
        expected_style = "{{zIndex: 2147483647, position: 'fixed'}}"
        expected_pattern = re.compile(
            r'<div className="overlay" style=' + re.escape(expected_style) + r'>')

        # When generating React components
        components = generate_react_components(html_input, output_dir)

        # Then the generated component HTML should have a style prop with double curly braces
        assert len(
            components) == 1, f"Expected 1 component, but got {len(components)}"
        result_html = components[0]["html"]
        assert expected_pattern.search(result_html), (
            f"Expected style prop with {expected_style} in {result_html}"
        )

    def test_self_closing_tag(self):
        """Test handling of self-closing HTML tags."""
        # Given an HTML string with a self-closing tag
        html_input = '<img class="icon" style="width: 100px;" />'
        output_dir = str(Path(__file__).parent / "temp_output")
        expected_style = "{{width: '100px'}}"
        expected_pattern = re.compile(
            r'<img className="icon" style=' + re.escape(expected_style) + r'\s*/>')

        # When generating React components
        components = generate_react_components(html_input, output_dir)

        # Then the generated component HTML should handle self-closing tags correctly
        assert len(
            components) == 1, f"Expected 1 component, but got {len(components)}"
        result_html = components[0]["html"]
        assert expected_pattern.search(result_html), (
            f"Expected style prop with {expected_style} in {result_html}"
        )

    def test_additional_jsx_attributes(self):
        """Test conversion of additional HTML attributes to JSX."""
        # Given an HTML string with various attributes
        html_input = '<input class="field" style="width: 200px;" onchange="handleChange()" tabindex="1" />'
        output_dir = str(Path(__file__).parent / "temp_output")
        expected_style = "{{width: '200px'}}"
        expected_pattern = re.compile(
            r'<input className="field" style=' + re.escape(expected_style) + r'\s*onChange=\{handleChange\(\)\}\s*tabIndex="1"\s*/>')

        # When generating React components
        components = generate_react_components(html_input, output_dir)

        # Then the generated component HTML should convert attributes to JSX correctly
        assert len(
            components) == 1, f"Expected 1 component, but got {len(components)}"
        result_html = components[0]["html"]
        assert expected_pattern.search(result_html), (
            f"Expected JSX attributes in {result_html}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
