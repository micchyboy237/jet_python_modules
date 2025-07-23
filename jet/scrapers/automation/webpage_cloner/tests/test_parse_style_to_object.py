import pytest
from jet.scrapers.automation.webpage_cloner.generate_components import parse_style_to_object


class TestParseStyleToObject:
    """Tests for parse_style_to_object function converting CSS styles to React style objects."""

    def test_empty_style(self):
        """Test handling of empty or whitespace-only style strings."""
        # Given an empty or whitespace-only style string
        style_input = ""
        expected = "{}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should return an empty JavaScript object
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_property(self):
        """Test parsing a single CSS property."""
        # Given a style string with a single property
        style_input = "background-color: red;"
        expected = "{backgroundColor: 'red'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should convert to camelCase with quoted value
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_multiple_properties(self):
        """Test parsing multiple CSS properties with various formats."""
        # Given a style string with multiple properties
        style_input = "font-size: 16px; margin-top: 10px; color: blue;"
        expected = "{fontSize: '16px', marginTop: '10px', color: 'blue'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should convert all properties to camelCase with quoted values
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_numeric_property(self):
        """Test parsing a numeric CSS property without quotes."""
        # Given a style string with a numeric property
        style_input = "z-index: 100;"
        expected = "{zIndex: 100}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should return the numeric value unquoted
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_rgba_property(self):
        """Test parsing an RGBA color value with decimal alpha."""
        # Given a style string with an RGBA color
        style_input = "background-color: rgba(255, 255, 255, .5);"
        expected = "{backgroundColor: 'rgba(255, 255, 255, 0.5)'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should standardize the decimal alpha
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_malformed_style(self):
        """Test handling of malformed style strings."""
        # Given a malformed style string missing a value
        style_input = "color: ; font-size: 12px;"
        expected = "{fontSize: '12px'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should skip invalid declarations
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_complex_properties(self):
        """Test parsing complex properties with multiple hyphens and values."""
        # Given a style string with complex properties
        style_input = "border-bottom-left-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, .2);"
        expected = "{borderBottomLeftRadius: '5px', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should handle multiple hyphens and RGBA values correctly
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_complex_css_value(self):
        """Test parsing a CSS property with multiple values."""
        # Given a style string with a complex multi-value property
        style_input = "border: 1px solid black;"
        expected = "{border: '1px solid black'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should handle the multi-value property as a quoted string
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_css_variable(self):
        """Test parsing a CSS variable."""
        # Given a style string with a CSS variable
        style_input = "color: var(--primary-color);"
        expected = "{color: 'var(--primary-color)'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should treat the variable as a quoted string
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_unicode_value(self):
        """Test parsing a CSS value with Unicode characters."""
        # Given a style string with a Unicode value
        style_input = "content: '★'; font-family: 'Noto Sans';"
        expected = "{content: '★', fontFamily: 'Noto Sans'}"

        # When parsing the style
        result = parse_style_to_object(style_input)

        # Then it should handle Unicode characters correctly
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_standard_css_properties(self):
        """Test parsing standard CSS properties with various value types."""
        # Given: A CSS string with mixed property types
        css_input = "background-color: rgba(255, 255, 255, .5); font-size: 16px; z-index: 1000;"
        expected = "{ backgroundColor: 'rgba(255, 255, 255, 0.5)', fontSize: '16px', zIndex: 1000 }"

        # When: Parsing the CSS string
        result = parse_style_to_object(css_input)

        # Then: The output matches the expected JavaScript object
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_missing_semicolon(self):
        """Test parsing CSS with missing semicolon."""
        # Given: A CSS string missing a semicolon
        css_input = "color: red; font-size: 14px"
        expected = "{ color: 'red', fontSize: '14px' }"

        # When: Parsing the CSS string
        result = parse_style_to_object(css_input)

        # Then: The output correctly handles the missing semicolon
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_value(self):
        """Test parsing CSS with empty property value."""
        # Given: A CSS string with an empty value
        css_input = "color:; font-size: 14px;"
        expected = "{ fontSize: '14px' }"

        # When: Parsing the CSS string
        result = parse_style_to_object(css_input)

        # Then: The output skips the empty value
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_string(self):
        """Test parsing empty CSS input."""
        # Given: An empty CSS string
        css_input = ""
        expected = "{}"

        # When: Parsing the empty string
        result = parse_style_to_object(css_input)

        # Then: The output is an empty object
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_complex_values(self):
        """Test parsing CSS with complex values like calc and url."""
        # Given: A CSS string with complex values
        css_input = "width: calc(100% - 20px); background: url('image.jpg');"
        expected = "{ width: 'calc(100% - 20px)', background: 'url(image.jpg)' }"

        # When: Parsing the CSS string
        result = parse_style_to_object(css_input)

        # Then: The output correctly handles complex values
        assert result == expected, f"Expected {expected}, but got {result}"


if __name__ == "__main__":
    pytest.main([__file__])
