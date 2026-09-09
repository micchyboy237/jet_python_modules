"""Unit tests for html_schema_generator module."""

from jet.adapters.unstructured.html_schema_generator import (
    generate_html_schema,
    schema_to_llm_context,
)

SAMPLE_HTML = """
<html>
<body>
  <h1>Main Title</h1>
  <p>Introduction paragraph.</p>
  <h2>Section One</h2>
  <p>Section one content.</p>
  <ul>
    <li>Item A</li>
    <li>Item B</li>
  </ul>
  <h2>Section Two</h2>
  <p>Section two content.</p>
</body>
</html>
"""

MINIMAL_HTML = "<html><body><p>Only a paragraph.</p></body></html>"

EMPTY_HTML = "<html><body></body></html>"

HTML_WITH_LINKS = """
<html><body>
  <h1>Links Page</h1>
  <p>Visit <a href="https://example.com">Example</a> for more.</p>
</body></html>
"""


class TestGenerateHtmlSchema:
    """Tests for generate_html_schema function."""

    def test_returns_root_nodes(self):
        schema = generate_html_schema(SAMPLE_HTML)
        assert isinstance(schema, list)
        assert len(schema) > 0

    def test_hierarchy_has_children(self):
        schema = generate_html_schema(SAMPLE_HTML)
        # At least one root should have children (Title -> NarrativeText)
        all_nodes = self._flatten(schema)
        nodes_with_children = [n for n in all_nodes if n["children"]]
        assert len(nodes_with_children) > 0, "Expected at least one node with children"

    def test_excludes_page_break_by_default(self):
        html = "<html><body><p>Text</p><!-- pagebreak --></body></html>"
        schema = generate_html_schema(html)
        all_types = {n["type"] for n in self._flatten(schema)}
        assert "PageBreak" not in all_types

    def test_custom_exclude_types(self):
        schema = generate_html_schema(SAMPLE_HTML, exclude_types={"Title"})
        all_types = {n["type"] for n in self._flatten(schema)}
        assert "Title" not in all_types

    def test_empty_html_returns_empty_list(self):
        schema = generate_html_schema(EMPTY_HTML)
        assert schema == []

    def test_minimal_html_single_root(self):
        schema = generate_html_schema(MINIMAL_HTML)
        assert len(schema) >= 1
        # Accept both NarrativeText and UncategorizedText for short inputs
        assert schema[0]["type"] in {"NarrativeText", "UncategorizedText"}

    def test_metadata_included_by_default(self):
        schema = generate_html_schema(SAMPLE_HTML)
        all_nodes = self._flatten(schema)
        nodes_with_meta = [n for n in all_nodes if "metadata" in n]
        assert len(nodes_with_meta) > 0

    def test_metadata_excluded_when_flag_false(self):
        schema = generate_html_schema(SAMPLE_HTML, include_metadata=False)
        all_nodes = self._flatten(schema)
        nodes_with_meta = [n for n in all_nodes if "metadata" in n]
        assert len(nodes_with_meta) == 0

    def test_link_metadata_preserved(self):
        schema = generate_html_schema(HTML_WITH_LINKS)
        all_nodes = self._flatten(schema)
        link_nodes = [n for n in all_nodes if n.get("metadata", {}).get("link_urls")]
        assert len(link_nodes) > 0, "Expected link_urls in metadata"

    def test_element_ids_are_unique(self):
        schema = generate_html_schema(SAMPLE_HTML)
        all_ids = [n["element_id"] for n in self._flatten(schema)]
        assert len(all_ids) == len(set(all_ids)), "Element IDs must be unique"

    @staticmethod
    def _flatten(nodes: list[dict]) -> list[dict]:
        result = []
        for node in nodes:
            result.append(node)
            result.extend(TestGenerateHtmlSchema._flatten(node.get("children", [])))
        return result


class TestSchemaToLlmContext:
    """Tests for schema_to_llm_context function."""

    def test_returns_string(self):
        schema = generate_html_schema(SAMPLE_HTML)
        result = schema_to_llm_context(schema)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_indentation_reflects_depth(self):
        schema = generate_html_schema(SAMPLE_HTML)
        result = schema_to_llm_context(schema)
        lines = result.split("\n")
        # Should have at least one indented line
        indented = [l for l in lines if l.startswith("  ")]
        assert len(indented) > 0

    def test_max_depth_limits_output(self):
        schema = generate_html_schema(SAMPLE_HTML)
        shallow = schema_to_llm_context(schema, max_depth=1)
        deep = schema_to_llm_context(schema, max_depth=10)
        assert len(shallow.split("\n")) <= len(deep.split("\n"))

    def test_empty_schema_returns_empty_string(self):
        result = schema_to_llm_context([])
        assert result == ""

    def test_contains_element_types(self):
        schema = generate_html_schema(SAMPLE_HTML)
        result = schema_to_llm_context(schema)
        assert "[Title]" in result or "[NarrativeText]" in result
