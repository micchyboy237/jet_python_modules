# jet_python_modules/jet/scrapers/automation/webpage_cloner/test_generate_components.py
import os
import shutil
from pathlib import Path
from typing import List
import pytest
from jet.scrapers.automation.webpage_cloner.generate_components import generate_entry_point, generate_react_components, Component


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> str:
    """Create a temporary output directory with assets."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    with open(assets_dir / "styles.css", "w", encoding="utf-8") as f:
        f.write(
            """
.test-class { color: blue; font-size: 16px; }
.multi-class { background-color: red; }
.test-class.multi-class { border: 1px solid black; }
"""
        )
    return str(tmp_path)


@pytest.fixture
def temp_output_dir_no_assets(tmp_path: Path) -> str:
    """Create a temporary output directory without assets."""
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

    def test_generates_component_with_external_css(self, temp_output_dir: str) -> None:
        """Test generating a component with styles from an external CSS file."""
        # Given: HTML with a class linked to external CSS
        input_html = '<div class="test-class">Test Content</div>'
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
        result = generate_react_components(input_html, temp_output_dir)

        # Then: Component metadata is correct
        assert result == [expected_component]

        # Then: JSX file is generated with correct content
        component_path = Path(temp_output_dir) / "components" / "TestClass.jsx"
        assert component_path.exists()
        with open(component_path, "r", encoding="utf-8") as f:
            result_jsx = f.read()
        assert result_jsx == expected_jsx.rstrip()

        # Then: CSS file is generated with correct styles
        css_path = Path(temp_output_dir) / "components" / "TestClass.css"
        assert css_path.exists()
        with open(css_path, "r", encoding="utf-8") as f:
            result_css = f.read()
        assert result_css == expected_css.rstrip()

    def test_generates_component_with_inline_styles(self, temp_output_dir: str) -> None:
        """Test generating a component with inline styles."""
        # Given: HTML with inline styles
        input_html = '<div style="color: green; padding: 10px;">Inline Content</div>'
        expected_component: Component = {
            "name": "Component0",
            "html": '<div className="component0" style="color: green; padding: 10px;">Inline Content</div>',
            "styles": "color: green; padding: 10px;"
        }
        expected_jsx = """\
import React from 'react';
import './Component0.css';

const Component0 = () => {
  return (
    <div className="component0" style="color: green; padding: 10px;">
      Inline Content
    </div>
  );
};

export default Component0;
"""
        expected_css = """\
.component0 {
  color: green;
  padding: 10px;
}
"""

        # When: Generating components
        result = generate_react_components(input_html, temp_output_dir)

        # Then: Component metadata is correct
        assert result == [expected_component]

        # Then: JSX file is generated with correct content
        component_path = Path(temp_output_dir) / \
            "components" / "Component0.jsx"
        assert component_path.exists()
        with open(component_path, "r", encoding="utf-8") as f:
            result_jsx = f.read()
        assert result_jsx == expected_jsx.rstrip()

        # Then: CSS file is generated with correct styles
        css_path = Path(temp_output_dir) / "components" / "Component0.css"
        assert css_path.exists()
        with open(css_path, "r", encoding="utf-8") as f:
            result_css = f.read()
        assert result_css == expected_css.rstrip()

    def test_generates_component_without_styles(self, temp_output_dir: str) -> None:
        """Test generating a component without styles."""
        # Given: HTML without styles
        input_html = '<div>Test Content</div>'
        expected_component: Component = {
            "name": "Component0",
            "html": '<div className="component0">Test Content</div>',
            "styles": ""
        }
        expected_jsx = """\
import React from 'react';

const Component0 = () => {
  return <div className="component0">Test Content</div>;
};

export default Component0;
"""

        # When: Generating components
        result = generate_react_components(input_html, temp_output_dir)

        # Then: Component metadata is correct
        assert result == [expected_component]

        # Then: JSX file is generated without CSS import
        component_path = Path(temp_output_dir) / \
            "components" / "Component0.jsx"
        assert component_path.exists()
        with open(component_path, "r", encoding="utf-8") as f:
            result_jsx = f.read()
        assert result_jsx == expected_jsx.rstrip()

        # Then: No CSS file is created
        css_path = Path(temp_output_dir) / "components" / "Component0.css"
        assert not css_path.exists()

    def test_generates_component_with_multiple_classes(self, temp_output_dir: str) -> None:
        """Test generating a component with multiple classes."""
        # Given: HTML with multiple classes
        input_html = '<div class="test-class multi-class">Multi Class Content</div>'
        expected_component: Component = {
            "name": "TestClass",
            "html": '<div className="test-class multi-class">Multi Class Content</div>',
            "styles": "color: blue; font-size: 16px;\nborder: 1px solid black;"
        }
        expected_jsx = """\
import React from 'react';
import './TestClass.css';

const TestClass = () => {
  return <div className="test-class multi-class">Multi Class Content</div>;
};

export default TestClass;
"""
        expected_css = """\
.test-class {
  color: blue;
  font-size: 16px;
  border: 1px solid black;
}
"""

        # When: Generating components
        result = generate_react_components(input_html, temp_output_dir)

        # Then: Component metadata is correct
        assert result == [expected_component]

        # Then: JSX file is generated with correct content
        component_path = Path(temp_output_dir) / "components" / "TestClass.jsx"
        assert component_path.exists()
        with open(component_path, "r", encoding="utf-8") as f:
            result_jsx = f.read()
        assert result_jsx == expected_jsx.rstrip()

        # Then: CSS file is generated with combined styles
        css_path = Path(temp_output_dir) / "components" / "TestClass.css"
        assert css_path.exists()
        with open(css_path, "r", encoding="utf-8") as f:
            result_css = f.read()
        assert result_css == expected_css.rstrip()

    def test_generates_component_no_assets_directory(self, temp_output_dir_no_assets: str) -> None:
        """Test generating a component when no assets directory exists."""
        # Given: HTML with a class but no assets directory
        input_html = '<div class="test-class">No Assets Content</div>'
        expected_component: Component = {
            "name": "TestClass",
            "html": '<div className="test-class">No Assets Content</div>',
            "styles": ""
        }
        expected_jsx = """\
import React from 'react';

const TestClass = () => {
  return <div className="test-class">No Assets Content</div>;
};

export default TestClass;
"""

        # When: Generating components
        result = generate_react_components(
            input_html, temp_output_dir_no_assets)

        # Then: Component metadata is correct with no styles
        assert result == [expected_component]

        # Then: JSX file is generated without CSS import
        component_path = Path(temp_output_dir_no_assets) / \
            "components" / "TestClass.jsx"
        assert component_path.exists()
        with open(component_path, "r", encoding="utf-8") as f:
            result_jsx = f.read()
        assert result_jsx == expected_jsx.rstrip()

        # Then: No CSS file is created
        css_path = Path(temp_output_dir_no_assets) / \
            "components" / "TestClass.css"
        assert not css_path.exists()


class TestGenerateEntryPoint:
    def test_generates_index_html_with_single_component(self, temp_output_dir: str) -> None:
        """Test generating index.html with a single component."""
        # Given: HTML input with a single component and an output directory
        input_html = '<div class="test-class">Test Content</div>'
        expected_components: List[Component] = [{
            "name": "TestClass",
            "html": '<div className="test-class">Test Content</div>',
            "styles": "color: blue; font-size: 16px;"
        }]
        expected_index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Components Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.20.15/babel.min.js"></script>
</head>
<body>
    <div id="root" class="p-4"></div>
    <script type="text/babel">
        import TestClass from './components/TestClass.jsx';

        function App() {
            return (
                <div className="space-y-4">
                    <TestClass />
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>"""

        # When: Generating components and entry point
        components = generate_react_components(input_html, temp_output_dir)
        assert components == expected_components, "Generated components do not match expected"
        generate_entry_point(components, temp_output_dir)

        # Then: Verify index.html exists and contains the expected content
        index_path = Path(temp_output_dir) / "index.html"
        assert index_path.exists()
        with open(index_path, "r", encoding="utf-8") as f:
            result_content = f.read()
        assert result_content == expected_index_content.rstrip()

    def test_generates_index_html_with_multiple_components(self, temp_output_dir: str) -> None:
        """Test generating index.html with multiple components."""
        # Given: HTML input with multiple components and an output directory
        input_html = """
        <div class="test-class">Test Content</div>
        <section class="multi-class">Multi Content</section>
        """
        expected_components: List[Component] = [
            {
                "name": "TestClass",
                "html": '<div className="test-class">Test Content</div>',
                "styles": "color: blue; font-size: 16px;"
            },
            {
                "name": "MultiClass",
                "html": '<section className="multi-class">Multi Content</section>',
                "styles": "background-color: red;"
            }
        ]
        expected_index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Components Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.20.15/babel.min.js"></script>
</head>
<body>
    <div id="root" class="p-4"></div>
    <script type="text/babel">
        import TestClass from './components/TestClass.jsx';
        import MultiClass from './components/MultiClass.jsx';

        function App() {
            return (
                <div className="space-y-4">
                    <TestClass />
                    <MultiClass />
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>"""

        # When: Generating components and entry point
        components = generate_react_components(input_html, temp_output_dir)
        assert components == expected_components, "Generated components do not match expected"
        generate_entry_point(components, temp_output_dir)

        # Then: Verify index.html exists and contains the expected content
        index_path = Path(temp_output_dir) / "index.html"
        assert index_path.exists()
        with open(index_path, "r", encoding="utf-8") as f:
            result_content = f.read()
        assert result_content == expected_index_content.rstrip()

    def test_generates_index_html_no_components(self, temp_output_dir: str) -> None:
        """Test generating index.html with no components."""
        # Given: Empty components list and an output directory
        expected_components: List[Component] = []
        expected_index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Components Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.20.15/babel.min.js"></script>
</head>
<body>
    <div id="root" class="p-4"></div>
    <script type="text/babel">
        function App() {
            return (
                <div className="space-y-4">
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>"""

        # When: Generating entry point with no components
        generate_entry_point(expected_components, temp_output_dir)

        # Then: Verify index.html exists and contains the expected content
        index_path = Path(temp_output_dir) / "index.html"
        assert index_path.exists()
        with open(index_path, "r", encoding="utf-8") as f:
            result_content = f.read()
        assert result_content == expected_index_content.rstrip()
