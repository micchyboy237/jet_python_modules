import pytest
from jet.libs.smolagents.agents.grok_equivalent_agent import (
    AgentConfig,
    _fetch_url,
    _process_to_markdown,
    create_manager_agent,
    create_web_agent,
    display_config_table,
    visit_webpage,
)
from smolagents import CodeAgent, ToolCallingAgent

create_manager_agent


# Mock model for testing (since LLM calls are non-deterministic, we focus on structure)
class MockModel:
    pass


@pytest.fixture
def mock_config() -> AgentConfig:
    return {
        "model_id": "test-model",
        "max_steps": 5,
        "executor_type": "local",
    }


class TestVisitWebpageComponents:
    def test_fetch_url_success(self, requests_mock):
        # Given: A valid URL with mock content
        url = "https://example.com"
        mock_content = "<html><body>Test</body></html>"
        expected = mock_content

        requests_mock.get(url, text=mock_content, status_code=200)

        # When: Fetch the URL
        result = _fetch_url(url)

        # Then: Assert the full raw content matches expected
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_fetch_url_error(self, requests_mock):
        # Given: An invalid URL that raises an error
        url = "https://invalid.com"
        expected_error_prefix = "Error fetching the webpage:"

        requests_mock.get(url, exc=requests.exceptions.RequestException("Mock error"))

        # When: Attempt to fetch the URL
        with pytest.raises(ValueError) as exc_info:
            _fetch_url(url)

        # Then: Assert the error message starts with expected prefix
        assert expected_error_prefix in str(exc_info.value), (
            f"Expected error containing '{expected_error_prefix}', but got '{exc_info.value}'"
        )

    def test_process_to_markdown(self):
        # Given: Raw HTML content
        raw_content = "<html><body><h1>Title</h1><p>Para1</p><p>Para2</p></body></html>"
        expected = "# Title\n\nPara1\n\nPara2"

        # When: Process to Markdown
        result = _process_to_markdown(raw_content)

        # Then: Assert the full cleaned Markdown matches expected
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_visit_webpage_integration(self, requests_mock):
        # Given: A valid URL with mock HTML
        url = "https://example.com"
        mock_html = "<html><body><h1>Test Page</h1><p>This is a test.</p></body></html>"
        expected = "# Test Page\n\nThis is a test."

        requests_mock.get(url, text=mock_html, status_code=200)

        # When: Call the full tool
        result = visit_webpage(url)

        # Then: Assert the full Markdown content matches expected
        assert result == expected, f"Expected '{expected}', but got '{result}'"


class TestAgentCreation:
    def test_create_web_agent(self, mock_config):
        # Given: A mock model and custom tools for real-world override example
        model = MockModel()
        custom_tools: list = [MockModel()]  # Example custom tool list override
        expected_type = ToolCallingAgent
        expected_name = "web_search_agent"
        expected_description_contains = "web searches"

        # When: Create the agent with overrides
        agent = create_web_agent(
            model=model, tools=custom_tools, max_steps=mock_config["max_steps"]
        )

        # Then: Assert type, tools (full list), name, and description
        assert isinstance(agent, expected_type), (
            f"Expected type {expected_type}, but got {type(agent)}"
        )
        assert agent.tools == custom_tools, (
            f"Expected tools {custom_tools}, but got {agent.tools}"
        )
        assert agent.name == expected_name
        assert expected_description_contains in agent.description

    def test_create_manager_agent(self, mock_config):
        # Given: A mock model, managed agents, and custom instructions for override example
        model = MockModel()
        mock_managed_agents: list = [
            MockModel()
        ]  # Real-world example: single managed agent
        custom_instructions = "Custom test instructions with deep thinking."
        expected_type = CodeAgent
        expected_imports: list[str] = ["time", "numpy", "pandas"]
        expected_instructions_contains = "Custom test"

        # When: Create the agent with overrides
        agent = create_manager_agent(
            model=model,
            managed_agents=mock_managed_agents,
            instructions=custom_instructions,
            executor_type=mock_config["executor_type"],
        )

        # Then: Assert type, managed agents (full list), imports (full list), and instructions
        assert isinstance(agent, expected_type), (
            f"Expected type {expected_type}, but got {type(agent)}"
        )
        assert agent.managed_agents == mock_managed_agents, (
            f"Expected managed agents {mock_managed_agents}, but got {agent.managed_agents}"
        )
        assert agent.additional_authorized_imports == expected_imports, (
            f"Expected imports {expected_imports}, but got {agent.additional_authorized_imports}"
        )
        assert expected_instructions_contains in agent.instructions, (
            "Instructions should reflect override"
        )


class TestDisplayConfigTable:
    def test_display_config_table(self, mock_config, capsys):
        # Given: A sample config for real-world display
        expected_keys = list(mock_config.keys())
        expected_values = list(map(str, mock_config.values()))

        # When: Display the table (captures print)
        display_config_table(mock_config)
        captured = capsys.readouterr()

        # Then: Assert captured output contains expected keys and values (full check via string presence)
        for key, value in zip(expected_keys, expected_values):
            assert key in captured.out, f"Expected key '{key}' in output"
            assert value in captured.out, f"Expected value '{value}' in output"
