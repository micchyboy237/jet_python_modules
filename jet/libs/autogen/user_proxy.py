"""Define a simple UserProxyAgent in AutoGen v0.4.

This module illustrates the creation of a `UserProxyAgent` in AutoGen v0.4, which replaces the v0.2 `UserProxyAgent`. The agent is configured to handle user input without additional settings like human input mode or code execution, simplifying the v0.2 approach.
"""

from autogen_agentchat.agents import UserProxyAgent

user_proxy = UserProxyAgent("user_proxy")