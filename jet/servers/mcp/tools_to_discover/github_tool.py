# jet_python_modules/jet/servers/mcp/tools_to_discover/github_tool.py
# Demonstrates authenticated GitHub tool using user context
# Uses custom user class to store authentication tokens

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
import httpx
from typing import Optional

mcp = FastMCP(name="GitHub")


@mcp.tool()
async def create_issue(title: str, repo: str, ctx: Context[ServerSession, None], github_token: Optional[str] = None) -> str:
    """Create a GitHub issue.
    Args:
        title: Title of the GitHub issue
        repo: Repository name (e.g., owner/repo)
        ctx: Context object for session and request tracking
        github_token: GitHub API token for authentication (optional)
    Returns:
        String with issue creation result or error message
    """
    if not github_token:
        return "Error: GitHub token required for authentication."
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {"Authorization": f"Bearer {github_token}",
               "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json={"title": title})
        if response.status_code == 201:
            return f"Issue created: {response.json()['html_url']}"
        return f"Error: {response.text}"
