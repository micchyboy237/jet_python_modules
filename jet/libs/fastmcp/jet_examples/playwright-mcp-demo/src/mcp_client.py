from typing import Any, Dict, Optional
import requests
import uuid
from rich.console import Console
from rich.panel import Panel

console = Console()


class MCPClient:
    """Low-level JSON-RPC client for Playwright MCP server"""

    def __init__(self, base_url: str = "http://localhost:8931/mcp"):
        self.base_mcp_url = base_url.rstrip("/")
        self.messages_url = f"{self.base_mcp_url.rstrip('/mcp')}/messages"
        self.sse_url = f"{self.base_mcp_url.rstrip('/mcp')}/sse"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })
        self.session_id: str = str(uuid.uuid4())
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        console.print(Panel("[cyan]Starting MCP session handshake[/cyan]", expand=False))

        console.print(f"[dim]Using session ID: {self.session_id}[/dim]")

        sse_url_with_id = f"{self.sse_url}?sessionId={self.session_id}"
        console.print(f"[dim]→ Establishing SSE transport: GET {sse_url_with_id}[/dim]")

        try:
            self.sse_response = self.session.get(sse_url_with_id, timeout=10, stream=True)
            console.print(f"[dim]SSE GET → {self.sse_response.status_code}[/dim]")
            if self.sse_response.status_code != 200:
                raise RuntimeError(f"SSE connection failed: {self.sse_response.status_code}")
            console.print("[green]✓ SSE transport established[/green]")

            # Optional: small sleep to give server time to register transport (usually instant)
            import time
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]SSE setup failed:[/red] {e}")
            raise

        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "custom-python-client",
                    "version": "0.1.0"
                }
            }
        }

        url = f"{self.messages_url}?sessionId={self.session_id}"
        console.print(f"[dim]→ POST initialize → {url}[/dim]")

        try:
            response = self.session.post(url, json=payload, timeout=15)
            console.print(f"[dim]Initialize → {response.status_code}[/dim]")
            console.print("[dim]Headers:[/dim]", dict(response.headers))
            console.print("[dim]Body:[/dim]", response.text[:400] if response.text else "<empty>")
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise RuntimeError(f"Initialize error: {result['error']}")
        except Exception as e:
            console.print(f"[red]Initialize POST failed:[/red] {e}")
            raise

        self.call("initialized", {}, expect_result=False)

        console.print("[bold green]✓ Session initialized[/bold green]")
        self._initialized = True

    def call(self, method: str, params: Optional[Dict] = None, expect_result: bool = True) -> Any:
        if not self._initialized and method not in ("initialize", "initialized"):
            self.initialize()

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 42 if expect_result else None,
        }

        url = f"{self.messages_url}?sessionId={self.session_id}"
        console.print(f"[dim]→ {method} ({'notification' if payload['id'] is None else 'request'}) → {url}[/dim]")

        try:
            response = self.session.post(url, json=payload, timeout=30)
            console.print(f"[dim]Response: {response.status_code}[/dim]")
            if response.text.strip():
                console.print("[dim]Body snippet:[/dim]", response.text[:300])
            response.raise_for_status()

            if payload["id"] is None:
                return None

            result = response.json()
        except Exception as e:
            console.print(f"[red]{method} failed:[/red] {e}")
            raise

        if "error" in result:
            raise RuntimeError(result["error"]["message"])

        return result.get("result") if expect_result else None

    # Optional: add a close method to clean up SSE if needed
    def close(self):
        if hasattr(self, 'sse_response') and not self.sse_response.raw.closed:
            self.sse_response.close()