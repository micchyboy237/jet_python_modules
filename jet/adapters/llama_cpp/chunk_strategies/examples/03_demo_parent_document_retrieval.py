"""Demo: Parent-Document Retrieval chunking via ParentDocumentChunker.

Shows linked parent-child pair generation with auto-derived chunk sizes,
validates linkage integrity, and demonstrates multi-parent retrieval resolution.

Chunk sizes are NOT hardcoded — they are auto-derived from the configured
LLM and embedding model contexts via ParentDocumentChunker internals.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.chunk_strategies import estimate_tokens_safe, get_chunker
from jet.adapters.llama_cpp.config import LLM_MODEL
from rich.console import Console

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = LLM_MODEL

# ~2800 tokens — guarantees multiple parents with 1024-token budget
SAMPLE_TEXT = """\
# API Authentication & Security Guide

## Overview
The API uses Bearer token authentication for all endpoints. Every request must include
an Authorization header containing a valid JWT access token obtained from the /auth/login
endpoint. Access tokens expire after 24 hours and must be refreshed using the refresh
token flow described below. All API responses include an X-Request-Id header for
distributed tracing, debugging, and correlation across microservices. The platform
supports OAuth 2.0 authorization code flow for third-party integrations and SAML 2.0
for enterprise single sign-on. Rate limiting is enforced globally at 1000 requests per
minute per API key, with per-endpoint limits documented in each endpoint reference.

## Obtaining Tokens
Send a POST request to /auth/login with your email and password in the JSON request body.
The response includes an access_token (short-lived, 24-hour TTL) and a refresh_token
(long-lived, 30-day TTL). Store both tokens securely using encrypted storage, hardware
security modules, or environment variables injected at runtime. Never expose tokens in
client-side JavaScript, URL query parameters, server access logs, or error messages
returned to end users. The login endpoint enforces rate limiting: maximum 10 attempts
per minute per source IP address. After 5 consecutive failed authentication attempts,
the account is temporarily locked for 15 minutes and a security alert is generated.
Multi-factor authentication can be enabled per-user via PUT /auth/mfa/enroll, which
requires TOTP verification on subsequent logins. Password policy requires minimum 12
characters with uppercase, lowercase, digit, and special character classes.

## Refresh Token Flow
When the access token expires, send a POST request to /auth/refresh with the current
refresh_token in the Authorization header using the Bearer scheme. The server validates
the refresh token signature, expiration, and revocation status before issuing a new
access_token. For enhanced security, the server may optionally rotate the refresh_token,
returning a new refresh_token alongside the new access_token. If rotation occurs, the
previous refresh_token is immediately invalidated and any subsequent use triggers
session revocation. Clients must detect refresh token rotation by comparing the returned
token value with the one originally sent. Always update stored credentials atomically
using database transactions or file locks to prevent race conditions during concurrent
refresh attempts. Implement exponential backoff with jitter for retry logic when refresh
requests fail due to transient network errors or server overload conditions.

## Token Revocation & Session Management
Users can revoke all their active sessions by sending DELETE to /auth/sessions, which
invalidates every access_token and refresh_token associated with their account.
Administrators can revoke specific user sessions via DELETE /admin/users/{id}/sessions
with optional reason codes for audit compliance. Revoked tokens return HTTP 401
Unauthorized with error_code "token_revoked" on any subsequent API call. All token
revocation events are logged to the centralized audit trail with precise timestamp,
source IP address, user agent string, and geolocation data. Session binding ties tokens
to the originating device fingerprint and IP subnet, triggering automatic revocation
when requests originate from unexpected locations or devices. Idle session timeout is
configurable per-tenant, defaulting to 8 hours of inactivity before forced re-authentication.

## Error Handling Standards
All authentication failures return HTTP 401 Unauthorized with structured JSON error
bodies following RFC 7807 Problem Details format. Expired access tokens return
error_code "token_expired" with a suggested action of "refresh". Invalid or tampered
refresh tokens return error_code "invalid_refresh_token" and immediately revoke all
active sessions for that user as a security precaution. Malformed Authorization headers
return error_code "malformed_auth_header" with details about the expected format.
Rate limit violations return HTTP 429 Too Many Requests with a Retry-After header
indicating seconds until the next allowed request. All error responses include a unique
error_id UUID for support ticket correlation and log searching. Client SDKs should
implement automatic token refresh on 401 responses with error_code "token_expired",
falling back to redirect-to-login on "invalid_refresh_token" or "token_revoked".

## Security Best Practices & Compliance
Always enforce HTTPS/TLS 1.3 in production environments; plaintext HTTP connections
are rejected at the load balancer level. Set Secure, HttpOnly, and SameSite=Strict
flags on any cookie-stored tokens to prevent XSS and CSRF attacks. Implement token
binding using TLS channel binding or DPoP proofs to prevent token theft via
man-in-the-middle attacks or token replay. Rotate JWT signing keys quarterly and
maintain a versioned key history to allow graceful migration without invalidating
in-flight tokens. Monitor for anomalous token usage patterns including geographic
impossibility (login from two continents within minutes), unusual request volume spikes,
access to previously-unvisited endpoints, and off-hours activity for business-only accounts.
Integrate authentication logs with SIEM systems like Splunk or Datadog for real-time
alerting and forensic analysis. Conduct annual penetration testing of the authentication
surface and remediate findings within 30 days for critical and 90 days for high severity.
Maintain SOC 2 Type II compliance documentation covering access control, audit logging,
and incident response procedures related to token management and session security.
"""


def main() -> None:
    console.print("\n[bold cyan]═══ Parent-Document Retrieval Demo ═══[/]\n")

    # Auto-derived sizes: pass None to use model-aware defaults
    # For qwen3.5-uncensored:2b (16384 ctx): parent=1024, child=128
    # Child further capped by embed model context if needed
    chunker = get_chunker("pdr", model=MODEL)
    result = chunker.chunk_pdr(
        text=SAMPLE_TEXT,
        parent_chunk_size=450,  # Demo-only: forces multi-parent splitting
        child_chunk_size=120,  # Demo-only: visible child granularity
        chunk_overlap=0,  # Parent provides boundary context
    )

    parents = result["parents"]
    children = result["children"]

    # ── Display Parents ──────────────────────────────────────────────
    console.print(f"[bold green]📦 Parents ({len(parents)}):[/]")
    for p in parents:
        tok = p.get("num_tokens") or estimate_tokens_safe(p["content"], MODEL)
        n_children = len(p["child_ids"])
        preview = p["content"][:100].replace("\n", "\\n")
        console.print(
            f"  [{p['id']}] {tok:>3d} tok | {n_children} children | {preview}..."
        )

    # ── Display Children ─────────────────────────────────────────────
    console.print(f"\n[bold yellow]🔍 Children ({len(children)}):[/]")
    for c in children:
        tok = estimate_tokens_safe(c["content"], MODEL)
        preview = c["content"][:80].replace("\n", "\\n")
        console.print(
            f"  [{c['id']}] → {c['parent_id']} | {tok:>3d} tok | {preview}..."
        )

    # ── Validate Linkage Integrity ───────────────────────────────────
    parent_map = {p["id"]: p for p in parents}
    errors = []
    for c in children:
        pid = c["parent_id"]
        if pid not in parent_map:
            errors.append(f"Child {c['id']} references missing parent {pid}")
        elif c["id"] not in parent_map[pid]["child_ids"]:
            errors.append(f"Parent {pid} missing child {c['id']} in child_ids")

    console.print("\n[bold]🔗 Linkage Validation:[/]")
    if errors:
        for e in errors:
            console.print(f"  [red]❌ {e}[/]")
    else:
        console.print("  [green]✅ All parent↔child links valid[/]")

    # ── Simulate Cross-Parent Retrieval Resolution ───────────────────
    # Pick children spanning multiple parents to demonstrate deduplication
    simulated_hits = children[:6]
    seen_parents: set[str] = set()
    resolved = []
    for hit in simulated_hits:
        pid = hit["parent_id"]
        if pid not in seen_parents:
            seen_parents.add(pid)
            resolved.append(parent_map[pid])

    console.print(
        f"\n[bold magenta]🎯 Simulated Retrieval "
        f"({len(simulated_hits)} child hits → {len(resolved)} unique parents):[/]"
    )
    for r in resolved:
        tok = r.get("num_tokens") or estimate_tokens_safe(r["content"], MODEL)
        console.print(f"  ✅ {r['id']} ({tok} tok)")

    # ── Save Results ─────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "pdr_result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"\n💾 Saved to [link=file://{out_path}]{out_path.name}[/]")


if __name__ == "__main__":
    main()
