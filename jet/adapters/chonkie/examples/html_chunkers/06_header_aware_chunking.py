"""Header-aware chunking demo: complex H1-H6 hierarchy with mixed HTML elements."""

import json
import shutil
from pathlib import Path

from jet.adapters.chonkie.html_chunkers import ScrapedHTMLPipeline
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = ScrapedHTMLPipeline(chunk_size=256, table_rows_per_chunk=3)

complex_html = """
<html>
<body>

<h1>Platform Engineering Handbook</h1>
<p>This handbook covers the full lifecycle of our internal developer platform,
from architecture decisions through production operations.</p>

<h2>Architecture Overview</h2>
<p>The platform follows a service-oriented architecture with event-driven communication.</p>

<h3>Core Services</h3>
<ul>
  <li><strong>Auth Service</strong> – OAuth2/OIDC provider with RBAC</li>
  <li><strong>Pipeline Orchestrator</strong> – DAG-based CI/CD engine</li>
  <li><strong>Artifact Registry</strong> – OCI-compliant container and package store</li>
</ul>

<h4>Service Communication Patterns</h4>
<p>All inter-service calls use gRPC with protobuf schemas versioned via Buf.</p>
<table>
  <tr><th>Pattern</th><th>Use Case</th><th>Technology</th></tr>
  <tr><td>Synchronous RPC</td><td>Auth checks, config lookups</td><td>gRPC</td></tr>
  <tr><td>Async Events</td><td>Pipeline triggers, audit logs</td><td>NATS JetStream</td></tr>
  <tr><td>Pub/Sub Fan-out</td><td>Cache invalidation, notifications</td><td>Redis Streams</td></tr>
</table>

<h5>Retry and Backoff Policy</h5>
<p>All RPC clients implement exponential backoff with jitter. Maximum retry count
is configurable per-service via environment variables.</p>
<pre><code class="python">
import random, time

def backoff(attempt: int, base: float = 0.5, cap: float = 30.0) -> float:
    delay = min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, delay * 0.1)
</code></pre>

<h6>Timeout Configuration Matrix</h6>
<table>
  <tr><th>Service</th><th>Connect Timeout</th><th>Read Timeout</th><th>Retries</th></tr>
  <tr><td>Auth</td><td>2s</td><td>5s</td><td>3</td></tr>
  <tr><td>Orchestrator</td><td>5s</td><td>30s</td><td>2</td></tr>
  <tr><td>Registry</td><td>3s</td><td>60s</td><td>3</td></tr>
</table>

<h3>Data Storage Strategy</h3>
<p>We use polyglot persistence optimized for each workload's access patterns.</p>

<h4>Primary Databases</h4>
<ol>
  <li><strong>PostgreSQL 16</strong> – Transactional data, user accounts, permissions</li>
  <li><strong>ClickHouse</strong> – Telemetry, pipeline execution metrics, audit logs</li>
  <li><strong>MinIO</strong> – Build artifacts, test reports, documentation assets</li>
</ol>

<h5>Schema Migration Workflow</h5>
<p>All schema changes go through a three-stage process:</p>
<blockquote>
  Stage 1: Expand – Add new columns/tables alongside existing ones.<br/>
  Stage 2: Migrate – Dual-write and backfill historical data.<br/>
  Stage 3: Contract – Remove deprecated columns after validation period.
</blockquote>

<h6>Migration Safety Checklist</h6>
<ul>
  <li>Backward-compatible DDL only (no column renames or type narrowing)</li>
  <li>Index creation uses <code>CONCURRENTLY</code> to avoid locking</li>
  <li>Rollback script tested against staging snapshot before production deploy</li>
  <li>Data integrity verification query included in migration PR template</li>
</ul>

<h2>Deployment Procedures</h2>
<p>All deployments are GitOps-managed via ArgoCD with progressive delivery.</p>

<h3>Environment Promotion Path</h3>
<p>Changes flow through four environments with increasing validation gates.</p>
<img src="/diagrams/promotion-path.svg" alt="Environment promotion diagram" />

<h4>Canary Analysis Criteria</h4>
<p>Automated canary analysis compares error rate, latency P99, and throughput
against baseline using Mann-Whitney U statistical tests.</p>
<pre><code class="yaml">
canary:
  analysis:
    interval: 60s
    threshold: 95
    metrics:
      - name: error-rate
        type: gauge
        max: 0.01
      - name: latency-p99
        type: histogram
        max: 500ms
</code></pre>

<h5>Rollback Triggers</h5>
<table>
  <tr><th>Condition</th><th>Threshold</th><th>Action</th></tr>
  <tr><td>Error Rate Spike</td><td>&gt; 1% over 5min</td><td>Automatic rollback</td></tr>
  <tr><td>Latency Degradation</td><td>P99 &gt; 2x baseline</td><td>Pause + alert</td></tr>
  <tr><td>Health Check Failures</td><td>&gt; 3 consecutive</td><td>Automatic rollback</td></tr>
</table>

<h6>Emergency Override Procedure</h6>
<p>In case of automated system failure, on-call engineers can trigger manual
rollback via the ops dashboard or CLI:</p>
<pre><code class="bash">
platform-cli deploy rollback \
  --service auth-service \
  --to-revision abc1234 \
  --reason "Manual override: elevated 5xx errors"
</code></pre>

<footer>© 2025 Platform Engineering Team – Internal Use Only</footer>
</body>
</html>
"""

results = pipeline.process(
    complex_html,
    source_url="https://internal.example.com/platform-handbook",
)

output_file = OUTPUT_DIR / "header_aware_chunks.json"
serializable = [
    {
        "element_category": r.element_category,
        "breadcrumb": r.breadcrumb,
        "source_url": r.source_url,
        "page_number": r.page_number,
        "text": r.chunk.text,
        "token_count": r.chunk.token_count,
        "start_index": r.chunk.start_index,
        "end_index": r.chunk.end_index,
    }
    for r in results
]
output_file.write_text(json.dumps(serializable, indent=2))

table = Table(title="Header-Aware Chunking Results")
table.add_column("#", style="dim", width=3)
table.add_column("Category", style="cyan", max_width=18)
table.add_column("Breadcrumb", style="yellow", max_width=45)
table.add_column("Tokens", justify="right", width=6)
table.add_column("Preview", max_width=50)

for i, item in enumerate(results):
    preview = item.chunk.text[:50].strip().replace("\n", " ") + "…"
    table.add_row(
        str(i),
        item.element_category,
        item.breadcrumb or "(none)",
        str(item.chunk.token_count),
        preview,
    )

console.print(table)
console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
