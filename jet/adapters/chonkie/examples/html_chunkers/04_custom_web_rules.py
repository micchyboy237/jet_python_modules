"""Demonstrate custom RecursiveRules that split on header boundaries."""

import json
import shutil
from pathlib import Path

from chonkie.types import RecursiveLevel, RecursiveRules
from jet.adapters.chonkie.html_chunkers import HTMLAwareChunker
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom rules: split aggressively on headings first, then paragraphs
# Level 0: H2/H3 boundaries (primary split points)
# Level 1: Paragraph breaks (secondary)
# Level 2: Sentence breaks (tertiary fallback)
# Level 3: Whitespace (last resort)
web_rules = RecursiveRules(
    levels=[
        RecursiveLevel(delimiters=["\n## ", "\n### ", "\n#### "], include_delim="next"),
        RecursiveLevel(delimiters=["\n\n"], include_delim="next"),
        RecursiveLevel(delimiters=[". ", "! ", "? "], include_delim="prev"),
        RecursiveLevel(whitespace=True),
    ]
)

# Use small chunk_size to force splitting even within sections
chunker = HTMLAwareChunker(chunk_size=128, min_chars_per_chunk=30)
chunker._recursive_chunker.rules = web_rules
chunker._recursive_chunker.min_characters_per_chunk = 30

# Long, complex markdown with deep hierarchy and mixed content types
messy_markdown = """
# Platform Documentation Guide

This comprehensive guide covers all aspects of our analytics platform, from initial setup 
through advanced configuration and enterprise deployment scenarios.

## Getting Started

Welcome to the platform documentation. This section walks you through the initial setup 
process, account creation, and your first data connection. Before proceeding, ensure you 
have administrator privileges on your target system and have reviewed the prerequisites 
document available in the resources section.

### System Requirements

Our platform supports Linux, macOS, and Windows Server environments. Minimum requirements 
include 8GB RAM, 4 CPU cores, and 50GB of available disk space for the base installation. 
For production deployments handling more than 1 million events per day, we recommend 32GB 
RAM and 16 CPU cores with SSD storage for optimal query performance.

### Installation Steps

Download the installer from your account dashboard under Settings > Downloads. Run the 
installer with administrative privileges and follow the on-screen prompts. The installation 
typically completes within 5-10 minutes depending on your system configuration. After 
installation, navigate to http://localhost:8080 to access the setup wizard.

## Data Connectors

The platform supports over 50 data connectors out of the box, covering databases, cloud 
storage, APIs, and streaming platforms. Each connector can be configured independently 
with its own scheduling, transformation pipeline, and error handling policies.

### Database Connectors

We support PostgreSQL, MySQL, MariaDB, SQL Server, Oracle, MongoDB, Redis, Elasticsearch, 
and ClickHouse. Connection pooling is enabled by default with configurable pool sizes. 
SSL/TLS encryption is supported for all database connectors and can be enforced at the 
organization level through security policies.

### Cloud Storage Connectors

Connect to Amazon S3, Google Cloud Storage, Azure Blob Storage, MinIO, and any S3-compatible 
object store. Files are processed using a streaming architecture that handles datasets 
larger than available memory. Supported formats include CSV, Parquet, JSON, Avro, ORC, 
and Delta Lake tables with automatic schema detection and evolution tracking.

### API Connectors

REST and GraphQL API connectors support OAuth2, API key, and JWT authentication methods. 
Rate limiting is handled automatically with exponential backoff and jitter. Response 
pagination is detected automatically for most common API patterns including cursor-based, 
offset-based, and link-header pagination strategies.

## Pricing Tiers

We offer three tiers designed to scale with your organization's needs. All tiers include 
unlimited users, API access, and community support. Billing is monthly or annual with 
a 20% discount for annual commitments.

### Free Tier

The free tier includes up to 10,000 events per month, 3 data connectors, 7-day data 
retention, and access to the community forum. It is ideal for evaluation, prototyping, 
and small personal projects. No credit card is required to get started.

### Pro Tier

The Pro tier includes up to 1 million events per month, unlimited data connectors, 
90-day data retention, priority email support, custom dashboards, alerting, and team 
collaboration features. Price is $99/month billed annually or $119/month billed monthly.

### Enterprise Tier

The Enterprise tier includes unlimited events, unlimited connectors, configurable data 
retention up to 7 years, dedicated account manager, SLA guarantees, SSO/SAML integration, 
audit logging, on-premise deployment option, and custom contract terms. Contact our sales 
team for pricing tailored to your organization's requirements.

## Advanced Configuration

This section covers advanced topics for power users and administrators who need fine-grained 
control over platform behavior, performance tuning, and integration with external systems.

### Performance Tuning

Query performance can be optimized through materialized views, partition pruning, and 
adaptive indexing. The query planner automatically selects optimal execution strategies 
based on data statistics collected during ingestion. For workloads exceeding 10 million 
events per hour, consider enabling distributed query execution across multiple nodes.

### Security Policies

Organization-level security policies control data access, encryption, and compliance 
settings. Role-based access control supports custom roles with granular permissions 
down to the column level. All data is encrypted at rest using AES-256 and in transit 
using TLS 1.3. Audit logs capture all data access events and are retained for the 
configured retention period.

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Events/month | 10K | 1M | Unlimited |
| Connectors | 3 | Unlimited | Unlimited |
| Retention | 7 days | 90 days | Configurable |
| Support | Community | Priority Email | Dedicated Manager |
| SSO/SAML | No | No | Yes |
| On-Premise | No | No | Yes |

```python
from platform_sdk import Client

client = Client(api_key="your-key-here")
connector = client.connectors.create(
    type="postgresql",
    host="db.example.com",
    database="analytics",
    schedule="*/5 * * * *",
)
print(f"Created connector: {connector.id}")
```

### Monitoring and Alerting

Built-in monitoring tracks ingestion throughput, query latency, storage utilization, 
and connector health. Alerts can be configured via email, Slack, PagerDuty, or webhook. 
Custom metrics can be exported to Prometheus, Datadog, or Grafana for unified observability 
alongside your existing infrastructure monitoring stack.
"""

chunks = chunker.chunk(messy_markdown)

# --- Save output ---
output_file = OUTPUT_DIR / "custom_rules_chunks.json"
serializable = [
    {
        "text": c.text.strip(),
        "token_count": c.token_count,
        "start_index": c.start_index,
        "end_index": c.end_index,
    }
    for c in chunks
]
output_file.write_text(json.dumps(serializable, indent=2))

# --- Display summary ---
console.print(f"\n[bold]Total chunks:[/] {len(chunks)}")
for i, c in enumerate(chunks):
    preview = c.text.strip()[:80].replace("\n", " ")
    console.print(f"  [{i}] {c.token_count:>4} tokens | {preview}…")

# --- Display resource links ---
console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
