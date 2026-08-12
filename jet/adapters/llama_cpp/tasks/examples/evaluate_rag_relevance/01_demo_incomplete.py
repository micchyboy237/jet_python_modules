from jet.adapters.llama_cpp.tasks.evaluate_rag_relevance import evaluate_rag_relevance
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Long context that PARTIALLY answers the query — demonstrates incomplete RAG evaluation
# Contains company-wide financials but lacks segment-specific EPS and net income
LONG_CONTEXT = """
SECTION 1: COMPANY HISTORY
Acme Corp was founded in 1985 by John Smith in Portland, Oregon. Originally a bicycle manufacturer,
the company pivoted to consumer electronics in 2003. Over the decades, Acme has won numerous awards
for design excellence and sustainability practices. The headquarters moved to Austin, Texas in 2015.

SECTION 2: PRODUCT LINE OVERVIEW
Acme currently offers three main product lines: SmartHome devices, WearableTech fitness trackers,
and EcoKitchen appliances. Each line emphasizes energy efficiency and user privacy. The SmartHome
line includes thermostats, cameras, and lighting systems compatible with major voice assistants.

SECTION 3: LEGAL DISCLAIMERS AND TERMS
This document contains forward-looking statements subject to risks and uncertainties. Actual results
may differ materially. All trademarks are property of their respective owners. Warranty periods vary
by region and product category. See acme.example.com/terms for full details. Limitation of liability
applies to all consumer products sold after January 1, 2024.

SECTION 4: Q3 2025 FINANCIAL HIGHLIGHTS
Total Company Revenue: $847 million (up 12% YoY)
Gross Margin: 34.2%
Operating Expenses: $215 million
Total Net Income: $78.3 million
Total EPS: $1.42

Segment Breakdown — Q3 2025:
• SmartHome: Revenue $381M (45% of total), YoY Revenue Growth +8%
• WearableTech: Revenue $127M (15% of total), YoY Revenue Growth +28%
• EcoKitchen: Revenue $339M (40% of total), YoY Revenue Growth -3%

SECTION 5: SUSTAINABILITY REPORT
Acme achieved carbon neutrality in Scope 1 and 2 emissions in 2024. Water usage reduced by 18%.
Recycled packaging now used across 92% of product lines. Employee volunteer hours exceeded 50,000.
The company partnered with OceanCleanup Initiative donating 1% of EcoKitchen profits.

SECTION 6: LEADERSHIP TEAM
CEO: Maria Chen (appointed 2022)
CFO: Robert Williams
CTO: Dr. Aisha Patel
VP Engineering: Carlos Rodriguez
Board Chair: Elizabeth Thompson
"""

TEST_QUERY = (
    "What was Acme Corp's Q3 2025 net income, EPS, and year-over-year "
    "revenue growth rate for the WearableTech segment specifically?"
)

console.print(
    "\n[bold green]RAG Relevance Evaluation — Incomplete Context Test[/bold green]"
)
console.print(Panel(TEST_QUERY, title="Query", border_style="cyan"))
console.print(
    f"[dim]Context length: {len(LONG_CONTEXT)} chars "
    f"(~{len(LONG_CONTEXT.split())} words)[/dim]\n"
)

result = evaluate_rag_relevance(TEST_QUERY, LONG_CONTEXT)

# Show decomposed queries first
if result["decomposed_queries"]:
    console.print("[bold cyan]Decomposed Queries:[/bold cyan]")
    for i, dq in enumerate(result["decomposed_queries"], 1):
        console.print(f"  {i}. {dq}")
    console.print()
else:
    console.print("[dim]Decomposed Queries: (none)[/dim]\n")

# Main result table
table = Table(show_header=True, header_style="bold magenta", show_lines=True)
table.add_column("Field", style="bold", width=15)
table.add_column("Value", style="white")

complete_str = (
    "[bold green]✓ Complete[/bold green]"
    if result["is_complete"]
    else "[bold red]✗ Incomplete[/bold red]"
)
valid_str = (
    "[bold green]✓ Valid[/bold green]"
    if result["is_valid"]
    else "[bold red]✗ Invalid[/bold red]"
)

table.add_row("Status", complete_str)
table.add_row("Confidence", f"{result['confidence']:.2f}")
table.add_row("Valid Output", valid_str)
if result["error"]:
    table.add_row("Error", f"[red]{result['error']}[/red]")

console.print(table)

# Completed info list
if result["completed_info"]:
    console.print("\n[bold green]✓ Completed Info:[/bold green]")
    for i, item in enumerate(result["completed_info"], 1):
        console.print(f"  {i}. {item}")
else:
    console.print("\n[dim]✓ Completed Info: (none)[/dim]")

# Missing info list
if result["missing_info"]:
    console.print("\n[bold red]✗ Missing Info:[/bold red]")
    for i, item in enumerate(result["missing_info"], 1):
        console.print(f"  {i}. {item}")
else:
    console.print(
        "\n[bold green]✗ Missing Info: (none — context is complete)[/bold green]"
    )
