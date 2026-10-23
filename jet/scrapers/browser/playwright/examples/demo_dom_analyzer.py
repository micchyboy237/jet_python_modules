"""
Demo: Using DOM Analyzer for Agentic Web Surfing
Shows how an LLM can use pattern recognition and structural mapping
to decide how to scrape a page.
"""

import asyncio
import json
import shutil
from pathlib import Path

from jet.scrapers.browser.playwright.dom_analyzer import (
    analyze_patterns,
    analyze_structure,
)
from rich.console import Console

console = Console()

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Tech News</title></head>
<body>
    <nav>...</nav>
    <main>
        <h1>Latest Articles</h1>
        <div class="article-grid">
            <div class="card">
                <h2>AI Breakthrough</h2>
                <p>Details about AI...</p>
            </div>
            <div class="card">
                <h2>Quantum Computing</h2>
                <p>Details about Quantum...</p>
            </div>
            <div class="card">
                <h2>Robotics Update</h2>
                <p>Details about Robots...</p>
            </div>
        </div>
        <aside>Sidebar ads...</aside>
    </main>
</body>
</html>
"""

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def demo_pattern_recognition():
    console.rule("[bold blue]Demo 1: Pattern Recognition Engine[/bold blue]")
    url = "https://quotes.toscrape.com/"
    console.print(f"Analyzing URL: {url}")

    # Use the helper function which handles threading and object conversion
    patterns = await analyze_patterns(
        source=url,
        match_type="class",
        headless=False,
    )

    console.print("[bold green]Detected Patterns:[/bold green]")
    for p in patterns[:3]:
        # Access as dictionary keys since analyze_patterns returns List[PatternResult] (TypedDict)
        console.print(
            f"  - Parent: <{p['parent_tag']}> with {p['children_count']} children"
        )
        console.print(f"    Shared Class: .{p['common_class']}")
        console.print(f"    Context: '{p['sample_text']}'")


async def demo_structural_map():
    """Demonstrates the 'Structural Map' & 'Density Analyzer'."""
    console.rule("[bold blue]Demo 2: Structural Map & Density Analyzer[/bold blue]")
    console.print("Analyzing HTML for dense content areas...")

    structure = await analyze_structure(
        source=SAMPLE_HTML,
        min_children=2,
        headless=False,
    )

    output_file = OUTPUT_DIR / "structure.json"
    output_file.write_text(json.dumps(structure, indent=2), encoding="utf-8")

    console.print("[bold green]Dense Nodes Identified:[/bold green]")
    for s in structure:
        console.print(f"  - Node: <{s['tag']}> (ID: {s['node_id']})")
        console.print(f"    Depth: {s['depth']} | Children: {s['child_count']}")
        if s["child_css_selectors"]:
            console.print(f"    Top Selector: {s['child_css_selectors'][0]}")

    # Fixed typo: [dimm] -> [/dimm]
    console.print(f"\n[dimm]Results saved to: {output_file}[/dimm]")


async def demo_agentic_workflow():
    """Simulates an LLM Agent deciding how to scrape based on analysis."""
    console.rule("[bold magenta]Demo 3: Agentic Decision Workflow[/bold magenta]")
    console.print("[bold]LLM Prompt Simulation:[/bold]")
    console.print("'I need to extract all article titles from this page.'")
    console.print()

    console.print("[dim]Agent runs analyze_patterns()...[/dim]")
    patterns = await analyze_patterns(SAMPLE_HTML, match_type="class", headless=False)

    if patterns:
        best_pattern = patterns[0]
        console.print(
            f"[green]✓ Agent identified a list pattern:[/green] .{best_pattern['common_class']}"
        )
        console.print(
            f"   Context: '{best_pattern['sample_text']}' looks like articles."
        )

        console.print("\n[bold]Agent Decision:[/bold]")
        console.print(
            f"   Use CSS selector: '.{best_pattern['common_class']} > h2' to extract titles."
        )
        console.print(f"   Expected items: {best_pattern['children_count']}")
    else:
        console.print(
            "[yellow]⚠ No clear patterns found. Falling back to structural density analysis.[/yellow]"
        )
        structure = await analyze_structure(SAMPLE_HTML, min_children=2, headless=False)
        if structure:
            main_node = structure[0]
            console.print(f"   Focusing on dense node: <{main_node['tag']}>")


async def main():
    await demo_pattern_recognition()
    console.print()
    await demo_structural_map()
    console.print()
    await demo_agentic_workflow()


if __name__ == "__main__":
    asyncio.run(main())
