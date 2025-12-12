# jet/audio/speech/pyannote/utils/export_plotly_timeline.py
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console

console = Console()
_R_SCRIPT = Path(__file__).with_name("generate_timeline.R")


def export_plotly_timeline(
    turns: List[Dict[str, Any]],
    total_seconds: float,
    audio_name: str,
    output_dir: Path,
    *,
    height_per_speaker: int = 60,
    base_height: int = 300,
    title: str | None = None,
) -> Path:
    """
    Generate interactive Plotly timeline using external generate_timeline.R script.
    Much cleaner, debuggable, and robust than f-string R code.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "timeline.html"

    if total_seconds <= 0:
        raise ValueError("total_seconds must be > 0")
    if not turns:
        console.log("[yellow]No turns → empty timeline[/]")

    for turn in turns:
        turn.setdefault("confidence", 0.0)

    # 1. Write clean JSON file
    turns_json_path = output_dir / "turns.json"
    turns_json_path.write_text(json.dumps(turns, indent=2))

    if not _R_SCRIPT.exists():
        raise FileNotFoundError(f"R script missing: {_R_SCRIPT}")

    # Create a local assets directory next to the HTML file
    assets_dir = output_dir / "plotly_assets"
    assets_dir.mkdir(exist_ok=True)
    libdir_for_this_run = str(assets_dir)

    console.log("[bold cyan]Generating interactive Plotly timeline via R...[/]")

    # Pass audio_name as 5th arg, and assets_dir as 6th argument
    cmd = [
        "Rscript", "--vanilla",
        str(_R_SCRIPT),
        str(turns_json_path),
        str(total_seconds),
        str(html_path),
        audio_name,
        libdir_for_this_run,  # now a subdirectory → relative paths work perfectly
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        console.log("[green]Success[/] → [bold]timeline.html[/] (uses per-export plotly_assets)")
        if result.stdout.strip():
            console.log(result.stdout.strip())
        return html_path

    except subprocess.CalledProcessError as e:
        console.log("[red]R script failed[/]")
        console.print(e.stderr.strip() or "No stderr")
        console.log(
            "[yellow]Tip:[/] Install packages with:\n"
            '    R -e "install.packages(c(\'plotly\',\'htmlwidgets\',\'jsonlite\'), '
            'repos=\'https://cloud.r-project.org\')"'
        )
        raise

    except FileNotFoundError:
        console.log("[red]Rscript not found in PATH[/]")
        console.log("→ https://cran.r-project.org/")
        raise

    # finally:
        # Optional: clean up intermediate JSON (uncomment if desired)
        # turns_json_path.unlink(missing_ok=True)