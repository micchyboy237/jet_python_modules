import json
import os
import shutil
from pathlib import Path

from rich.console import Console
from semhash import SemHash

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = os.path.join(os.path.dirname(__file__), "mocks")
sample_data_path = os.path.join(BASE_DIR, "10_data.json")

with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

console.print("Initializing SemHash and running initial deduplication...")
sh = SemHash.from_records(sample_records)
result = sh.self_deduplicate(threshold=0.80)

# --- FIX: snapshot state BEFORE rethreshold() mutates result.selected/.filtered in place. ---
# result.selected is a live list object and duplicate_ratio/exact_duplicate_ratio are
# properties computed from the CURRENT state of result. If we don't capture them now,
# calling result.rethreshold(...) later will silently change these values out from
# under us, even though we already "saved" them into initial_results.
initial_selected_records = list(result.selected)  # copy, not a reference
initial_filtered = list(result.filtered)  # copy, not a reference
initial_duplicate_ratio = result.duplicate_ratio
initial_exact_duplicate_ratio = result.exact_duplicate_ratio
console.print(
    f"[dim][log] Snapshot captured before rethreshold: "
    f"{len(initial_selected_records)} selected, {len(initial_filtered)} filtered "
    f"@ threshold=0.80[/dim]"
)

initial_results_path = OUTPUT_DIR / "initial_deduplication.json"
initial_results = {
    "threshold": 0.80,
    "kept_count": len(initial_selected_records),
    "filtered_count": len(initial_filtered),
    "duplicate_ratio": initial_duplicate_ratio,
    "exact_duplicate_ratio": initial_exact_duplicate_ratio,
    "selected_records": initial_selected_records,
    "filtered_records": [
        {
            "record": dup.record,
            "exact": dup.exact,
            "duplicates": [
                {"duplicate_record": d, "score": score} for d, score in dup.duplicates
            ],
        }
        for dup in initial_filtered
    ],
}
with open(initial_results_path, "w") as f:
    json.dump(initial_results, f, indent=2, default=str)
console.print(
    f"Initial deduplication results saved to [link=file://{initial_results_path}]{initial_results_path.name}[/link]"
)

console.print("Auditing the borderline duplicates...")
borderline_pairs = result.get_least_similar_from_duplicates(n=5)
audit_results_path = OUTPUT_DIR / "borderline_audit.json"
audit_results = {"threshold": 0.80, "borderline_pairs": []}
for original, duplicate, score in borderline_pairs:
    console.print(f"Score: {score:.4f}")
    console.print(f" -> Kept: '{original}'")
    console.print(f" -> Dropped: '{duplicate}'\n")
    audit_results["borderline_pairs"].append(
        {
            "similarity_score": score,
            "kept_record": original,
            "dropped_record": duplicate,
        }
    )
with open(audit_results_path, "w") as f:
    json.dump(audit_results, f, indent=2, default=str)
console.print(
    f"Borderline audit results saved to [link=file://{audit_results_path}]{audit_results_path.name}[/link]"
)

console.print(f"Original kept count: {len(initial_selected_records)}")
console.print("Tightening threshold from 0.80 to 0.90...")
result.rethreshold(threshold=0.90)
console.print(
    f"[dim][log] rethreshold(0.90) applied. result.selected now has {len(result.selected)} items[/dim]"
)
console.print(f"Adjusted kept count: {len(result.selected)}")

adjusted_results_path = OUTPUT_DIR / "adjusted_deduplication.json"
adjusted_results = {
    "threshold": 0.90,
    "kept_count": len(result.selected),
    "filtered_count": len(result.filtered),
    "duplicate_ratio": result.duplicate_ratio,
    "exact_duplicate_ratio": result.exact_duplicate_ratio,
    "selected_records": result.selected,
    "filtered_records": [
        {
            "record": dup.record,
            "exact": dup.exact,
            "duplicates": [
                {"duplicate_record": d, "score": score} for d, score in dup.duplicates
            ],
        }
        for dup in result.filtered
    ],
    "comparison": {
        "initial_threshold": 0.80,
        "adjusted_threshold": 0.90,
        # FIX: compare against the snapshot taken before rethreshold(), not the
        # (now-mutated) initial_results["selected_records"].
        "records_restored": len(result.selected) - len(initial_selected_records),
    },
}
with open(adjusted_results_path, "w") as f:
    json.dump(adjusted_results, f, indent=2, default=str)
console.print(
    f"Adjusted deduplication results saved to [link=file://{adjusted_results_path}]{adjusted_results_path.name}[/link]"
)

selected_with_dupes_path = OUTPUT_DIR / "selected_with_duplicates.json"
selected_with_dupes = [
    {
        "selected_record": swd.record,
        "duplicates": [
            {"duplicate_record": d, "score": score} for d, score in swd.duplicates
        ],
    }
    for swd in result.selected_with_duplicates
]
with open(selected_with_dupes_path, "w") as f:
    json.dump(selected_with_dupes, f, indent=2, default=str)
console.print(
    f"Selected with duplicates mapping saved to [link=file://{selected_with_dupes_path}]{selected_with_dupes_path.name}[/link]"
)

summary_path = OUTPUT_DIR / "deduplication_summary.json"
records_restored = len(result.selected) - len(initial_selected_records)
console.print(
    f"[dim][log] records_restored = {len(result.selected)} (adjusted) - "
    f"{len(initial_selected_records)} (initial snapshot) = {records_restored}[/dim]"
)
summary = {
    "total_original_records": len(sample_records),
    "initial_deduplication": {
        "threshold": 0.80,
        # FIX: use the pre-rethreshold snapshot values, not the mutated result/initial_results.
        "kept_records": len(initial_selected_records),
        "removed_records": len(initial_filtered),
        "duplicate_ratio": initial_duplicate_ratio,
    },
    "adjusted_deduplication": {
        "threshold": 0.90,
        "kept_records": len(result.selected),
        "removed_records": len(result.filtered),
        "duplicate_ratio": result.duplicate_ratio,
    },
    "optimization": {
        "records_restored": records_restored,
        "removal_rate_initial": f"{(len(initial_filtered) / len(sample_records)) * 100:.1f}%",
        "removal_rate_adjusted": f"{(len(result.filtered) / len(sample_records)) * 100:.1f}%",
    },
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
console.print(
    f"Complete summary saved to [link=file://{summary_path}]{summary_path.name}[/link]"
)

console.print(
    f"\n[bold green]All results saved to [link=file://{OUTPUT_DIR}]{OUTPUT_DIR.name}[/link][/bold green]"
)
