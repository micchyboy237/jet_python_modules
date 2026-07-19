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

# 1. Setup sample data (mix of unique and near-duplicates)
# dataset = [
#     "The quick brown fox jumps over the lazy dog.",
#     "The quick brown fox jumped over a lazy dog!",  # Near duplicate
#     "Artificial intelligence is changing software development.",
#     "AI is transforming how we write code.",  # Semantically similar
#     "Baking bread requires flour, water, and yeast.",
# ]

# 2. Fit the index and run an initial, aggressive deduplication
console.print("Initializing SemHash and running initial deduplication...")
sh = SemHash.from_records(sample_records)
result = sh.self_deduplicate(threshold=0.80)

# Save initial deduplication results
initial_results_path = OUTPUT_DIR / "initial_deduplication.json"
initial_results = {
    "threshold": 0.80,
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
}
with open(initial_results_path, "w") as f:
    json.dump(initial_results, f, indent=2, default=str)
console.print(
    f"Initial deduplication results saved to [link=file://{initial_results_path}]{initial_results_path.name}[/link]"
)

# 3. AUDIT: Use get_least_similar_from_duplicates to check our work
console.print("Auditing the borderline duplicates...")
borderline_pairs = result.get_least_similar_from_duplicates(n=5)

# Save borderline pairs audit
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

# 4. ADJUST: If the audit shows unique data was dropped, tighten the threshold
# Let's say we notice 0.83 score swapped out completely distinct sentences.
console.print(f"Original kept count: {len(result.selected)}")  # Output: fewer records

console.print("Tightening threshold from 0.80 to 0.98...")
result.rethreshold(threshold=0.98)

console.print(
    f"Adjusted kept count: {len(result.selected)}"
)  # Output: records restored!

# Save adjusted deduplication results
adjusted_results_path = OUTPUT_DIR / "adjusted_deduplication.json"
adjusted_results = {
    "threshold": 0.98,
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
        "adjusted_threshold": 0.98,
        "records_restored": len(result.selected)
        - len(initial_results["selected_records"]),
    },
}
with open(adjusted_results_path, "w") as f:
    json.dump(adjusted_results, f, indent=2, default=str)
console.print(
    f"Adjusted deduplication results saved to [link=file://{adjusted_results_path}]{adjusted_results_path.name}[/link]"
)

# Save selected records with their duplicates mapping
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

# Save complete summary
summary_path = OUTPUT_DIR / "deduplication_summary.json"
summary = {
    "total_original_records": len(sample_records),
    "initial_deduplication": {
        "threshold": 0.80,
        "kept_records": len(initial_results["selected_records"]),
        "removed_records": len(initial_results["filtered_records"]),
        "duplicate_ratio": result.duplicate_ratio,
    },
    "adjusted_deduplication": {
        "threshold": 0.98,
        "kept_records": len(result.selected),
        "removed_records": len(result.filtered),
        "duplicate_ratio": result.duplicate_ratio,
    },
    "optimization": {
        "records_restored": len(result.selected)
        - len(initial_results["selected_records"]),
        "removal_rate_initial": f"{(len(initial_results['filtered_records']) / len(sample_records)) * 100:.1f}%",
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
