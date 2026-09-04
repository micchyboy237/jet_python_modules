"""Partition HTML pages from URLs and save extracted elements as JSON files.

Usage:
    # Use default URLs
    python script.py

    # Specify custom URLs
    python script.py "https://example.com/page1" "https://example.com/page2"

    # Custom output directory
    python script.py -o ./my_output
"""

import argparse
import shutil
from pathlib import Path

from jet.file.utils import save_file
from unstructured.partition.html import partition_html

DEFAULT_URLS = [
    "https://www.onlinejobs.ph/jobseekers/jobsearch/0?jobkeyword=AI+Python",
    "https://www.onlinejobs.ph/jobseekers/job/senior-python-automation-engineer-ai-powered-lead-generation-platform-experienced-developers-only-1722557",
    "https://docs.unstructured.io",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Partition HTML pages and save elements as JSON files."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        default=DEFAULT_URLS,
        help="One or more URLs to process (default: predefined list)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./generated/<script_stem>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = (
        args.output_dir or Path(__file__).parent / "generated" / Path(__file__).stem
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, url in enumerate(args.urls):
        print(f"Processing [{idx + 1}/{len(args.urls)}]: {url}")
        elements = partition_html(url=url)

        filename = f"elements_{idx + 1}.json"
        save_file(elements, output_dir / filename)
        print(f"  Saved {len(elements)} elements to {filename}")

    print(f"\nAll done! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
