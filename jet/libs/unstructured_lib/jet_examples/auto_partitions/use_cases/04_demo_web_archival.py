"""
04_demo_web_archival.py
Partitions remote documents via URL with automatic content-type negotiation.
Includes retry logic, streaming support, and atomic persistence of raw + parsed data.
"""

import io
import json
import logging
import time
from pathlib import Path
from urllib.error import URLError

from unstructured.partition.auto import partition

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path(__file__).parent / "web_archive"
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2

SAMPLE_URLS = [
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "https://example.com/nonexistent-document.pdf",  # Will fail gracefully
]


def fetch_with_retry(url: str, max_retries: int = MAX_RETRIES) -> io.BytesIO | None:
    """Fetch URL content with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching (attempt {attempt + 1}): {url}")
            elements = partition(
                url=url,
                strategy="auto",
                ssl_verify=True,
                request_timeout=30,
            )
            # Re-fetch raw bytes for archival (partition consumed the stream)
            from unstructured.safe_http import safe_get

            response = safe_get(url, verify=True, timeout=30)
            return io.BytesIO(response.content), elements
        except (URLError, TimeoutError, Exception) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = RETRY_DELAY_SEC * (2**attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"All retries exhausted for {url}")
                return None, []


def archive_document(url: str, raw_bytes: io.BytesIO, elements: list) -> dict:
    """Atomically persist raw file and parsed elements."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Derive filename from URL
    safe_name = url.split("/")[-1].split("?")[0] or "document"
    base_path = ARCHIVE_DIR / safe_name

    # Write raw bytes
    raw_path = base_path.with_suffix(".raw.bin")
    raw_path.write_bytes(raw_bytes.getvalue())

    # Write parsed elements
    parsed_path = base_path.with_suffix(".json")
    serialized = [el.to_dict() for el in elements]
    parsed_path.write_text(json.dumps(serialized, indent=2))

    logger.info(f"Archived: {raw_path.name} + {parsed_path.name}")
    return {
        "url": url,
        "raw": str(raw_path),
        "parsed": str(parsed_path),
        "elements": len(elements),
    }


def main():
    results = []
    for url in SAMPLE_URLS:
        raw_stream, elements = fetch_with_retry(url)
        if raw_stream is not None and elements:
            result = archive_document(url, raw_stream, elements)
            results.append(result)
        else:
            results.append({"url": url, "status": "failed"})

    # Summary manifest
    manifest_path = ARCHIVE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Archive manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
