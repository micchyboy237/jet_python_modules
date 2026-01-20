#!/usr/bin/env bash
# _start_playwright_mcp.sh

# Automatically get directory where THIS script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="$SCRIPT_DIR/generated"

echo "Output base : $OUTPUT_BASE"

npx @playwright/mcp@latest \
  --port 8931 \
  --output-dir="$OUTPUT_BASE" \
  --save-video=1280x720 \
  --viewport-size=1280x720
  # --headless
  # --shared-browser-context \
  # --save-trace \
  # --save-session \