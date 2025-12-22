#!/bin/bash
# =============================================================================
# Download, extract, and convert OPUS-MT ja→en model to CTranslate2 format
# =============================================================================

set -euo pipefail          # Exit on error, undefined vars, and pipeline failures

# File names
ZIP_FILE="opus-2019-12-18.zip"
MODEL_DIR="."
OUTPUT_DIR="opus-ja-en-ct2"

echo "Starting OPUS-MT ja→en model conversion..."

# 1. Download the model zip
echo "Downloading model..."
curl -L -o "$ZIP_FILE" "https://object.pouta.csc.fi/OPUS-MT-models/ja-en/opus-2019-12-18.zip"

# 2. Extract the zip
echo "Extracting archive..."
unzip -q -o "$ZIP_FILE" -d "$MODEL_DIR"

# 3. Convert to CTranslate2 format
echo "Converting to CTranslate2 format..."
ct2-opus-mt-converter --model_dir "$MODEL_DIR" --output_dir "$OUTPUT_DIR"

# 4. Clean up: delete the zip file
echo "Cleaning up..."
rm -f "$ZIP_FILE"

echo ""
echo "Done! Model is ready in: $OUTPUT_DIR"
echo "You can now use it with CTranslate2."