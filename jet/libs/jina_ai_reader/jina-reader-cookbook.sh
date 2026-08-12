#!/usr/bin/env bash
# =============================================================================
# Jina Reader Local Cookbook — All Recipes with Output Saving
# Target: local self-hosted instance on HTTP/1.1 port 3001
# Usage:  chmod +x jina-reader-cookbook.sh && ./jina-reader-cookbook.sh
# =============================================================================

set -euo pipefail

BASE="http://localhost:3001"

# ─── Derive OUTPUT_DIR equivalent to Python's Path(__file__).parent / "generated" / stem ──
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
SCRIPT_STEM="$(basename "$SCRIPT_PATH" .sh)"
OUTPUT_DIR="${SCRIPT_DIR}/generated/${SCRIPT_STEM}"

# Clean and recreate (equivalent to shutil.rmtree + mkdir)
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Color helpers
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

header() {
  echo ""
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}▶ $1${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

note() {
  echo -e "${YELLOW}⚠ $1${NC}"
}

# Save curl output to file and print status
# Usage: save <filename> curl [args...]
save() {
  local outfile="$1"
  shift
  local filepath="${OUTPUT_DIR}/${outfile}"
  echo -e "\n${CYAN}\$ $* > ${outfile}${NC}"
  "$@" > "$filepath" 2>&1 || true
  if [[ -s "$filepath" ]]; then
    echo -e "${GREEN}  ✓ Saved $(wc -c < "$filepath" | tr -d ' ') bytes → ${outfile}${NC}"
  else
    echo -e "${YELLOW}  ✗ Empty or failed → ${outfile}${NC}"
  fi
}

echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"

# ─── 0. Health Check ──────────────────────────────────────────────────────────
header "0. Health Check"
save "00-health-check.txt" \
  curl -s -w "\nHTTP %{http_code} from ${BASE}\n" "${BASE}/https://example.com"

# ─── 1. Using Presets ─────────────────────────────────────────────────────────
header "1. Using Presets"

save "01-preset-index.md" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'x-preset: index'

save "01-preset-index-override-links.md" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'x-preset: index' \
    -H 'x-retain-links: all'

# ─── 2. RAG Inference ─────────────────────────────────────────────────────────
header "2. RAG Inference"

save "02-rag-default.md" \
  curl -s "${BASE}/https://example.com/article"

note "VLM alt-text requires OPENROUTER_API_KEY or equivalent env var"
save "02-rag-vlm-alt.md" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'x-with-generated-alt: true'

# ─── 3. Semantic Indexing ─────────────────────────────────────────────────────
header "3. Semantic Indexing"

save "03-index-h3-chunks.json" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'Accept: application/json' \
    -H 'x-retain-links: text' \
    -H 'x-retain-images: alt' \
    -H 'x-markdown-chunking: h3'

save "03-index-s3-chunks.json" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'Accept: application/json' \
    -H 'x-retain-links: text' \
    -H 'x-retain-images: alt' \
    -H 'x-markdown-chunking: s3'

# ─── 4. Deep Research ─────────────────────────────────────────────────────────
header "4. Deep Research"

save "04-research-link-footer.md" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'x-retain-links: text' \
    -H 'x-with-links-summary: true' \
    -H 'x-retain-images: alt'

save "04-research-gpt-oss-citations.md" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'x-retain-links: gpt-oss' \
    -H 'x-retain-images: alt'

# ─── 5. Visual Snapshot / Pageshot ────────────────────────────────────────────
header "5. Visual Snapshot / Pageshot"

save "05-pageshot-full.png" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'x-respond-with: pageshot' \
    -H 'x-remove-overlay: true' \
    -H 'x-timeout: 30'

save "05-screenshot-viewport.png" \
  curl -s "${BASE}/https://example.com/article" \
    -H 'x-respond-with: screenshot' \
    -H 'x-remove-overlay: true' \
    -H 'x-timeout: 30'

# ─── 6. Scrape Known Template ─────────────────────────────────────────────────
header "6. Scrape Known Template"

save "06-target-selector-body.md" \
  curl -s "${BASE}/https://example.com/blog/post-slug" \
    -H 'x-target-selector: article.post-body' \
    -H 'x-remove-selector: nav, .related-posts, .comments, footer'

# ─── 7. Inject Page Script ────────────────────────────────────────────────────
header "7. Inject Page Script"

save "07-youtube-transcript.json" \
  curl -s -X POST "${BASE}/" \
    -F 'url=https://www.youtube.com/watch?v=dQw4w9WgXcQ' \
    -F "injectPageScript=waitForSelector('ytd-video-description-transcript-section-renderer button').then((el) => el.click())" \
    -H 'Accept: application/json'

save "07-multi-script-inject.json" \
  curl -s -X POST "${BASE}/" \
    -F 'url=https://example.com/page' \
    -F "injectPageScript=waitForSelector('.expand-btn').then((el) => el.click())" \
    -F "injectPageScript=waitForSelector('.hidden-content').then((el) => el.scrollIntoView())" \
    -H 'x-timeout: 30' \
    -H 'Accept: application/json'

# ─── 8. Iframes and Shadow DOM ────────────────────────────────────────────────
header "8. Iframes and Shadow DOM"

save "08-iframe-shadow-dom.md" \
  curl -s "${BASE}/https://example.com/docs-page" \
    -H 'x-with-iframe: true' \
    -H 'x-with-shadow-dom: true' \
    -H 'x-timeout: 60'

save "08-iframe-quoted-shadow-dom.md" \
  curl -s "${BASE}/https://example.com/docs-page" \
    -H 'x-with-iframe: quoted' \
    -H 'x-with-shadow-dom: true' \
    -H 'x-timeout: 60'

# ─── 9. Geo- and Locale-Sensitive Scraping ────────────────────────────────────
header "9. Geo- and Locale-Sensitive Scraping"
note "Requires premium Jina API key for x-proxy; will skip silently without one"

save "09-geo-de-locale.md" \
  curl -s "${BASE}/https://shop.example.com/product/123" \
    -H 'x-proxy: de' \
    -H 'x-locale: de-DE' \
    -H 'x-set-cookie: country=DE; Path=/'

# ─── 10. PDF and Office File Uploads ──────────────────────────────────────────
header "10. PDF and Office File Uploads"
note "Office uploads require LibreOffice installed locally"

if [[ ! -f ./test-report.pdf ]]; then
  note "Creating dummy test-report.pdf for demonstration"
  echo "%PDF-1.4 dummy" > ./test-report.pdf
fi

save "10-pdf-upload-chunked.json" \
  curl -s -X POST "${BASE}/" \
    -F 'file=@./test-report.pdf' \
    -H 'Accept: application/json' \
    -H 'x-markdown-chunking: s3'

save "10-pdf-page-1.json" \
  curl -s -X POST "${BASE}/" \
    -F 'file=@./test-report.pdf' \
    -F 'page=1' \
    -H 'Accept: application/json'

# ─── 11. Raw HTML Upload ──────────────────────────────────────────────────────
header "11. Raw HTML Upload"

save "11-html-upload-with-url.md" \
  curl -s -X POST "${BASE}/" \
    -H 'Content-Type: application/json' \
    -d '{"html": "<html><body><h1>Hello</h1><p>World</p></body></html>", "url": "https://example.com/source"}'

save "11-html-upload-standalone.md" \
  curl -s -X POST "${BASE}/" \
    -H 'Content-Type: application/json' \
    -d '{"html": "<html><body><article>Standalone content</article></body></html>"}'

# ─── Summary ───────────────────────────────────────────────────────────────────
header "✅ All Cookbook Recipes Complete"
echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"
echo -e "${GREEN}Files generated:  $(find "$OUTPUT_DIR" -type f | wc -l | tr -d ' ')${NC}"
echo ""
ls -lhS "$OUTPUT_DIR"
echo ""