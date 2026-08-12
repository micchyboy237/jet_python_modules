#!/usr/bin/env bash
# =============================================================================
# Jina Reader Local Cookbook — All Recipes as a Single Shell Script
# Target: local self-hosted instance on HTTP/1.1 port 3001
# Usage:  chmod +x jina-reader-cookbook.sh && ./jina-reader-cookbook.sh
# =============================================================================

set -euo pipefail

BASE="http://localhost:3001"

# Color helpers for readable output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

header() {
  echo ""
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}▶ $1${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

note() {
  echo -e "${YELLOW}⚠ $1${NC}"
}

run() {
  echo -e "\n${CYAN}\$ $*${NC}"
  "$@" || true
}

# ─── Health Check ──────────────────────────────────────────────────────────────
header "0. Health Check"
run curl -s -o /dev/null -w "HTTP %{http_code} from ${BASE}\n" "${BASE}/https://example.com"

# ─── 1. Using Presets ─────────────────────────────────────────────────────────
header "1. Using Presets"

run curl -s "${BASE}/https://example.com/article" \
  -H 'x-preset: index'

run curl -s "${BASE}/https://example.com/article" \
  -H 'x-preset: index' \
  -H 'x-retain-links: all'

# ─── 2. RAG Inference (User Sees What LLM Sees) ──────────────────────────────
header "2. RAG Inference"

run curl -s "${BASE}/https://example.com/article"

note "VLM alt-text requires OPENROUTER_API_KEY or equivalent env var"
run curl -s "${BASE}/https://example.com/article" \
  -H 'x-with-generated-alt: true'

# ─── 3. Semantic Indexing (URLs Are Noise) ────────────────────────────────────
header "3. Semantic Indexing"

run curl -s "${BASE}/https://example.com/article" \
  -H 'Accept: application/json' \
  -H 'x-retain-links: text' \
  -H 'x-retain-images: alt' \
  -H 'x-markdown-chunking: h3'

run curl -s "${BASE}/https://example.com/article" \
  -H 'Accept: application/json' \
  -H 'x-retain-links: text' \
  -H 'x-retain-images: alt' \
  -H 'x-markdown-chunking: s3'

# ─── 4. Deep Research (URLs Needed, But Only Once) ───────────────────────────
header "4. Deep Research"

run curl -s "${BASE}/https://example.com/article" \
  -H 'x-retain-links: text' \
  -H 'x-with-links-summary: true' \
  -H 'x-retain-images: alt'

run curl -s "${BASE}/https://example.com/article" \
  -H 'x-retain-links: gpt-oss' \
  -H 'x-retain-images: alt'

# ─── 5. Visual Snapshot / Pageshot ────────────────────────────────────────────
header "5. Visual Snapshot / Pageshot"

run curl -s "${BASE}/https://example.com/article" \
  -H 'x-respond-with: pageshot' \
  -H 'x-remove-overlay: true' \
  -H 'x-timeout: 30'

run curl -s "${BASE}/https://example.com/article" \
  -H 'x-respond-with: screenshot' \
  -H 'x-remove-overlay: true' \
  -H 'x-timeout: 30'

# ─── 6. Scrape a Known Template (Article Body Only) ──────────────────────────
header "6. Scrape Known Template"

run curl -s "${BASE}/https://example.com/blog/post-slug" \
  -H 'x-target-selector: article.post-body' \
  -H 'x-remove-selector: nav, .related-posts, .comments, footer'

# ─── 7. Inject Page Script (Click-to-Reveal) ─────────────────────────────────
header "7. Inject Page Script"

run curl -s -X POST "${BASE}/" \
  -F 'url=https://www.youtube.com/watch?v=dQw4w9WgXcQ' \
  -F "injectPageScript=waitForSelector('ytd-video-description-transcript-section-renderer button').then((el) => el.click())" \
  -H 'Accept: application/json'

run curl -s -X POST "${BASE}/" \
  -F 'url=https://example.com/page' \
  -F "injectPageScript=waitForSelector('.expand-btn').then((el) => el.click())" \
  -F "injectPageScript=waitForSelector('.hidden-content').then((el) => el.scrollIntoView())" \
  -H 'x-timeout: 30' \
  -H 'Accept: application/json'

# ─── 8. Iframes and Shadow DOM ────────────────────────────────────────────────
header "8. Iframes and Shadow DOM"

run curl -s "${BASE}/https://example.com/docs-page" \
  -H 'x-with-iframe: true' \
  -H 'x-with-shadow-dom: true' \
  -H 'x-timeout: 60'

run curl -s "${BASE}/https://example.com/docs-page" \
  -H 'x-with-iframe: quoted' \
  -H 'x-with-shadow-dom: true' \
  -H 'x-timeout: 60'

# ─── 9. Geo- and Locale-Sensitive Scraping ────────────────────────────────────
header "9. Geo- and Locale-Sensitive Scraping"
note "Requires premium Jina API key for x-proxy; will skip silently without one"

run curl -s "${BASE}/https://shop.example.com/product/123" \
  -H 'x-proxy: de' \
  -H 'x-locale: de-DE' \
  -H 'x-set-cookie: country=DE; Path=/'

# ─── 10. PDF and Office File Uploads ──────────────────────────────────────────
header "10. PDF and Office File Uploads"
note "Office uploads require LibreOffice installed locally"

# Create a tiny test PDF if none exists
if [[ ! -f ./test-report.pdf ]]; then
  note "Creating dummy test-report.pdf for demonstration"
  echo "%PDF-1.4 dummy" > ./test-report.pdf
fi

run curl -s -X POST "${BASE}/" \
  -F 'file=@./test-report.pdf' \
  -H 'Accept: application/json' \
  -H 'x-markdown-chunking: s3'

run curl -s -X POST "${BASE}/" \
  -F 'file=@./test-report.pdf' \
  -F 'page=1' \
  -H 'Accept: application/json'

# ─── 11. Raw HTML Upload ──────────────────────────────────────────────────────
header "11. Raw HTML Upload"

run curl -s -X POST "${BASE}/" \
  -H 'Content-Type: application/json' \
  -d '{"html": "<html><body><h1>Hello</h1><p>World</p></body></html>", "url": "https://example.com/source"}'

run curl -s -X POST "${BASE}/" \
  -H 'Content-Type: application/json' \
  -d '{"html": "<html><body><article>Standalone content</article></body></html>"}'

# ─── Done ──────────────────────────────────────────────────────────────────────
header "✅ All Cookbook Recipes Complete"
echo ""