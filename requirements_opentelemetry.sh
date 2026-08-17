pip list --format=columns | grep -E "opentelemetry|arize|phoenix" > requirements_opentelemetry_packages.md || true && \
pip list --format=freeze | grep -E "opentelemetry|arize|phoenix" > requirements_opentelemetry_packages.txt || true && \
pip list --format=json | python3 -c "
import json, sys
pkgs = [p for p in json.load(sys.stdin) if any(k in p['name'].lower() for k in ['opentelemetry', 'arize', 'phoenix'])]
print(json.dumps(pkgs, indent=2))
" > requirements_opentelemetry_packages.json
