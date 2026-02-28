import json
from pathlib import Path

# Cache schemas and patterns for maximum efficiency
class ExtractionCache:
    def __init__(self):
        self.schemas = {}
        self.patterns = {}
    def get_schema(self, site_name):
        if site_name not in self.schemas:
            schema_file = Path(f"./cache/{site_name}_schema.json")
            if schema_file.exists():
                self.schemas[site_name] = json.load(schema_file.open())
        return self.schemas.get(site_name)
    def save_schema(self, site_name, schema):
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        schema_file = cache_dir / f"{site_name}_schema.json"
        json.dump(schema, schema_file.open("w"), indent=2)
        self.schemas[site_name] = schema

cache = ExtractionCache()
# Usage: see docs or previous examples

# Optimize selectors for speed
fast_schema = {
    "name": "Optimized Extraction",
    "baseSelector": "#products > .product",
    "fields": [
        {"name": "title", "selector": "> h3", "type": "text"},
        {"name": "price", "selector": ".price:first-child", "type": "text"}
    ]
}

# Avoid slow selectors
slow_schema = {
    "baseSelector": "div div div .product",
    "fields": [
        {"selector": "* h3", "type": "text"}
    ]
}
