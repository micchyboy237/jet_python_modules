# Playwright-MCP Demos

Educational, progressive examples showing how to use **Playwright-MCP** tools.

## Structure

```
01_basic_navigation.py           ← start here
02_interaction_basics.py
...
10_error_handling_and_debug.py
└── utils/
    ├── __init__.py
    ├── demo_helpers.py          ← reusable step(), status, error printing
    └── starting_points.py       ← good demo websites
```

## Recommended order

1. Basic navigation
2. Click / type / hover / select
3. Forms & file upload
4. Waiting strategies
...

## Running a demo

```bash
uv run python 01_basic_navigation.py
# or
poetry run python 01_basic_navigation.py
# or just
python -m asyncio 01_basic_navigation.py
```

Most demos start on https://www.saucedemo.com/ — change in `main()` if desired.

Happy automating!
