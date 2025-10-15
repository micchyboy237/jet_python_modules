# Protobuf Examples

Demonstrates Protocol Buffers v32.0+ usage with proto3 and edition 2024 syntax.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download `protoc` from [GitHub releases](https://github.com/protocolbuffers/protobuf/releases)

3. Generate Python files:

   ```bash
   # Mac/Linux
   chmod +x generate_protos.sh
   ./generate_protos.sh

   # Windows
   generate_protos.bat
   ```

## Usage

```python
# Run examples
python -m src.search_example.demo_search_workflow()
python -m src.book_example.demo_book_operations()

# Run tests
pytest tests/
```

## Structure

- `proto/` - `.proto` schema files
- `generated/` - Auto-generated Python classes
- `src/` - Example usage and utilities
- `tests/` - pytest unit tests

## Notes

- Use `--experimental_allow_proto3_optional` for older protoc versions
- Generated files should be gitignored in production
- For M1 Mac: Ensure ARM-compatible protoc binary

````

## Usage Instructions

1. **Setup**: `pip install -r requirements.txt`
2. **Generate**: Run `generate_protos.sh` (Mac) or `.bat` (Windows)
3. **Run Examples**:
   ```bash:disable-run
   python -m src.search_example.demo_search_workflow()
   python -m src.book_example.demo_book_operations()
````

4. **Test**: `pytest tests/ -v`

The structure is modular, testable, and follows Python best practices with type hints and pytest fixtures. Generated files are separated to avoid manual editing and support CI/CD workflows.
