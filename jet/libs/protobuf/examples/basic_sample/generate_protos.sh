#!/bin/bash
# Generate Python protobuf files

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "protoc not found. Install from: https://github.com/protocolbuffers/protobuf/releases"
    exit 1
fi

mkdir -p src/generated

# Generate search.proto
protoc --python_out=src/generated proto/search.proto
protoc --python_out=src/generated proto/book.proto

echo "Generated protobuf files in 'src/generated/' directory"
echo "Run 'python -m pytest tests/' to run tests"