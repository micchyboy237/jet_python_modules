@echo off
REM Generate Python protobuf files

REM Check if protoc is available
protoc --version >nul 2>&1
if errorlevel 1 (
    echo protoc not found. Download from: https://github.com/protocolbuffers/protobuf/releases
    pause
    exit /b 1
)

if not exist generated mkdir generated

REM Generate search.proto
protoc --python_out=generated proto\search.proto

REM Generate book.proto
protoc --experimental_allow_proto3_optional --python_out=generated proto\book.proto

echo Generated protobuf files in 'generated\' directory
echo Run 'pytest tests\' to run tests
pause