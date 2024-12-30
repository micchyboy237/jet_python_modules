#!/bin/zsh

export PYTHONPATH="$PYTHONPATH"

# Start the uvicorn server
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
