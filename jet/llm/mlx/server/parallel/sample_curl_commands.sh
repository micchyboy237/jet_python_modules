#!/bin/zsh

# Sample curl commands for MLX server endpoints
# Assumes server is running at http://localhost:9000

# 1. /generate (Streaming Text Generation)
echo "\n(Streaming Text Generation)"
curl -X POST "http://localhost:9000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b-instruct-4bit",
    "prompts": ["Write a short poem about the stars", "Describe a futuristic city"],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "repetition_context_size": 20,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "logit_bias": null,
    "logprobs": -1,
    "stop": ["\n\n"],
    "verbose": true,
    "worker_verbose": true,
    "task_id": "text-001"
  }'

# 2. /chat (Streaming Chat Generation)
echo "\n(Streaming Chat Generation)"
curl -X POST "http://localhost:9000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b-instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about the history of AI."}
    ],
    "max_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "repetition_context_size": 15,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "logit_bias": null,
    "logprobs": -1,
    "stop": ["###"],
    "role_mapping": {"system": "System", "user": "User", "assistant": "Assistant"},
    "tools": [{"type": "function", "function": {"name": "search", "description": "Search the web"}}],
    "system_prompt": "Provide concise and accurate answers.",
    "verbose": true,
    "worker_verbose": true,
    "task_id": "chat-001",
    "session_id": "session-123"
  }'

# 3. /generate_non_stream (Non-Streaming Text Generation)
echo "\n(Non-Streaming Text Generation)"
curl -X POST "http://localhost:9000/generate_non_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b-instruct-4bit",
    "prompts": ["Explain quantum computing in simple terms"],
    "max_tokens": 200,
    "temperature": 0.5,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "repetition_context_size": 10,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "logit_bias": null,
    "logprobs": -1,
    "stop": null,
    "verbose": true,
    "worker_verbose": true,
    "task_id": "text-002"
  }'

# 4. /chat_non_stream (Non-Streaming Chat Generation)
echo "\n(Non-Streaming Chat Generation)"
curl -X POST "http://localhost:9000/chat_non_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b-instruct-4bit",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": null,
    "repetition_context_size": 20,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "logit_bias": null,
    "logprobs": -1,
    "stop": null,
    "role_mapping": null,
    "tools": null,
    "system_prompt": null,
    "verbose": true,
    "worker_verbose": true,
    "task_id": "chat-002",
    "session_id": "session-456"
  }'

# 5. /health (Health Check)
echo "\n(Health Check)"
curl -X GET "http://localhost:9000/health" \
  -H "Accept: application/json"