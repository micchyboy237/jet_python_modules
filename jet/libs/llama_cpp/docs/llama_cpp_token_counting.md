# llama.cpp Token Counting

## Overview

This document explains the key technical differences between the two token counting endpoints provided by llama.cpp: `/v1/chat/completions/input_tokens` and `/v1/responses/input_tokens`.

## Key Technical Differences

### `/v1/chat/completions/input_tokens`

- Applies the chat template (e.g., `<|im_start|>`, `<|im_end|>` for Qwen)
- Counts special tokens added by the template
- Includes tool definition tokens in the count
- More accurate for actual generation scenarios

### `/v1/responses/input_tokens`

- Direct tokenization of raw text
- No chat template applied
- Faster for simple cases
- May give different counts than chat endpoint for the same text

## Usage Examples

### Chat Completions Input Tokens

This endpoint counts tokens with chat template and tool definitions applied:

```bash
curl -X POST "${LLAMA_CPP_LLM_HOST}/v1/chat/completions/input_tokens" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${LLAMA_CPP_LLM_MODEL}"'",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": { "type": "string" }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

**Expected Output:**

```json
{ "input_tokens": 278, "object": "response.input_tokens" }
```

### Responses Input Tokens

This endpoint counts tokens from raw text input without any template:

```bash
curl -X POST "${LLAMA_CPP_LLM_HOST}/v1/responses/input_tokens" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${LLAMA_CPP_LLM_MODEL}"'",
    "input": "Hello, how are you?"
  }'
```

**Expected Output:**

```json
{ "input_tokens": 16, "object": "response.input_tokens" }
```

## When to Use Which Endpoint

- Use `/v1/chat/completions/input_tokens` when you need accurate token counts for chat completion requests, especially when tools are involved
- Use `/v1/responses/input_tokens` for quick, simple token counts without chat template overhead
