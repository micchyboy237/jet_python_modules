import concurrent.futures
from typing import List, Dict, Any
from ollama import Client
from jet._token.token_utils import token_counter, truncate_texts
from jet.logger import logger

def generate_streamed_response(client: Client, model: str, prompt: str, max_tokens: int = 4096) -> List[Dict[str, Any]]:
    """
    Generate a streamed response for a single prompt, truncating if necessary.
    
    Args:
        client: Ollama client instance
        model: Model name (e.g., 'gemma3')
        prompt: Input prompt text
        max_tokens: Maximum token limit for the prompt
    
    Returns:
        List of response chunks
    """
    token_count = token_counter([prompt], model, prevent_total=True)[0]
    if token_count > max_tokens:
        logger.warning(f"Prompt exceeds token limit ({token_count} > {max_tokens}): {prompt[:50]}...")
        prompt = truncate_texts([prompt], model, max_tokens)[0]
        logger.debug(f"Truncated prompt to {token_counter([prompt], model, prevent_total=True)[0]} tokens")
    
    chunks = []
    try:
        for part in client.generate(model, prompt, stream=True, options={"num_ctx": max_tokens}):
            chunks.append(part)
        return chunks
    except Exception as e:
        logger.error(f"Error generating response for prompt '{prompt[:50]}...': {e}")
        raise

def parallel_generate(client: Client, model: str, prompts: List[str], max_parallel: int = 3, max_tokens: int = 4096):
    """
    Generate responses for multiple prompts in parallel with streamed output.
    
    Args:
        client: Ollama client instance
        model: Model name (e.g., 'gemma3')
        prompts: List of input prompts
        max_parallel: Maximum number of parallel requests
        max_tokens: Maximum token limit per prompt
    """
    print(f"\nGenerating responses for {len(prompts)} prompts using {model} with up to {max_parallel} parallel calls...\n")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all prompts for parallel execution
        future_to_prompt = {
            executor.submit(generate_streamed_response, client, model, prompt, max_tokens): (i, prompt)
            for i, prompt in enumerate(prompts[:max_parallel])  # Limit to max_parallel
        }
        
        # Process streamed responses as they arrive
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt_idx, prompt = future_to_prompt[future]
            prompt_summary = prompt[:30].replace("\n", " ") + ("..." if len(prompt) > 30 else "")
            try:
                chunks = future.result()
                for part in chunks:
                    response_chunk = part.get('response', '')
                    if response_chunk:
                        print(f"[Prompt {prompt_idx + 1}: {prompt_summary}] {response_chunk}", end='', flush=True)
                print("\n" + "-"*50)  # Separator after each prompt's response
            except Exception as e:
                logger.error(f"Failed to process prompt {prompt_idx + 1} ({prompt_summary}): {e}")

if __name__ == '__main__':
    client = Client(host='http://localhost:11434')  # Updated to match your Ollama server
    model = "llama3.2"
    prompts = [
        "Why is the sky blue?",
        "What causes rainbows to form?",
        "How do clouds affect weather patterns?"
    ]
    parallel_generate(client, model, prompts, max_parallel=2, max_tokens=512)
